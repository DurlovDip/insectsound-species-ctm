#!/usr/bin/env python3
"""Small CTM-style head on frozen mn10_as embeddings.

Backbone is not touched. Embeddings are reused. Architecture is chosen on
validation macro F1 of seed 0 only. The frozen test set is scored after that
choice, for seeds 0, 1, and 2 of the selected head.
"""
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, recall_score
from sklearn.preprocessing import StandardScaler

ROOT = "/workspace/insect-sound"
MANIFEST = os.path.join(ROOT, "manifest_quarter.json")
EMB_PATH = os.path.join(ROOT, "embeddings_mn10_as.f32")
PROG_PATH = os.path.join(ROOT, "embeddings_mn10_as_progress.json")
PRETRAIN_METRICS = os.path.join(ROOT, "metrics_pretrain.json")
QUARTER_METRICS = os.path.join(ROOT, "metrics_quarter.json")
METRICS_PATH = os.path.join(ROOT, "metrics_pretrain_ctm.json")
REPORT_PATH = os.path.join(ROOT, "PROJECT-PRETRAIN-CTM.md")

SPECIES = [
    "Aphidoletes_aphidimyza", "Bombus_terrestris", "Bradysia_difformis",
    "Coccinella_septempunctata", "Episyrphus_balteatus", "Halyomorpha_halys",
    "Myzus_persicae", "Nezara_viridula", "Palomena_prasina",
    "Rhaphigaster_nebulos", "Trialeurodes_vaporariorum", "Tuta_absoluta",
]
LAB = {s: i for i, s in enumerate(SPECIES)}

# Locked before any test score. Seed 0 only. Ties: fewer parameters, then fewer ticks.
CONFIGS = [
    {"name": "small", "n_tokens": 12, "d_model": 32, "d_in": 32, "memory": 4, "ticks": 4, "k_sync": 8},
    {"name": "mid", "n_tokens": 16, "d_model": 48, "d_in": 40, "memory": 8, "ticks": 6, "k_sync": 10},
    {"name": "wide", "n_tokens": 16, "d_model": 64, "d_in": 48, "memory": 8, "ticks": 6, "k_sync": 12},
]
SEEDS = [0, 1, 2]
MAX_EPOCHS = 30
PATIENCE = 5
LR = 1e-3
BATCH = 256
MLP_ACC_TARGET = None  # filled from metrics_pretrain.json, not typed in


def macro_f1(y, p):
    return float(f1_score(y, p, average="macro", labels=list(range(12)), zero_division=0))


def recalls(y, p):
    r = recall_score(y, p, average=None, labels=list(range(12)), zero_division=0)
    return {SPECIES[i]: float(r[i]) for i in range(12)}


def load_split():
    prog = json.load(open(PROG_PATH))
    if prog["n_done"] != 41495 or prog["dim"] != 960 or prog["n"] != 41495:
        raise RuntimeError(f"embedding progress is not a full 960-d run: {prog}")
    nbytes = os.path.getsize(EMB_PATH)
    expect = 41495 * 960 * 4
    if nbytes != expect:
        raise RuntimeError(f"embedding file bytes {nbytes} != {expect}")
    emb = np.memmap(EMB_PATH, dtype=np.float32, mode="r", shape=(41495, 960))
    man = json.load(open(MANIFEST))
    files = man["files"]
    if len(files) != 41495:
        raise RuntimeError(f"manifest {len(files)}")
    y = np.empty(len(files), np.int64)
    split = []
    recs = []
    for i, r in enumerate(files):
        y[i] = LAB[r["species"]]
        split.append(r["split"])
        recs.append(r["recording_id"])
    split = np.asarray(split)
    by = {}
    for s, rid in zip(split.tolist(), recs):
        by.setdefault(rid, set()).add(s)
    if any(len(ss) > 1 for ss in by.values()):
        raise RuntimeError("recording id overlap")
    counts = {s: int((split == s).sum()) for s in ("train", "val", "test")}
    if counts != {"train": 28426, "val": 6226, "test": 6843}:
        raise RuntimeError(counts)
    if not np.isfinite(emb).all():
        raise RuntimeError("non-finite embeddings")
    return np.array(emb), y, split, counts


class SuperLinear(nn.Module):
    def __init__(self, in_dims, out_dims, n):
        super().__init__()
        lim = 1.0 / math.sqrt(in_dims + out_dims)
        self.w = nn.Parameter(torch.empty(in_dims, out_dims, n).uniform_(-lim, lim))
        self.b = nn.Parameter(torch.zeros(1, n, out_dims))
        self.temp = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return (torch.einsum("bnm,mhn->bnh", x, self.w) + self.b) / self.temp.clamp(min=0.05)


class CTMHead(nn.Module):
    """Same tick / private-history / sync readout as the spectrogram CTM-style model.

    The conv stem is gone. The frozen 960-d vector is split into contiguous
    tokens and those tokens are the attention memory. Neurons still recur for
    several internal ticks.
    """

    def __init__(self, n_tokens, d_model, d_in, memory, ticks, k_sync, d_emb=960):
        super().__init__()
        if d_emb % n_tokens != 0:
            raise RuntimeError("n_tokens must divide 960")
        if k_sync > d_model:
            raise RuntimeError("k_sync > d_model")
        self.n_tokens = n_tokens
        self.token_dim = d_emb // n_tokens
        self.d_model = d_model
        self.memory = memory
        self.ticks = ticks
        self.k = k_sync
        self.token_proj = nn.Linear(self.token_dim, d_in)
        self.synapses = nn.Sequential(
            nn.Linear(d_in + d_model, 2 * d_model), nn.GLU(dim=-1), nn.LayerNorm(d_model),
        )
        self.nlm = SuperLinear(memory, 2, d_model)
        self.attn = nn.MultiheadAttention(d_in, num_heads=1, batch_first=True)
        sync_dim = k_sync * (k_sync + 1) // 2
        self.q_proj = nn.Linear(sync_dim, d_in)
        self.out_proj = nn.Linear(sync_dim, 12)
        self.decay_action = nn.Parameter(torch.zeros(sync_dim))
        self.decay_out = nn.Parameter(torch.zeros(sync_dim))
        self.start_trace = nn.Parameter(torch.empty(d_model, memory).uniform_(-0.1, 0.1))
        self.start_z = nn.Parameter(torch.zeros(d_model))
        idx = torch.triu_indices(k_sync, k_sync)
        self.register_buffer("triu_i", idx[0])
        self.register_buffer("triu_j", idx[1])

    def _sync(self, selected, alpha, beta, r):
        outer = selected.unsqueeze(2) * selected.unsqueeze(1)
        pair = outer[:, self.triu_i, self.triu_j]
        if alpha is None:
            alpha, beta = pair, torch.ones_like(pair)
        else:
            alpha = r * alpha + pair
            beta = r * beta + 1
        return alpha / torch.sqrt(beta), alpha, beta

    def forward(self, x):
        # x: (B, 960) standardized embedding
        B = x.shape[0]
        tok = x.view(B, self.n_tokens, self.token_dim)
        kv = self.token_proj(tok)
        z = self.start_z.unsqueeze(0).expand(B, -1)
        trace = self.start_trace.unsqueeze(0).expand(B, -1, -1).contiguous()
        r_a = torch.exp(-self.decay_action.clamp(0, 15)).unsqueeze(0)
        r_o = torch.exp(-self.decay_out.clamp(0, 15)).unsqueeze(0)
        a_alpha = a_beta = o_alpha = o_beta = None
        logits, certs = [], []
        for _ in range(self.ticks):
            sync_a, a_alpha, a_beta = self._sync(z[:, -self.k:], a_alpha, a_beta, r_a)
            q = self.q_proj(sync_a).unsqueeze(1)
            attn_out, _ = self.attn(q, kv, kv, need_weights=False)
            pre = self.synapses(torch.cat([attn_out.squeeze(1), z], dim=-1))
            trace = torch.cat([trace[:, :, 1:], pre.unsqueeze(-1)], dim=-1)
            z = F.glu(self.nlm(trace), dim=-1)
            if z.dim() == 3:
                z = z.squeeze(-1)
            sync_o, o_alpha, o_beta = self._sync(z[:, :self.k], o_alpha, o_beta, r_o)
            logit = self.out_proj(sync_o)
            prob = F.softmax(logit, dim=-1)
            ent = -(prob * torch.log(prob.clamp_min(1e-8))).sum(-1)
            cert = 1.0 - ent / math.log(12)
            logits.append(logit)
            certs.append(cert)
        return torch.stack(logits, dim=-1), torch.stack(certs, dim=-1)


def ctm_loss(logits, certs, y, class_w):
    T = logits.shape[-1]
    target = y[:, None].expand(-1, T)
    losses = F.cross_entropy(logits, target, weight=class_w, reduction="none")
    b = torch.arange(logits.shape[0], device=logits.device)
    t_best, t_cert = losses.argmin(1), certs.argmax(1)
    return 0.5 * (losses[b, t_best].mean() + losses[b, t_cert].mean())


def predict(logits, certs):
    t = certs.argmax(1)
    b = torch.arange(logits.shape[0])
    chosen = logits[b, :, t]
    return chosen.argmax(1), t


def n_params(model):
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def run_one(cfg, seed, xtr, ytr, xva, yva, class_w):
    torch.manual_seed(seed)
    model = CTMHead(**{k: cfg[k] for k in ("n_tokens", "d_model", "d_in", "memory", "ticks", "k_sync")})
    params = n_params(model)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    xt = torch.from_numpy(xtr)
    yt = torch.from_numpy(ytr)
    xv = torch.from_numpy(xva)
    best, best_f1, best_ep, stall = None, -1.0, -1, 0
    hist = []
    t0 = time.time()
    for ep in range(1, MAX_EPOCHS + 1):
        model.train()
        perm = torch.randperm(xt.shape[0])
        for s in range(0, xt.shape[0], BATCH):
            idx = perm[s:s + BATCH]
            opt.zero_grad(set_to_none=True)
            logits, certs = model(xt[idx])
            loss = ctm_loss(logits, certs, yt[idx], class_w)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            logits, certs = model(xv)
            pred, ticks = predict(logits, certs)
            pred_np = pred.numpy()
        vf1 = macro_f1(yva, pred_np)
        vacc = float((pred_np == yva).mean())
        hist.append({"epoch": ep, "val_acc": vacc, "val_macro_f1": vf1})
        print(f"  {cfg['name']} seed {seed} ep {ep} val_acc {vacc:.4f} val_f1 {vf1:.4f}", flush=True)
        if vf1 > best_f1 + 1e-4:
            best_f1, best_ep, stall = vf1, ep, 0
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stall += 1
            if stall >= PATIENCE:
                break
    model.load_state_dict(best)
    model.eval()
    with torch.no_grad():
        logits, certs = model(xv)
        pred, ticks = predict(logits, certs)
    pred_np = pred.numpy()
    return {
        "model": model,
        "params": params,
        "best_epoch": best_ep,
        "epochs_ran": hist[-1]["epoch"],
        "seconds": time.time() - t0,
        "val_macro_f1": macro_f1(yva, pred_np),
        "val_accuracy": float((pred_np == yva).mean()),
        "val_correct": int((pred_np == yva).sum()),
        "history": hist,
    }


def score_test(model, xte, yte):
    model.eval()
    with torch.no_grad():
        logits, certs = model(torch.from_numpy(xte))
        pred, ticks = predict(logits, certs)
    pred_np = pred.numpy()
    tick_np = ticks.numpy()
    correct = int((pred_np == yte).sum())
    n = int(yte.shape[0])
    return {
        "test_correct": correct,
        "test_n": n,
        "test_accuracy": correct / n,
        "test_macro_f1": macro_f1(yte, pred_np),
        "test_recall": recalls(yte, pred_np),
        "frac_most_certain_is_last": float((tick_np == (model.ticks - 1)).mean()),
        "tick_hist": [int((tick_np == t).sum()) for t in range(model.ticks)],
    }


def main():
    t_all = time.time()
    pre = json.load(open(PRETRAIN_METRICS))
    quarter = json.load(open(QUARTER_METRICS))
    mlp_acc = float(pre["test_accuracy"])
    mlp_correct = int(pre["test_correct"])
    mlp_f1 = float(pre["test_macro_f1"])
    mlp_n = int(pre["test_n"])
    if mlp_n != 6843 or mlp_correct != 3253:
        raise RuntimeError(f"unexpected stored MLP result {mlp_correct}/{mlp_n}")

    emb, y, split, counts = load_split()
    tr, va, te = split == "train", split == "val", split == "test"
    scaler = StandardScaler()
    xtr = scaler.fit_transform(emb[tr]).astype(np.float32)
    xva = scaler.transform(emb[va]).astype(np.float32)
    xte = scaler.transform(emb[te]).astype(np.float32)
    ytr, yva, yte = y[tr], y[va], y[te]
    counts_c = np.bincount(ytr, minlength=12).astype(np.float64)
    class_w = torch.tensor(counts_c.sum() / (12.0 * np.maximum(counts_c, 1.0)), dtype=torch.float32)

    # Selection: seed 0 only. Test tensors are not passed in.
    grid = []
    for cfg in CONFIGS:
        row = run_one(cfg, 0, xtr, ytr, xva, yva, class_w)
        row["cfg"] = cfg
        grid.append(row)
        print(
            f"GRID {cfg['name']} params={row['params']} val_f1={row['val_macro_f1']:.6f} "
            f"val_acc={row['val_accuracy']:.6f}",
            flush=True,
        )

    def rank(row):
        return (-row["val_macro_f1"], row["params"], row["cfg"]["ticks"])

    chosen = min(grid, key=rank)
    print("CHOSEN", chosen["cfg"]["name"], "val_f1", chosen["val_macro_f1"], flush=True)

    # Seeds of the locked config. Seed 0 is the grid run.
    seed_runs = {0: chosen}
    for seed in SEEDS:
        if seed == 0:
            continue
        seed_runs[seed] = run_one(chosen["cfg"], seed, xtr, ytr, xva, yva, class_w)

    # Test once per seed, after the architecture is locked.
    seed_reports = []
    for seed in SEEDS:
        run = seed_runs[seed]
        scored = score_test(run["model"], xte, yte)
        rep = {
            "seed": seed,
            "params": run["params"],
            "best_epoch": run["best_epoch"],
            "epochs_ran": run["epochs_ran"],
            "seconds": run["seconds"],
            "val_macro_f1": run["val_macro_f1"],
            "val_accuracy": run["val_accuracy"],
            "val_correct": run["val_correct"],
            **scored,
        }
        seed_reports.append(rep)
        print(
            f"TEST seed {seed} {scored['test_correct']}/{scored['test_n']} "
            f"acc {scored['test_accuracy']:.6f} f1 {scored['test_macro_f1']:.6f}",
            flush=True,
        )

    accs = [r["test_accuracy"] for r in seed_reports]
    f1s = [r["test_macro_f1"] for r in seed_reports]
    corrects = [r["test_correct"] for r in seed_reports]
    mean_acc = float(np.mean(accs))
    mean_f1 = float(np.mean(f1s))
    beats = bool(mean_acc > mlp_acc)

    grid_public = []
    for row in grid:
        grid_public.append({
            "name": row["cfg"]["name"],
            **row["cfg"],
            "params": row["params"],
            "seed": 0,
            "val_macro_f1": row["val_macro_f1"],
            "val_accuracy": row["val_accuracy"],
            "val_correct": row["val_correct"],
            "best_epoch": row["best_epoch"],
            "epochs_ran": row["epochs_ran"],
            "seconds": row["seconds"],
            "selected": row["cfg"]["name"] == chosen["cfg"]["name"],
        })

    metrics = {
        "backbone": "EfficientAT mn10_as frozen, embeddings reused",
        "embedding_dim": 960,
        "embedding_file": EMB_PATH,
        "subsampled": False,
        "split_counts": counts,
        "recording_id_overlap": 0,
        "standardize": "StandardScaler fit on train embeddings only",
        "head": "CTM-style on contiguous embedding tokens",
        "loss": "0.5 * (CE at lowest-loss tick + CE at highest-certainty tick)",
        "decision": "most certain tick",
        "class_weight": "inverse train frequency",
        "optimizer": {"name": "Adam", "lr": LR, "weight_decay": 1e-4},
        "early_stop": "validation macro F1 of most-certain-tick predictions",
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "selection_rule": "seed 0 only; max val macro F1; ties: fewer parameters, then fewer ticks",
        "grid_seed0": grid_public,
        "selected_config": chosen["cfg"],
        "selected_params": chosen["params"],
        "seeds": seed_reports,
        "mean_test_accuracy": mean_acc,
        "mean_test_macro_f1": mean_f1,
        "mean_test_correct": float(np.mean(corrects)),
        "mlp_test_accuracy": mlp_acc,
        "mlp_test_correct": mlp_correct,
        "mlp_test_macro_f1": mlp_f1,
        "mlp_test_n": mlp_n,
        "beats_mlp_accuracy_0.475376": beats,
        "per_seed_beats_mlp": [bool(a > mlp_acc) for a in accs],
        "baselines": {
            "majority_accuracy": quarter["models"]["majority_test_accuracy"],
            "majority_correct": quarter["models"]["majority_test_correct"],
            "majority_macro_f1": quarter["models"]["majority_test_macro_f1"],
            "linear_accuracy": quarter["models"]["linear_test_accuracy"],
            "linear_correct": quarter["models"]["linear_test_correct"],
            "linear_macro_f1": quarter["models"]["linear_test_macro_f1"],
            "plain_cnn_mean_accuracy": quarter["summary"]["plain_acc"]["mean"],
            "plain_cnn_mean_macro_f1": quarter["summary"]["plain_f1"]["mean"],
            "ctm_spectrogram_mean_accuracy": quarter["summary"]["ctm_acc"]["mean"],
            "ctm_spectrogram_mean_macro_f1": quarter["summary"]["ctm_f1"]["mean"],
        },
        "elapsed_seconds": time.time() - t_all,
    }
    # drop nothing unserializable
    json.dump(metrics, open(METRICS_PATH, "w"), indent=2)

    cfg = chosen["cfg"]
    b = metrics["baselines"]
    lines = []
    a = lines.append
    a("# InsectSound1000 quarter subset, CTM-style head on frozen mn10_as embeddings")
    a("")
    a("## Problem")
    a("")
    a("Same frozen recording-id split and the same frozen EfficientAT `mn10_as` embeddings as PROJECT-PRETRAIN.md. The backbone is not trained. Only a small CTM-style head is trained. This is the same simplified tick model used on log-mel spectrograms in PROJECT-QUARTER.md (private neuron history, synchronization readout, loss at the lowest-loss tick and the most certain tick, test decision at the most certain tick). It is not the full Darlow et al. model.")
    a("")
    a("## Data")
    a("")
    a(f"Embeddings reused from `{EMB_PATH}`: 41495 clips, 960 dimensions, extraction not repeated. Split from `manifest_quarter.json`: train {counts['train']}, val {counts['val']}, test {counts['test']}. Recording-id overlap: 0. `StandardScaler` fit on train embeddings only. No subsampling. Full test n={mlp_n}.")
    a("")
    a("## Head")
    a("")
    a("Each 960-d vector is split into contiguous tokens (no learned audio front-end). Each tick: synchronization of a neuron subset builds an attention query over those tokens; a shared GLU synapse updates a private pre-activation history; a per-neuron SuperLinear+GLU writes the next state; a second synchronization vector is the class logit. Certainty is one minus the normalized softmax entropy. Loss is the mean of cross-entropy at the lowest-loss tick and cross-entropy at the highest-certainty tick. Inverse-frequency class weights. Adam, learning rate 0.001, weight decay 1e-4. Early stop on validation macro F1 of the most-certain-tick prediction, patience 5, at most 30 epochs. CPU.")
    a("")
    a("Architecture chosen on seed 0 validation macro F1 only, before any test score. Ties would go to fewer parameters, then fewer ticks.")
    a("")
    a("| config | tokens | D | d_in | memory | ticks | sync | params | seed 0 val macro F1 | seed 0 val accuracy |")
    a("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for g in grid_public:
        mark = " **" if g["selected"] else ""
        a(f"| {g['name']}{mark} | {g['n_tokens']} | {g['d_model']} | {g['d_in']} | {g['memory']} | {g['ticks']} | {g['k_sync']} | {g['params']} | {g['val_macro_f1']:.6f} | {g['val_accuracy']:.6f} |")
    a("")
    a(f"Selected: **{cfg['name']}**. Head parameters: **{chosen['params']}**. Tokens {cfg['n_tokens']} (width {960 // cfg['n_tokens']}), D={cfg['d_model']}, d_in={cfg['d_in']}, memory {cfg['memory']}, ticks {cfg['ticks']}, sync subset {cfg['k_sync']}.")
    a("")
    a("## Frozen test")
    a("")
    a("Scored only for the selected head, seeds 0, 1, and 2. Seed 0 is the selection run, not a retrain. Not a subsample.")
    a("")
    a("| model | test accuracy | correct | test macro F1 |")
    a("|---|---:|---:|---:|")
    a(f"| Majority | {b['majority_accuracy']:.6f} | {b['majority_correct']}/6843 | {b['majority_macro_f1']:.6f} |")
    a(f"| Linear log-mel | {b['linear_accuracy']:.6f} | {b['linear_correct']}/6843 | {b['linear_macro_f1']:.6f} |")
    a(f"| Plain CNN mean | {b['plain_cnn_mean_accuracy']:.6f} | — | {b['plain_cnn_mean_macro_f1']:.6f} |")
    a(f"| CTM-style on spectrograms, mean | {b['ctm_spectrogram_mean_accuracy']:.6f} | — | {b['ctm_spectrogram_mean_macro_f1']:.6f} |")
    a(f"| Frozen mn10_as + MLP 960→64→12 | {mlp_acc:.6f} | {mlp_correct}/{mlp_n} | {mlp_f1:.6f} |")
    a(f"| Frozen mn10_as + CTM-style head, mean of 3 seeds | {mean_acc:.6f} | — | {mean_f1:.6f} |")
    a("")
    for r in seed_reports:
        a(f"- Seed {r['seed']}: {r['test_correct']}/{r['test_n']} = {r['test_accuracy']:.6f}, macro F1 {r['test_macro_f1']:.6f}. Val macro F1 {r['val_macro_f1']:.6f} at epoch {r['best_epoch']} (ran {r['epochs_ran']}, {r['seconds']:.1f}s). Most-certain tick is the last tick on {r['frac_most_certain_is_last']:.6f} of test clips. Tick counts {r['tick_hist']}. Beats MLP accuracy: **{str(r['test_accuracy'] > mlp_acc)}**.")
    a("")
    if beats:
        a(f"The CTM-style head mean accuracy beats the frozen MLP ({mean_acc:.6f} vs {mlp_acc:.6f}).")
    else:
        a(f"The CTM-style head does not beat the frozen MLP. Mean accuracy {mean_acc:.6f} versus MLP {mlp_acc:.6f} (3253/6843).")
    a("")
    a("### Per-species recall, mean over the 3 seeds")
    a("")
    a("| species | recall mean |")
    a("|---|---:|")
    for sp in SPECIES:
        mrec = float(np.mean([r["test_recall"][sp] for r in seed_reports]))
        a(f"| {sp.replace('_', ' ')} | {mrec:.6f} |")
    a("")
    a(f"Numbers were written by `pretrain_ctm_head.py` from `metrics_pretrain_ctm.json`. Baseline rows were read from `metrics_pretrain.json` and `metrics_quarter.json`. Wall time {metrics['elapsed_seconds']:.1f}s.")
    a("")
    open(REPORT_PATH, "w").write("\n".join(lines))
    print("WROTE", REPORT_PATH, "mean_acc", mean_acc, "beats", beats, flush=True)


if __name__ == "__main__":
    main()
