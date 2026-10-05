#!/usr/bin/env python3
"""Session-holdout InsectSound1000: linear vs plain CNN vs small CTM-style model."""
import json, os, math, time, struct
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, recall_score

ROOT = "/workspace/insect-sound"
RAW = os.path.join(ROOT, "raw")
MANIFEST = os.path.join(ROOT, "manifest.json")
SEED_SPLIT = 0  # tie-break only; assignment is deterministic by name
SEEDS = (0, 1)
N_FFT, HOP, N_MELS, SR = 512, 512, 32, 16000
MAX_EPOCHS, PATIENCE, LR, BATCH = 12, 3, 1e-3, 32
TICKS = 6
N_BOOT = 1000

SPECIES = [
    "Aphidoletes_aphidimyza", "Bombus_terrestris", "Bradysia_difformis",
    "Coccinella_septempunctata", "Episyrphus_balteatus", "Halyomorpha_halys",
    "Myzus_persicae", "Nezara_viridula", "Palomena_prasina",
    "Rhaphigaster_nebulos", "Trialeurodes_vaporariorum", "Tuta_absoluta",
]
LAB = {s: i for i, s in enumerate(SPECIES)}

def read_channel0(path):
    raw = open(path, "rb").read()
    off = 12
    while off + 8 <= len(raw):
        cid = raw[off:off + 4]
        sz = struct.unpack_from("<I", raw, off + 4)[0]
        if cid == b"data":
            data = raw[off + 8:off + 8 + sz]
            break
        off += 8 + sz + (sz & 1)
    else:
        raise RuntimeError("no data " + path)
    x = np.frombuffer(data, dtype="<f4")
    mono = np.array(x.reshape(-1, 4)[:, 0], dtype=np.float32)
    if mono.shape[0] != 40000:
        raise RuntimeError(f"len {mono.shape[0]} {path}")
    return mono

def mel_filterbank(n_fft, n_mels, sr, fmin=20.0, fmax=8000.0):
    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)
    def mel_to_hz(m):
        return 700.0 * (10 ** (m / 2595.0) - 1.0)
    n_freq = n_fft // 2 + 1
    mels = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz = mel_to_hz(mels)
    bins = np.clip(np.floor((n_fft + 1) * hz / sr).astype(int), 0, n_freq - 1)
    fb = np.zeros((n_mels, n_freq), dtype=np.float32)
    for i in range(n_mels):
        left, center, right = int(bins[i]), int(bins[i + 1]), int(bins[i + 2])
        if center <= left:
            center = left + 1
        if right <= center:
            right = center + 1
        for j in range(left, min(center, n_freq)):
            fb[i, j] = (j - left) / (center - left)
        for j in range(center, min(right, n_freq)):
            fb[i, j] = (right - j) / max(right - center, 1)
    return fb

def session_split(manifest):
    """Whole recording-ids only. For each species, test id is the one whose
    clip count is closest to 15% of that species' subset; val is the next
    closest remaining id; every other id is train. Never shares an id across splits.
    """
    groups = defaultdict(lambda: defaultdict(list))
    for rec in manifest:
        sp = rec["species"]
        rid = rec["base"].split(f"_{sp}_")[0]
        groups[sp][rid].append(rec)
    assigned = []
    summary = {}
    for sp in SPECIES:
        ids = [(rid, recs) for rid, recs in groups[sp].items()]
        total = sum(len(recs) for _, recs in ids)
        target = 0.15 * total
        def pick(cands):
            return min(cands, key=lambda it: (abs(len(it[1]) - target), it[0]))
        test_id, test_recs = pick(ids)
        rest = [it for it in ids if it[0] != test_id]
        val_id, val_recs = pick(rest)
        train_items = [it for it in rest if it[0] != val_id]
        buckets = {"test": test_recs, "val": val_recs, "train": []}
        train_ids = []
        for rid, recs in train_items:
            buckets["train"].extend(recs)
            train_ids.append(rid)
        for split, recs in (("test", test_recs), ("val", val_recs), ("train", buckets["train"])):
            for rec in recs:
                q = dict(rec)
                q["split"] = split
                q["recording_id"] = rec["base"].split(f"_{sp}_")[0]
                assigned.append(q)
        summary[sp] = {
            "n": total,
            "train": len(buckets["train"]),
            "val": len(val_recs),
            "test": len(test_recs),
            "train_ids": sorted(train_ids),
            "val_ids": [val_id],
            "test_ids": [test_id],
            "n_ids": len(ids),
        }
    return assigned, summary

def nparams(m):
    return int(sum(p.numel() for p in m.parameters()))

class ConvStem(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, width, 3, padding=1),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.width = width

    def forward(self, x):
        return self.net(x)

class PlainCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = ConvStem(32)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(32 * 16, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 12),
        )

    def forward(self, x):
        return self.head(self.stem(x))

class SuperLinear(nn.Module):
    """Private linear map per neuron over its pre-activation history. Sakana SuperLinear, shallow."""
    def __init__(self, in_dims, out_dims, n):
        super().__init__()
        lim = 1.0 / math.sqrt(in_dims + out_dims)
        self.w = nn.Parameter(torch.empty(in_dims, out_dims, n).uniform_(-lim, lim))
        self.b = nn.Parameter(torch.zeros(1, n, out_dims))
        self.temp = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        # x: B, N, M
        out = torch.einsum("bnm,mhn->bnh", x, self.w) + self.b
        return out / self.temp.clamp(min=0.05)

class CTMStyle(nn.Module):
    """Small CTM-style classifier.

    Included from Darlow et al. (arXiv:2505.05522), scaled down:
      internal ticks decoupled from the clip
      shared synapse MLP producing pre-activations
      FIFO pre-activation history
      privately parameterized neuron-level model (per-neuron weights on that history)
      synchronization (decayed pairwise product of post-activations) as the latent
      sync -> attention query into log-mel tokens, and sync -> class logits
      certainty = 1 - normalized entropy
    Simplified vs the paper / Sakana code:
      synapse is one GLU MLP, not a U-Net
      shallow NLM (one private GLU), not a wide deep NLM
      D, M, T and sync subsets are small
      no learnable Fourier positions, no ResNet, channel 0 log-mel only
      neuron subset is first/last, not random pairing
    """
    def __init__(self, d_model=64, d_in=32, memory=8, ticks=TICKS, k_sync=12):
        super().__init__()
        self.d_model = d_model
        self.memory = memory
        self.ticks = ticks
        self.k = k_sync
        self.stem = ConvStem(32)
        self.token_proj = nn.Linear(32, d_in)
        self.synapses = nn.Sequential(
            nn.Linear(d_in + d_model, 2 * d_model),
            nn.GLU(dim=-1),
            nn.LayerNorm(d_model),
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
            alpha = pair
            beta = torch.ones_like(pair)
        else:
            alpha = r * alpha + pair
            beta = r * beta + 1
        return alpha / torch.sqrt(beta), alpha, beta

    def forward(self, x, return_diag=False):
        h = self.stem(x)
        tok = F.adaptive_avg_pool2d(h, (4, 6))
        tok = tok.flatten(2).transpose(1, 2)
        kv = self.token_proj(tok)
        B = x.shape[0]
        z = self.start_z.unsqueeze(0).expand(B, -1)
        trace = self.start_trace.unsqueeze(0).expand(B, -1, -1).contiguous()
        r_a = torch.exp(-self.decay_action.clamp(0, 15)).unsqueeze(0)
        r_o = torch.exp(-self.decay_out.clamp(0, 15)).unsqueeze(0)
        a_alpha = a_beta = o_alpha = o_beta = None
        logits = []
        certs = []
        att_ent = []
        for _ in range(self.ticks):
            sync_a, a_alpha, a_beta = self._sync(z[:, -self.k:], a_alpha, a_beta, r_a)
            q = self.q_proj(sync_a).unsqueeze(1)
            attn_out, attn_w = self.attn(q, kv, kv, need_weights=return_diag, average_attn_weights=True)
            pre = self.synapses(torch.cat([attn_out.squeeze(1), z], dim=-1))
            trace = torch.cat([trace[:, :, 1:], pre.unsqueeze(-1)], dim=-1)
            z = F.glu(self.nlm(trace), dim=-1).squeeze(-1)
            sync_o, o_alpha, o_beta = self._sync(z[:, :self.k], o_alpha, o_beta, r_o)
            logit = self.out_proj(sync_o)
            prob = F.softmax(logit, dim=-1)
            ent = -(prob * torch.log(prob.clamp_min(1e-8))).sum(-1)
            cert = 1.0 - ent / math.log(12)
            logits.append(logit)
            certs.append(cert)
            if return_diag:
                p = attn_w.squeeze(1).clamp_min(1e-8)
                att_ent.append(-(p * p.log()).sum(-1))
        logits = torch.stack(logits, dim=-1)  # B, C, T
        certs = torch.stack(certs, dim=-1)    # B, T
        if return_diag:
            return logits, certs, torch.stack(att_ent, dim=-1)
        return logits, certs

def ctm_loss(logits, certs, y, class_w):
    # logits B,C,T ; certs B,T
    T = logits.shape[-1]
    target = y[:, None].expand(-1, T)
    losses = F.cross_entropy(logits, target, weight=class_w, reduction="none")
    b = torch.arange(logits.shape[0], device=logits.device)
    t_best = losses.argmin(dim=1)
    t_cert = certs.argmax(dim=1)
    loss = 0.5 * (losses[b, t_best].mean() + losses[b, t_cert].mean())
    return loss, t_cert

def macro_f1(y, p):
    return float(f1_score(y, p, average="macro", labels=list(range(12)), zero_division=0))

def per_recall(y, p):
    r = recall_score(y, p, average=None, labels=list(range(12)), zero_division=0)
    return {SPECIES[i]: float(r[i]) for i in range(12)}

def bootstrap_diff(y, pred_a, pred_b, n_boot=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y)
    diffs = np.empty(n_boot)
    acc_a = np.empty(n_boot)
    acc_b = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yy = y[idx]
        acc_a[i] = (pred_a[idx] == yy).mean()
        acc_b[i] = (pred_b[idx] == yy).mean()
        diffs[i] = acc_a[i] - acc_b[i]
    def ci(a):
        return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    return {
        "n_boot": n_boot,
        "diff_ctm_minus_cnn_mean": float(diffs.mean()),
        "diff_ci95": ci(diffs),
        "includes_zero": bool(ci(diffs)[0] <= 0 <= ci(diffs)[1]),
        "ctm_acc_ci95": ci(acc_a),
        "cnn_acc_ci95": ci(acc_b),
    }

def train_plain(model, loader, xva, yva, class_w, seed):
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    best, best_f1, best_ep, stall = None, -1.0, -1, 0
    hist = []
    t0 = time.time()
    for ep in range(1, MAX_EPOCHS + 1):
        model.train()
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xb), yb, weight=class_w)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(xva).argmax(1).cpu().numpy()
        vf1 = macro_f1(yva.numpy(), pred)
        vacc = float((pred == yva.numpy()).mean())
        hist.append({"epoch": ep, "val_acc": vacc, "val_macro_f1": vf1})
        print(f"  plain s{seed} ep {ep} val_acc {vacc:.3f} val_f1 {vf1:.3f}", flush=True)
        if vf1 > best_f1 + 1e-4:
            best_f1, best_ep, stall = vf1, ep, 0
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stall += 1
            if stall >= PATIENCE:
                break
    model.load_state_dict(best)
    return {"best_epoch": best_ep, "best_val_macro_f1": best_f1, "epochs_ran": hist[-1]["epoch"],
            "history": hist, "seconds": time.time() - t0}

def train_ctm(model, loader, xva, yva, class_w, seed):
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    best, best_f1, best_ep, stall = None, -1.0, -1, 0
    hist = []
    t0 = time.time()
    for ep in range(1, MAX_EPOCHS + 1):
        model.train()
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            logits, certs = model(xb)
            loss, _ = ctm_loss(logits, certs, yb, class_w)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            logits, certs = model(xva)
            pred = logits.argmax(1)[torch.arange(len(yva)), certs.argmax(1)].cpu().numpy()
        vf1 = macro_f1(yva.numpy(), pred)
        vacc = float((pred == yva.numpy()).mean())
        hist.append({"epoch": ep, "val_acc": vacc, "val_macro_f1": vf1})
        print(f"  ctm s{seed} ep {ep} val_acc {vacc:.3f} val_f1 {vf1:.3f}", flush=True)
        if vf1 > best_f1 + 1e-4:
            best_f1, best_ep, stall = vf1, ep, 0
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stall += 1
            if stall >= PATIENCE:
                break
    model.load_state_dict(best)
    return {"best_epoch": best_ep, "best_val_macro_f1": best_f1, "epochs_ran": hist[-1]["epoch"],
            "history": hist, "seconds": time.time() - t0}

def predict_ctm(model, x):
    model.eval()
    logits, certs, att = model(x, return_diag=True)
    t = certs.argmax(1)
    b = torch.arange(len(x))
    pred = logits.argmax(1)[b, t]
    final = logits.argmax(1)[:, -1]
    # oracle best tick uses labels? no, caller may pass separately. here just certainty path
    return {
        "pred": pred.cpu().numpy().astype(np.int64),
        "tick": t.cpu().numpy().astype(np.int64),
        "final_pred": final.cpu().numpy().astype(np.int64),
        "cert_mean": certs.mean(0).cpu().numpy(),
        "attn_ent_mean": att.mean(0).cpu().numpy(),
        "logits": logits.detach(),
        "certs": certs.detach(),
    }

def main():
    torch.set_num_threads(8)
    manifest = json.load(open(MANIFEST))
    assigned, summary = session_split(manifest)
    json.dump(assigned, open(os.path.join(ROOT, "session_manifest.json"), "w"))
    counts = {s: int(sum(1 for r in assigned if r["split"] == s)) for s in ("train", "val", "test")}
    print("SPLIT", counts, flush=True)
    for sp, info in summary.items():
        print(f"  {sp} train {info['train']} val {info['val']} test {info['test']} ids {info['n_ids']} test_id {info['test_ids']}", flush=True)

    # overlap check
    by = defaultdict(lambda: defaultdict(set))
    for r in assigned:
        by[r["species"]][r["split"]].add(r["recording_id"])
    leaks = 0
    for sp, d in by.items():
        if d["test"] & d["train"] or d["test"] & d["val"] or d["val"] & d["train"]:
            leaks += 1
    if leaks:
        raise SystemExit(f"recording id leaked across splits: {leaks}")

    print("loading audio", flush=True)
    waves, labels, splits, files, species, rids = [], [], [], [], [], []
    for rec in assigned:
        waves.append(read_channel0(os.path.join(RAW, rec["base"])))
        labels.append(LAB[rec["species"]])
        splits.append(rec["split"])
        files.append(rec["base"])
        species.append(rec["species"])
        rids.append(rec["recording_id"])
    waves = np.stack(waves)
    labels = np.asarray(labels, np.int64)
    splits = np.asarray(splits)
    fb = torch.from_numpy(mel_filterbank(N_FFT, N_MELS, SR))
    window = torch.hann_window(N_FFT)
    specs = []
    with torch.no_grad():
        for i in range(0, len(waves), 64):
            w = torch.from_numpy(waves[i:i+64])
            spec = torch.stft(w, n_fft=N_FFT, hop_length=HOP, win_length=N_FFT, window=window,
                              return_complex=True, center=False).abs().pow(2)
            mel = torch.matmul(fb, spec)
            specs.append(torch.log(mel + 1e-6).numpy())
    specs = np.concatenate(specs, 0).astype(np.float32)
    print("specs", specs.shape, flush=True)
    idx = {s: np.where(splits == s)[0] for s in ("train", "val", "test")}
    y = {s: labels[idx[s]] for s in idx}
    mu, sd = float(specs[idx["train"]].mean()), float(specs[idx["train"]].std())
    specs_n = (specs - mu) / (sd + 1e-6)

    def xt(s):
        return torch.from_numpy(specs_n[idx[s]][:, None, :, :])
    xtr, xva, xte = xt("train"), xt("val"), xt("test")
    ytr_t = torch.from_numpy(y["train"])
    yva_t = torch.from_numpy(y["val"])

    # class weights from train
    counts_c = np.bincount(y["train"], minlength=12).astype(np.float64)
    class_w = torch.tensor(counts_c.sum() / (12.0 * np.maximum(counts_c, 1.0)), dtype=torch.float32)
    print("train class counts", counts_c.tolist(), flush=True)

    mean = specs[idx["train"]].mean(2)
    std = specs[idx["train"]].std(2)
    # features from raw logmel (not the normalized tensor) so the linear model is the simple baseline
    feats = np.concatenate([
        specs.mean(2), specs.std(2)
    ], 1)
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(feats[idx["train"]])
    Xva = scaler.transform(feats[idx["val"]])
    Xte = scaler.transform(feats[idx["test"]])
    grid = []
    best_C, best_vf1 = None, -1.0
    for C in (0.01, 0.1, 1.0, 10.0, 100.0):
        clf = LogisticRegression(C=C, max_iter=2000, solver="lbfgs", class_weight="balanced")
        clf.fit(Xtr, y["train"])
        vp = clf.predict(Xva)
        vf1 = macro_f1(y["val"], vp)
        grid.append({"C": C, "val_macro_f1": vf1, "val_acc": float((vp == y["val"]).mean())})
        print(f"linear C {C} val_f1 {vf1:.3f}", flush=True)
        if vf1 > best_vf1:
            best_vf1, best_C = vf1, C
    clf = LogisticRegression(C=best_C, max_iter=2000, solver="lbfgs", class_weight="balanced")
    clf.fit(Xtr, y["train"])
    lin_pred = clf.predict(Xte).astype(np.int64)
    lin_val = clf.predict(Xva).astype(np.int64)
    print("linear chosen", best_C, "test", (lin_pred == y["test"]).mean(), flush=True)

    # majority on train mode
    maj = int(np.argmax(counts_c))
    maj_pred = np.full(len(y["test"]), maj, dtype=np.int64)

    plain_n = nparams(PlainCNN())
    ctm_n = nparams(CTMStyle())
    print("PARAMS plain", plain_n, "ctm", ctm_n, flush=True)

    seed_rows = []
    cnn_preds, ctm_preds = {}, {}
    diag = {}
    for seed in SEEDS:
        print(f"=== seed {seed} ===", flush=True)
        torch.manual_seed(seed)
        np.random.seed(seed)
        g = torch.Generator().manual_seed(seed)
        loader = DataLoader(TensorDataset(xtr, ytr_t), batch_size=BATCH, shuffle=True, generator=g)
        plain = PlainCNN()
        info_p = train_plain(plain, loader, xva, yva_t, class_w, seed)
        plain.eval()
        with torch.no_grad():
            p_te = plain(xte).argmax(1).cpu().numpy().astype(np.int64)
            p_va = plain(xva).argmax(1).cpu().numpy().astype(np.int64)
        torch.manual_seed(seed)
        g2 = torch.Generator().manual_seed(seed)
        loader2 = DataLoader(TensorDataset(xtr, ytr_t), batch_size=BATCH, shuffle=True, generator=g2)
        ctm = CTMStyle()
        info_c = train_ctm(ctm, loader2, xva, yva_t, class_w, seed)
        with torch.no_grad():
            out = predict_ctm(ctm, xte)
            # val most-certain
            lv, cv = ctm(xva)
            c_va = lv.argmax(1)[torch.arange(len(yva_t)), cv.argmax(1)].cpu().numpy().astype(np.int64)
        boot = bootstrap_diff(y["test"], out["pred"], p_te, seed=seed)
        print(f"seed {seed} test cnn {(p_te==y['test']).mean():.3f} ctm {(out['pred']==y['test']).mean():.3f} diff_ci {boot['diff_ci95']} incl0 {boot['includes_zero']}", flush=True)
        rec = {
            "seed": seed,
            "plain": {
                **{k: info_p[k] for k in ("best_epoch", "best_val_macro_f1", "epochs_ran", "seconds")},
                "history": info_p["history"],
                "test_accuracy": float((p_te == y["test"]).mean()),
                "test_macro_f1": macro_f1(y["test"], p_te),
                "test_correct": int((p_te == y["test"]).sum()),
                "val_accuracy": float((p_va == y["val"]).mean()),
                "val_macro_f1": macro_f1(y["val"], p_va),
                "per_species_recall": per_recall(y["test"], p_te),
            },
            "ctm_style": {
                **{k: info_c[k] for k in ("best_epoch", "best_val_macro_f1", "epochs_ran", "seconds")},
                "history": info_c["history"],
                "test_accuracy_most_certain_tick": float((out["pred"] == y["test"]).mean()),
                "test_macro_f1_most_certain_tick": macro_f1(y["test"], out["pred"]),
                "test_correct_most_certain": int((out["pred"] == y["test"]).sum()),
                "test_accuracy_final_tick": float((out["final_pred"] == y["test"]).mean()),
                "test_macro_f1_final_tick": macro_f1(y["test"], out["final_pred"]),
                "val_accuracy_most_certain": float((c_va == y["val"]).mean()),
                "val_macro_f1_most_certain": macro_f1(y["val"], c_va),
                "per_species_recall_most_certain": per_recall(y["test"], out["pred"]),
                "certainty_mean_per_tick": [float(v) for v in out["cert_mean"]],
                "attention_entropy_mean_per_tick": [float(v) for v in out["attn_ent_mean"]],
                "chosen_tick_hist": np.bincount(out["tick"], minlength=TICKS).tolist(),
                "fraction_chosen_tick_is_last": float((out["tick"] == TICKS - 1).mean()),
            },
            "bootstrap_ctm_minus_plain": boot,
        }
        seed_rows.append(rec)
        cnn_preds[seed] = p_te
        ctm_preds[seed] = out["pred"]
        diag[seed] = out["tick"]
        torch.save({"plain": plain.state_dict(), "ctm": ctm.state_dict()},
                   os.path.join(ROOT, f"session_models_seed{seed}.pt"))

    def mean_std(vals):
        a = np.asarray(vals, dtype=np.float64)
        return {"values": [float(v) for v in a], "mean": float(a.mean()), "min": float(a.min()), "max": float(a.max()),
                "spread_max_minus_min": float(a.max() - a.min())}

    results = {
        "data": {
            "source": "InsectSound1000 official OpenAgrar ZIP, real data, not a fallback",
            "doi": "10.5073/20231024-173119-0",
            "paper": "10.1038/s41597-024-03301-4",
            "ctm_paper": "arXiv:2505.05522",
            "subset": "768 clips already on disk (64 per species). No additional download.",
            "clips_per_species": 64,
            "n_clips": 768,
            "audio": "channel 0 of 4-channel 2.5 s 16 kHz float32",
            "split_rule": "Recording-id hold-out. Within each species, the recording id whose on-disk clip count is closest to 15% of that species' 64 clips is the entire test session; the remaining id closest to 15% is the entire validation session; all other recording ids are train. No recording id appears in more than one split. With only 3 ids (Bombus, Bradysia) this cannot be 70/15/15 of clips.",
            "split_counts": counts,
            "per_species": summary,
            "test_n": int(len(y["test"])),
            "val_n": int(len(y["val"])),
            "train_n": int(len(y["train"])),
            "train_class_counts": counts_c.tolist(),
            "recording_id_overlap_train_test": 0,
        },
        "features": {"n_fft": N_FFT, "hop": HOP, "n_mels": N_MELS, "logmel_shape": list(specs.shape[1:]),
                     "train_logmel_mean": mu, "train_logmel_std": sd},
        "models": {
            "linear": {
                "C_grid_val": grid,
                "C_selected_on_val": best_C,
                "class_weight": "balanced",
                "features": "mean and std of 32 log-mel bands (64-d)",
                "test_accuracy": float((lin_pred == y["test"]).mean()),
                "test_macro_f1": macro_f1(y["test"], lin_pred),
                "test_correct": int((lin_pred == y["test"]).sum()),
                "val_accuracy": float((lin_val == y["val"]).mean()),
                "val_macro_f1": macro_f1(y["val"], lin_val),
                "per_species_recall": per_recall(y["test"], lin_pred),
            },
            "majority": {
                "label": SPECIES[maj],
                "test_accuracy": float((maj_pred == y["test"]).mean()),
                "test_macro_f1": macro_f1(y["test"], maj_pred),
                "test_correct": int((maj_pred == y["test"]).sum()),
                "per_species_recall": per_recall(y["test"], maj_pred),
            },
            "plain_cnn": {"parameters": plain_n, "description": "One forward pass. Two conv blocks, 4x4 pool, MLP."},
            "ctm_style": {
                "parameters": ctm_n,
                "ticks": TICKS,
                "d_model": 64,
                "memory": 8,
                "k_sync": 12,
                "test_decision": "argmax class at the most certain internal tick (does not use the label)",
                "loss": "(cross-entropy at minimum-loss tick + cross-entropy at maximum-certainty tick) / 2",
                "not": "Not the full Sakana CTM.",
            },
        },
        "seeds": seed_rows,
        "summary": {},
    }
    results["summary"] = {
        "plain_test_accuracy": mean_std([r["plain"]["test_accuracy"] for r in seed_rows]),
        "plain_test_macro_f1": mean_std([r["plain"]["test_macro_f1"] for r in seed_rows]),
        "ctm_test_accuracy": mean_std([r["ctm_style"]["test_accuracy_most_certain_tick"] for r in seed_rows]),
        "ctm_test_macro_f1": mean_std([r["ctm_style"]["test_macro_f1_most_certain_tick"] for r in seed_rows]),
        "ctm_final_tick_accuracy": mean_std([r["ctm_style"]["test_accuracy_final_tick"] for r in seed_rows]),
        "fraction_most_certain_tick_is_last": mean_std([r["ctm_style"]["fraction_chosen_tick_is_last"] for r in seed_rows]),
        "bootstrap_includes_zero": [r["bootstrap_ctm_minus_plain"]["includes_zero"] for r in seed_rows],
        "diff_point": mean_std([
            r["ctm_style"]["test_accuracy_most_certain_tick"] - r["plain"]["test_accuracy"] for r in seed_rows
        ]),
    }
    # mean per-species recall
    for name, key in (("plain", "per_species_recall"), ("ctm", "per_species_recall_most_certain")):
        block = {}
        for sp in SPECIES:
            vals = [r["plain" if name == "plain" else "ctm_style"][key][sp] for r in seed_rows]
            block[sp] = {"values": vals, "mean": float(np.mean(vals))}
        results["summary"][name + "_recall"] = block

    json.dump(results, open(os.path.join(ROOT, "session_metrics.json"), "w"), indent=2)

    # predictions
    test_files = [files[i] for i in idx["test"]]
    test_sp = [species[i] for i in idx["test"]]
    with open(os.path.join(ROOT, "session_test_predictions.csv"), "w") as f:
        cols = ["file", "species", "y_true", "pred_linear", "pred_majority"]
        for s in SEEDS:
            cols += [f"pred_plain_s{s}", f"pred_ctm_s{s}", f"ctm_tick_s{s}"]
        f.write(",".join(cols) + "\n")
        for local, gi in enumerate(idx["test"]):
            row = [files[gi], species[gi], str(int(labels[gi])), str(int(lin_pred[local])), str(int(maj_pred[local]))]
            for s in SEEDS:
                row += [str(int(cnn_preds[s][local])), str(int(ctm_preds[s][local])), str(int(diag[s][local]))]
            f.write(",".join(row) + "\n")
    print("WROTE metrics", flush=True)
    write_report(results)
    print("WROTE PROJECT.md", flush=True)

def write_report(m):
    s = m["summary"]
    lin = m["models"]["linear"]
    maj = m["models"]["majority"]
    lines = []
    a = lines.append
    a("# Insect sounds, held-out nights: a small CTM-style model vs a plain CNN")
    a("")
    a("## Problem")
    a("")
    a("Can a small model that thinks in internal ticks tell 12 insect species apart from a 2.5-second microphone clip, when it is not allowed to train on the same recording night as the test clip?")
    a("")
    a("The data are InsectSound1000 (Branding et al., Scientific Data 2024, https://doi.org/10.1038/s41597-024-03301-4; dataset https://doi.org/10.5073/20231024-173119-0). The neural comparison follows the idea in Continuous Thought Machines (Darlow, Regan, Risi, Seely, Jones; Sakana AI; arXiv:2505.05522, NeurIPS 2025), at a scale that fits a CPU. This is not the full Darlow model. It only borrows the experimental shape: a simple baseline, a one-pass network, and a second network of similar size whose extra machinery is internal time.")
    a("")
    a("## Data")
    a("")
    a("Real InsectSound1000, not a fallback. Clips already on disk from the official OpenAgrar ZIP (`openagrar_derivate_00055778`), fetched earlier by HTTP range and inflate. **No further download.** Each clip is one WAV: 2.5 s, 16 kHz, 32-bit float, 4 channels. Models use **channel 0 only**.")
    a("")
    a(f"Subset size: **{m['data']['clips_per_species']} clips × 12 species = {m['data']['n_clips']} files.** Log-mel: n_fft {m['features']['n_fft']}, hop {m['features']['hop']}, {m['features']['n_mels']} mels, shape {m['features']['logmel_shape'][0]} × {m['features']['logmel_shape'][1]}.")
    a("")
    a("## Split")
    a("")
    a(m["data"]["split_rule"])
    a("")
    a(f"Counts: **train {m['data']['train_n']}, val {m['data']['val_n']}, test {m['data']['test_n']}.** Recording-id overlap between train and test: **{m['data']['recording_id_overlap_train_test']}.** The test set was frozen. `C` for logistic regression and early stopping for the networks used validation macro F1 only.")
    a("")
    a("| species | ids | train clips | val clips | test clips | test recording id |")
    a("|---|---:|---:|---:|---:|---|")
    for sp, info in m["data"]["per_species"].items():
        a(f"| {sp.replace('_',' ')} | {info['n_ids']} | {info['train']} | {info['val']} | {info['test']} | {info['test_ids'][0]} |")
    a("")
    a("Species with only three recording ids in this subset (Bombus terrestris, Bradysia difformis) cannot yield a 70/15/15 clip split. One id each goes to train, val, and test, so those species contribute about a third of their clips to test. Coccinella has two tiny ids (1 and 2 clips); the rule did not pick those as the test night, because a larger id was closer to 15%.")
    a("")
    a("This is a harder protocol than the earlier file-level split of the same 768 clips, where every test file shared a recording id with train. That earlier run (linear 52/108, CNN 48/108, GRU 51/108) is not repeated here and is not the result below.")
    a("")
    a("## Methods")
    a("")
    a("All three scored models see the same log-mel. Train-set class weights (inverse frequency) are used for the neural nets. Logistic regression uses `class_weight='balanced'`. Two training seeds, 0 and 1. Adam, learning rate 0.001, weight decay 1e-4, batch 32, at most 12 epochs, early stop after 3 epochs without a validation macro-F1 improvement. CPU only.")
    a("")
    a(f"- **Majority.** Most frequent training class (`{maj['label']}`). Because the session split is not balanced, this is not exactly 1/12.")
    a(f"- **Linear baseline.** Mean and standard deviation of each of 32 mel bands (64 features), standard-scaled on train. `C` chosen on validation from {{0.01, 0.1, 1, 10, 100}}. Selected **C = {lin['C_selected_on_val']}**.")
    a(f"- **Plain CNN.** One forward pass. Conv 16 then 32, two pools, adaptive pool to 4×4, linear 512→64→12. **{m['models']['plain_cnn']['parameters']} parameters.**")
    a(f"- **CTM-style.** **{m['models']['ctm_style']['parameters']} parameters.** {m['models']['ctm_style']['ticks']} internal ticks. Width D={m['models']['ctm_style']['d_model']}. Each neuron keeps a length-{m['models']['ctm_style']['memory']} FIFO of pre-activations and applies its **own** weights (a private GLU) to that history. A shared synapse MLP mixes an attention readout with the post-activation vector to produce the next pre-activation. Synchronization is a decayed upper-triangle of pairwise products of post-activations, on the first {m['models']['ctm_style']['k_sync']} neurons for the output and the last {m['models']['ctm_style']['k_sync']} for the attention query. Class logits are a linear map of the **output synchronization vector**, not of the neuron activations themselves. Certainty is 1 minus entropy divided by log(12).")
    a("")
    a("Loss for the CTM-style model, as in the paper: half the cross-entropy at the tick with the lowest loss, plus half the cross-entropy at the tick with the highest certainty. At test time the label is not available, so the reported decision is the class at the **most certain tick**. Final-tick accuracy is reported only as a diagnostic.")
    a("")
    a("What was left out of the Sakana CTM on purpose: the U-Net synapse, deep wide neuron-level MLPs, ImageNet-scale width (D=2048, M=25, T=75), learnable positional embeddings, and a ResNet backbone. The name in this report is **CTM-style**, not Continuous Thought Machine.")
    a("")
    a("## Results")
    a("")
    a(f"Frozen test set, n={m['data']['test_n']}. Neural numbers are two seeds. Spread is max minus min.")
    a("")
    pa, pf = s["plain_test_accuracy"], s["plain_test_macro_f1"]
    ca, cf = s["ctm_test_accuracy"], s["ctm_test_macro_f1"]
    a("| model | params | test accuracy | test macro F1 | correct / n |")
    a("|---|---:|---|---|---|")
    a(f"| Majority | 0 | {maj['test_accuracy']:.6f} | {maj['test_macro_f1']:.6f} | {maj['test_correct']}/{m['data']['test_n']} |")
    a(f"| Linear log-mel | — | {lin['test_accuracy']:.6f} | {lin['test_macro_f1']:.6f} | {lin['test_correct']}/{m['data']['test_n']} |")
    a(f"| Plain CNN | {m['models']['plain_cnn']['parameters']} | mean {pa['mean']:.6f} (seeds {pa['values'][0]:.6f}, {pa['values'][1]:.6f}) | mean {pf['mean']:.6f} (seeds {pf['values'][0]:.6f}, {pf['values'][1]:.6f}) | see seeds |")
    a(f"| CTM-style, most certain tick | {m['models']['ctm_style']['parameters']} | mean {ca['mean']:.6f} (seeds {ca['values'][0]:.6f}, {ca['values'][1]:.6f}) | mean {cf['mean']:.6f} (seeds {cf['values'][0]:.6f}, {cf['values'][1]:.6f}) | see seeds |")
    a("")
    a("Seed detail:")
    a("")
    for r in m["seeds"]:
        b = r["bootstrap_ctm_minus_plain"]
        a(f"- Seed {r['seed']}. Plain CNN {r['plain']['test_correct']}/{m['data']['test_n']} = {r['plain']['test_accuracy']:.6f}, macro F1 {r['plain']['test_macro_f1']:.6f} (best val epoch {r['plain']['best_epoch']}, ran {r['plain']['epochs_ran']}, {r['plain']['seconds']:.1f}s). CTM-style most-certain {r['ctm_style']['test_correct_most_certain']}/{m['data']['test_n']} = {r['ctm_style']['test_accuracy_most_certain_tick']:.6f}, macro F1 {r['ctm_style']['test_macro_f1_most_certain_tick']:.6f}; final-tick accuracy {r['ctm_style']['test_accuracy_final_tick']:.6f} (best val epoch {r['ctm_style']['best_epoch']}, ran {r['ctm_style']['epochs_ran']}, {r['ctm_style']['seconds']:.1f}s).")
        a(f"  Bootstrap of CTM-style minus plain accuracy, {b['n_boot']} resamples of this seed's test predictions: mean {b['diff_ctm_minus_cnn_mean']:.6f}, 95% interval [{b['diff_ci95'][0]:.6f}, {b['diff_ci95'][1]:.6f}]. Includes 0: **{b['includes_zero']}**.")
    a("")
    dlt = s["diff_point"]
    a(f"Point difference (CTM-style minus plain), two seeds: {dlt['values'][0]:.6f} and {dlt['values'][1]:.6f}, mean {dlt['mean']:.6f}.")
    a("")
    both_zero = all(s["bootstrap_includes_zero"])
    if both_zero:
        a("Both seed-wise 95% bootstrap intervals on the accuracy difference include 0. **The CTM-style model did not significantly beat the plain CNN.**")
    else:
        a("At least one seed-wise 95% bootstrap interval on the accuracy difference excludes 0. See the intervals above before calling that a stable win. With two seeds and this test size, a single interval that excludes 0 is not a strong claim.")
    a("")
    a("### Per-species recall")
    a("")
    a("Recall is correct predictions divided by the number of test clips of that species. The table is the mean of the two seeds for the neural nets. Low recall is a hard species under this night hold-out, not a claim about the full 166k-file corpus.")
    a("")
    a("| species | test clips | majority | linear | plain CNN mean | CTM-style mean |")
    a("|---|---:|---:|---:|---:|---:|")
    for sp, info in m["data"]["per_species"].items():
        a(f"| {sp.replace('_',' ')} | {info['test']} | {maj['per_species_recall'][sp]:.6f} | {lin['per_species_recall'][sp]:.6f} | {s['plain_recall'][sp]['mean']:.6f} | {s['ctm_recall'][sp]['mean']:.6f} |")
    a("")
    # hard species: lowest mean of linear/cnn/ctm
    scores = []
    for sp in m["data"]["per_species"]:
        scores.append((sp, (lin["per_species_recall"][sp] + s["plain_recall"][sp]["mean"] + s["ctm_recall"][sp]["mean"]) / 3))
    scores.sort(key=lambda z: z[1])
    hard = ", ".join(f"{sp.replace('_',' ')} ({v:.3f})" for sp, v in scores[:4])
    a(f"Hardest species by average recall of linear, CNN, and CTM-style: {hard}. Pentatomid bugs (Nezara, Palomena, Halyomorpha, Rhaphigaster) are the usual cluster to check, but the ranking above is whatever this test set actually produced.")
    a("")
    a("### Did certainty move, and was the sure tick always the last one?")
    a("")
    frac = s["fraction_most_certain_tick_is_last"]
    a(f"Fraction of test clips whose most certain tick is the last tick (tick {m['models']['ctm_style']['ticks']-1}): seeds {frac['values'][0]:.6f} and {frac['values'][1]:.6f}, mean {frac['mean']:.6f}.")
    a("")
    for r in m["seeds"]:
        certs = ", ".join(f"{v:.3f}" for v in r["ctm_style"]["certainty_mean_per_tick"])
        ents = ", ".join(f"{v:.3f}" for v in r["ctm_style"]["attention_entropy_mean_per_tick"])
        hist = r["ctm_style"]["chosen_tick_hist"]
        a(f"- Seed {r['seed']}. Mean certainty by tick: {certs}. Histogram of chosen ticks {hist}. Mean attention entropy by tick (nats, over the 24 pooled spectrogram tokens): {ents}.")
    # adaptive compute judgement from numbers
    if frac["mean"] >= 0.8:
        a("")
        a("The most certain tick is usually the last one. **Adaptive compute did not emerge** in this small model: it does not stop early on easy clips, because certainty keeps peaking at the end of the fixed 6 ticks.")
    elif frac["mean"] <= 0.4:
        a("")
        a("The most certain tick is often earlier than the last tick, so certainty is not just 'wait until the end'. That is a weak sign of uneven internal compute, not a demonstration of the paper's early-exit behavior. We did not implement a certainty threshold halt, and we did not measure wall-clock savings.")
    else:
        a("")
        a("Certainty sometimes peaks before the last tick and sometimes does not. **Adaptive compute did not clearly emerge.** We did not halt early, and there is no evidence here of the paper's pattern where easy items finish in the first few ticks.")
    a("")
    a("## Conclusion")
    a("")
    if both_zero:
        conclusion = (f"On a frozen recording-night hold-out (n={m['data']['test_n']}), the CTM-style model did not significantly beat a plain CNN of similar size: "
                      f"mean accuracy {ca['mean']:.3f} vs {pa['mean']:.3f}, and both bootstrap intervals on the difference include 0.")
    else:
        conclusion = (f"On a frozen recording-night hold-out (n={m['data']['test_n']}), mean accuracy was {ca['mean']:.3f} for the CTM-style model and {pa['mean']:.3f} for the plain CNN; "
                      f"read the bootstrap intervals before treating the gap as real.")
    # linear comparison phrase
    conclusion += f" The linear mel baseline scored {lin['test_accuracy']:.3f} accuracy and {lin['test_macro_f1']:.3f} macro F1 on the same test files."
    a(conclusion)
    a("")
    a("## Artifacts")
    a("")
    a("- `/workspace/insect-sound/PROJECT.md` — this report")
    a("- `/workspace/insect-sound/session_metrics.json` — every number above")
    a("- `/workspace/insect-sound/session_test_predictions.csv` — test predictions for both seeds")
    a("- `/workspace/insect-sound/session_manifest.json` — recording-id split")
    a("- `/workspace/insect-sound/session_models_seed0.pt`, `session_models_seed1.pt`")
    a("")
    text = "\n".join(lines) + "\n"
    open(os.path.join(ROOT, "PROJECT.md"), "w").write(text)

if __name__ == "__main__":
    main()
