#!/usr/bin/env python3
"""Five-seed session-holdout run on the expanded InsectSound1000 subset."""
import json, os, math, time, struct
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, recall_score

ROOT = "/workspace/insect-sound"
RAW = os.path.join(ROOT, "raw")
MANIFEST = os.path.join(ROOT, "manifest_20k.json")
SEEDS = (0, 1, 2)
N_FFT, HOP, N_MELS, SR = 512, 512, 32, 16000
MAX_EPOCHS, PATIENCE, LR, BATCH = 30, 6, 1e-3, 32
TICKS = 6
N_BOOT = 1000
D_MODEL, D_IN, MEMORY, K_SYNC = 80, 40, 8, 14

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

def nparams(m):
    return int(sum(p.numel() for p in m.parameters()))

def macro_f1(y, p):
    return float(f1_score(y, p, average="macro", labels=list(range(12)), zero_division=0))

def per_recall(y, p):
    r = recall_score(y, p, average=None, labels=list(range(12)), zero_division=0)
    return {SPECIES[i]: float(r[i]) for i in range(12)}

class ConvStem(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(16, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.net(x)

class PlainCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = ConvStem(32)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
            nn.Linear(32 * 16, 64), nn.ReLU(inplace=True), nn.Linear(64, 12),
        )

    def forward(self, x):
        return self.head(self.stem(x))

class SuperLinear(nn.Module):
    def __init__(self, in_dims, out_dims, n):
        super().__init__()
        lim = 1.0 / math.sqrt(in_dims + out_dims)
        self.w = nn.Parameter(torch.empty(in_dims, out_dims, n).uniform_(-lim, lim))
        self.b = nn.Parameter(torch.zeros(1, n, out_dims))
        self.temp = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return (torch.einsum("bnm,mhn->bnh", x, self.w) + self.b) / self.temp.clamp(min=0.05)

class CTMStyle(nn.Module):
    def __init__(self, d_model=D_MODEL, d_in=D_IN, memory=MEMORY, ticks=TICKS, k_sync=K_SYNC):
        super().__init__()
        self.d_model, self.memory, self.ticks, self.k = d_model, memory, ticks, k_sync
        self.stem = ConvStem(32)
        self.token_proj = nn.Linear(32, d_in)
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

    def forward(self, x, return_diag=False):
        h = self.stem(x)
        tok = F.adaptive_avg_pool2d(h, (4, 6)).flatten(2).transpose(1, 2)
        kv = self.token_proj(tok)
        B = x.shape[0]
        z = self.start_z.unsqueeze(0).expand(B, -1)
        trace = self.start_trace.unsqueeze(0).expand(B, -1, -1).contiguous()
        r_a = torch.exp(-self.decay_action.clamp(0, 15)).unsqueeze(0)
        r_o = torch.exp(-self.decay_out.clamp(0, 15)).unsqueeze(0)
        a_alpha = a_beta = o_alpha = o_beta = None
        logits, certs, att_ent = [], [], []
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
        logits = torch.stack(logits, dim=-1)
        certs = torch.stack(certs, dim=-1)
        if return_diag:
            return logits, certs, torch.stack(att_ent, dim=-1)
        return logits, certs

def ctm_loss(logits, certs, y, class_w):
    T = logits.shape[-1]
    target = y[:, None].expand(-1, T)
    losses = F.cross_entropy(logits, target, weight=class_w, reduction="none")
    b = torch.arange(logits.shape[0], device=logits.device)
    t_best, t_cert = losses.argmin(1), certs.argmax(1)
    return 0.5 * (losses[b, t_best].mean() + losses[b, t_cert].mean())

def augment(x):
    # x: B,1,F,T normalized log-mel. Train only.
    noise = torch.randn_like(x) * 0.15
    shift = int(torch.randint(-3, 4, (1,)).item())
    y = torch.roll(x + noise, shifts=shift, dims=-1)
    f0 = int(torch.randint(0, x.shape[-2] - 3, (1,)).item())
    y = y.clone()
    y[:, :, f0:f0 + 3, :] = 0
    return y

def train_plain(model, loader, xva, yva, class_w):
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    best, best_f1, best_ep, stall, hist = None, -1.0, -1, 0, []
    t0 = time.time()
    for ep in range(1, MAX_EPOCHS + 1):
        model.train()
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(augment(xb)), yb, weight=class_w)
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(xva).argmax(1).cpu().numpy()
        vf1 = macro_f1(yva.numpy(), pred)
        vacc = float((pred == yva.numpy()).mean())
        hist.append({"epoch": ep, "val_acc": vacc, "val_macro_f1": vf1})
        print(f"    plain ep {ep} val_acc {vacc:.3f} val_f1 {vf1:.3f}", flush=True)
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

def train_ctm(model, loader, xva, yva, class_w):
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    best, best_f1, best_ep, stall, hist = None, -1.0, -1, 0, []
    t0 = time.time()
    for ep in range(1, MAX_EPOCHS + 1):
        model.train()
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            logits, certs = model(augment(xb))
            loss = ctm_loss(logits, certs, yb, class_w)
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            logits, certs = model(xva)
            pred = logits.argmax(1)[torch.arange(len(yva)), certs.argmax(1)].cpu().numpy()
        vf1 = macro_f1(yva.numpy(), pred)
        vacc = float((pred == yva.numpy()).mean())
        hist.append({"epoch": ep, "val_acc": vacc, "val_macro_f1": vf1})
        print(f"    ctm ep {ep} val_acc {vacc:.3f} val_f1 {vf1:.3f}", flush=True)
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

def bootstrap_diff(y, pred_ctm, pred_cnn, seed):
    rng = np.random.default_rng(seed)
    n = len(y)
    diffs = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        yy = y[idx]
        diffs[i] = (pred_ctm[idx] == yy).mean() - (pred_cnn[idx] == yy).mean()
    lo, hi = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
    return {"n_boot": N_BOOT, "diff_mean": float(diffs.mean()), "diff_ci95": [lo, hi],
            "includes_zero": bool(lo <= 0 <= hi)}

def mean_pack(vals):
    a = np.asarray(vals, dtype=np.float64)
    return {"values": [float(v) for v in a], "mean": float(a.mean()),
            "min": float(a.min()), "max": float(a.max()),
            "spread_max_minus_min": float(a.max() - a.min())}

def main():
    torch.set_num_threads(8)
    blob = json.load(open(MANIFEST))
    assigned, summary = blob["files"], blob["summary"]
    missing = [r["base"] for r in assigned if not os.path.exists(os.path.join(RAW, r["base"]))]
    if missing:
        raise SystemExit(f"missing {len(missing)} wavs, e.g. {missing[:3]}")
    # leak check
    seen = defaultdict(set)
    for r in assigned:
        seen[(r["species"], r["recording_id"])].add(r["split"])
    if any(len(v) > 1 for v in seen.values()):
        raise SystemExit("recording id leaked")
    counts = {s: int(sum(1 for r in assigned if r["split"] == s)) for s in ("train", "val", "test")}
    print("N", len(assigned), counts, "params", nparams(PlainCNN()), nparams(CTMStyle()), flush=True)

    print("loading audio", flush=True)
    waves, labels, splits, files, species = [], [], [], [], []
    for rec in assigned:
        waves.append(read_channel0(os.path.join(RAW, rec["base"])))
        labels.append(LAB[rec["species"]])
        splits.append(rec["split"])
        files.append(rec["base"])
        species.append(rec["species"])
    waves = np.stack(waves)
    labels = np.asarray(labels, np.int64)
    splits = np.asarray(splits)
    fb = torch.from_numpy(mel_filterbank(N_FFT, N_MELS, SR))
    window = torch.hann_window(N_FFT)
    specs = []
    with torch.no_grad():
        for i in range(0, len(waves), 64):
            w = torch.from_numpy(waves[i:i + 64])
            spec = torch.stft(w, n_fft=N_FFT, hop_length=HOP, win_length=N_FFT, window=window,
                              return_complex=True, center=False).abs().pow(2)
            specs.append(torch.log(torch.matmul(fb, spec) + 1e-6).numpy())
    del waves
    specs = np.concatenate(specs, 0).astype(np.float32)
    print("specs", specs.shape, flush=True)
    idx = {s: np.where(splits == s)[0] for s in ("train", "val", "test")}
    y = {s: labels[idx[s]] for s in idx}
    mu, sd = float(specs[idx["train"]].mean()), float(specs[idx["train"]].std())
    specs_n = ((specs - mu) / (sd + 1e-6)).astype(np.float32)

    def xt(s):
        return torch.from_numpy(np.ascontiguousarray(specs_n[idx[s]][:, None]))
    xtr, xva, xte = xt("train"), xt("val"), xt("test")
    ytr_t, yva_t = torch.from_numpy(y["train"]), torch.from_numpy(y["val"])
    cc = np.bincount(y["train"], minlength=12).astype(np.float64)
    class_w = torch.tensor(cc.sum() / (12.0 * np.maximum(cc, 1.0)), dtype=torch.float32)
    print("train counts", cc.tolist(), flush=True)

    feats = np.concatenate([specs.mean(2), specs.std(2)], 1)
    scaler = StandardScaler().fit(feats[idx["train"]])
    Xtr, Xva, Xte = scaler.transform(feats[idx["train"]]), scaler.transform(feats[idx["val"]]), scaler.transform(feats[idx["test"]])
    grid, best_C, best_vf1 = [], None, -1.0
    for C in (0.01, 0.1, 1.0, 10.0, 100.0):
        clf = LogisticRegression(C=C, max_iter=3000, solver="lbfgs", class_weight="balanced")
        clf.fit(Xtr, y["train"])
        vp = clf.predict(Xva)
        vf1 = macro_f1(y["val"], vp)
        grid.append({"C": C, "val_macro_f1": vf1, "val_acc": float((vp == y["val"]).mean())})
        print(f"linear C {C} val_f1 {vf1:.3f} val_acc {(vp==y['val']).mean():.3f}", flush=True)
        if vf1 > best_vf1:
            best_vf1, best_C = vf1, C
    clf = LogisticRegression(C=best_C, max_iter=3000, solver="lbfgs", class_weight="balanced")
    clf.fit(Xtr, y["train"])
    lin_pred = clf.predict(Xte).astype(np.int64)
    lin_val = clf.predict(Xva).astype(np.int64)
    maj = int(np.argmax(cc))
    maj_pred = np.full(len(y["test"]), maj, dtype=np.int64)
    print("linear C", best_C, "chosen on val only; test not used to pick it", flush=True)

    plain_n, ctm_n = nparams(PlainCNN()), nparams(CTMStyle())
    seed_rows, cnn_preds, ctm_preds, ticks = [], {}, {}, {}
    for seed in SEEDS:
        print(f"=== seed {seed} ===", flush=True)
        torch.manual_seed(seed); np.random.seed(seed)
        g = torch.Generator().manual_seed(seed)
        loader = DataLoader(TensorDataset(xtr, ytr_t), batch_size=BATCH, shuffle=True, generator=g)
        plain = PlainCNN()
        info_p = train_plain(plain, loader, xva, yva_t, class_w)
        plain.eval()
        with torch.no_grad():
            p_te = plain(xte).argmax(1).cpu().numpy().astype(np.int64)
            p_va = plain(xva).argmax(1).cpu().numpy().astype(np.int64)
        torch.manual_seed(seed)
        g2 = torch.Generator().manual_seed(seed)
        loader2 = DataLoader(TensorDataset(xtr, ytr_t), batch_size=BATCH, shuffle=True, generator=g2)
        ctm = CTMStyle()
        info_c = train_ctm(ctm, loader2, xva, yva_t, class_w)
        ctm.eval()
        with torch.no_grad():
            logits, certs, att = ctm(xte, return_diag=True)
            t = certs.argmax(1)
            b = torch.arange(len(y["test"]))
            c_te = logits.argmax(1)[b, t].cpu().numpy().astype(np.int64)
            final = logits.argmax(1)[:, -1].cpu().numpy().astype(np.int64)
            tick = t.cpu().numpy().astype(np.int64)
            cert_mean = certs.mean(0).cpu().numpy()
            att_mean = att.mean(0).cpu().numpy()
            lv, cv = ctm(xva)
            c_va = lv.argmax(1)[torch.arange(len(yva_t)), cv.argmax(1)].cpu().numpy().astype(np.int64)
        boot = bootstrap_diff(y["test"], c_te, p_te, seed)
        print(f"seed {seed} cnn {(p_te==y['test']).mean():.3f} ctm {(c_te==y['test']).mean():.3f} ci {boot['diff_ci95']} zero {boot['includes_zero']}", flush=True)
        seed_rows.append({
            "seed": seed,
            "plain": {**{k: info_p[k] for k in ("best_epoch", "best_val_macro_f1", "epochs_ran", "seconds")},
                      "history": info_p["history"],
                      "test_accuracy": float((p_te == y["test"]).mean()),
                      "test_macro_f1": macro_f1(y["test"], p_te),
                      "test_correct": int((p_te == y["test"]).sum()),
                      "val_accuracy": float((p_va == y["val"]).mean()),
                      "val_macro_f1": macro_f1(y["val"], p_va),
                      "per_species_recall": per_recall(y["test"], p_te)},
            "ctm_style": {**{k: info_c[k] for k in ("best_epoch", "best_val_macro_f1", "epochs_ran", "seconds")},
                          "history": info_c["history"],
                          "test_accuracy_most_certain_tick": float((c_te == y["test"]).mean()),
                          "test_macro_f1_most_certain_tick": macro_f1(y["test"], c_te),
                          "test_correct_most_certain": int((c_te == y["test"]).sum()),
                          "test_accuracy_final_tick": float((final == y["test"]).mean()),
                          "val_accuracy_most_certain": float((c_va == y["val"]).mean()),
                          "val_macro_f1_most_certain": macro_f1(y["val"], c_va),
                          "per_species_recall_most_certain": per_recall(y["test"], c_te),
                          "certainty_mean_per_tick": [float(v) for v in cert_mean],
                          "attention_entropy_mean_per_tick": [float(v) for v in att_mean],
                          "chosen_tick_hist": np.bincount(tick, minlength=TICKS).tolist(),
                          "fraction_chosen_tick_is_last": float((tick == TICKS - 1).mean())},
            "bootstrap_ctm_minus_plain": boot,
        })
        cnn_preds[seed], ctm_preds[seed], ticks[seed] = p_te, c_te, tick
        torch.save({"plain": plain.state_dict(), "ctm": ctm.state_dict()},
                   os.path.join(ROOT, f"models_20k_s{seed}.pt"))

    results = {
        "data": {
            "source": "InsectSound1000 official OpenAgrar ZIP, real clips, not a fallback",
            "doi": "10.5073/20231024-173119-0",
            "n_clips": len(assigned),
            "clips_per_species": {sp: int(sum(1 for r in assigned if r["species"] == sp)) for sp in SPECIES},
            "split_counts": counts,
            "per_species": summary,
            "split_rule": "Whole recording ids. For each species the id whose full-archive file count is closest to 15% is entirely test; the remaining id closest to 15% is entirely validation; all other ids are train. Clips are then subsampled evenly inside those ids up to 320 per species. No id is in two splits.",
            "test_n": int(len(y["test"])), "val_n": int(len(y["val"])), "train_n": int(len(y["train"])),
            "train_class_counts": cc.tolist(),
            "audio": "channel 0 of 4-ch 2.5s 16kHz float32",
            "features": {"n_fft": N_FFT, "hop": HOP, "n_mels": N_MELS, "shape": list(specs.shape[1:])},
        },
        "protocol_locked_before_test": {
            "seeds": list(SEEDS), "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "lr": LR,
            "augmentation_train_only": "gaussian noise std 0.15, time roll +/-3 frames, 3-bin frequency mask",
            "class_weight_neural": "inverse train frequency",
            "linear_class_weight": "balanced",
            "C_grid": [0.01, 0.1, 1.0, 10.0, 100.0],
            "ctm_decision": "most certain tick",
            "ctm_loss": "0.5 * (CE at lowest-loss tick + CE at highest-certainty tick)",
            "early_stop": "validation macro F1",
        },
        "models": {
            "majority_label": SPECIES[maj],
            "majority_test_accuracy": float((maj_pred == y["test"]).mean()),
            "majority_test_macro_f1": macro_f1(y["test"], maj_pred),
            "majority_test_correct": int((maj_pred == y["test"]).sum()),
            "majority_recall": per_recall(y["test"], maj_pred),
            "linear_C": best_C, "linear_grid": grid,
            "linear_test_accuracy": float((lin_pred == y["test"]).mean()),
            "linear_test_macro_f1": macro_f1(y["test"], lin_pred),
            "linear_test_correct": int((lin_pred == y["test"]).sum()),
            "linear_val_accuracy": float((lin_val == y["val"]).mean()),
            "linear_val_macro_f1": macro_f1(y["val"], lin_val),
            "linear_recall": per_recall(y["test"], lin_pred),
            "plain_parameters": plain_n,
            "ctm_parameters": ctm_n,
            "ctm_ticks": TICKS, "d_model": D_MODEL, "memory": MEMORY, "k_sync": K_SYNC,
        },
        "seeds": seed_rows,
    }
    results["summary"] = {
        "plain_acc": mean_pack([r["plain"]["test_accuracy"] for r in seed_rows]),
        "plain_f1": mean_pack([r["plain"]["test_macro_f1"] for r in seed_rows]),
        "ctm_acc": mean_pack([r["ctm_style"]["test_accuracy_most_certain_tick"] for r in seed_rows]),
        "ctm_f1": mean_pack([r["ctm_style"]["test_macro_f1_most_certain_tick"] for r in seed_rows]),
        "diff": mean_pack([r["ctm_style"]["test_accuracy_most_certain_tick"] - r["plain"]["test_accuracy"] for r in seed_rows]),
        "frac_last": mean_pack([r["ctm_style"]["fraction_chosen_tick_is_last"] for r in seed_rows]),
        "bootstrap_includes_zero": [r["bootstrap_ctm_minus_plain"]["includes_zero"] for r in seed_rows],
    }
    for key, src in (("plain_recall", "per_species_recall"), ("ctm_recall", "per_species_recall_most_certain")):
        block = {}
        side = "plain" if key.startswith("plain") else "ctm_style"
        for sp in SPECIES:
            vals = [r[side][src][sp] for r in seed_rows]
            block[sp] = {"values": vals, "mean": float(np.mean(vals))}
        results["summary"][key] = block
    json.dump(results, open(os.path.join(ROOT, "metrics_20k.json"), "w"), indent=2)
    with open(os.path.join(ROOT, "test_predictions_20k.csv"), "w") as f:
        cols = ["file", "species", "y_true", "pred_linear", "pred_majority"]
        for s in SEEDS:
            cols += [f"pred_plain_s{s}", f"pred_ctm_s{s}", f"ctm_tick_s{s}"]
        f.write(",".join(cols) + "\n")
        for local, gi in enumerate(idx["test"]):
            row = [files[gi], species[gi], str(int(labels[gi])), str(int(lin_pred[local])), str(int(maj_pred[local]))]
            for s in SEEDS:
                row += [str(int(cnn_preds[s][local])), str(int(ctm_preds[s][local])), str(int(ticks[s][local]))]
            f.write(",".join(row) + "\n")
    write_report(results)
    print("WROTE PROJECT-20K.md", flush=True)


PREV = {
    "n_clips": 3840, "test_n": 641,
    "linear_acc": 0.43057722308892354, "linear_f1": 0.4265350420376932,
    "cnn_acc": 0.36536661466458653, "cnn_f1": 0.364671307831986,
    "ctm_acc": 0.3372854914196568, "ctm_f1": 0.32228317725731176,
}

def write_report(m):
    s, mo, d = m["summary"], m["models"], m["data"]
    a = []
    w = a.append
    w("# InsectSound1000, about 20,000 clips, three seeds")
    w("")
    w("## Problem")
    w("")
    w("Twelve-way species classification on a larger slice of InsectSound1000 (Branding et al., Scientific Data 2024; dataset DOI 10.5073/20231024-173119-0). Same three systems as the 3,840-clip run: a linear log-mel baseline, a one-pass CNN, and a CTM-style model (private neuron histories, synchronization readout, six internal ticks, loss at the best tick and the most certain tick, test decision at the most certain tick). Not the full Darlow et al. model (arXiv:2505.05522).")
    w("")
    w("## Data and split")
    w("")
    w(f"Real OpenAgrar ZIP members, channel 0 only. **{d['n_clips']} clips.** Target was 1,830 per species where the archive has that many files. Trialeurodes vaporariorum and Tuta absoluta are smaller than 1,830 in the full set, so those two species use every clip. Log-mel n_fft {d['features']['n_fft']}, hop {d['features']['hop']}, {d['features']['n_mels']} bands, shape {d['features']['shape']}.")
    w("")
    w(d["split_rule"])
    w("")
    w(f"Split sizes: **train {d['train_n']}, val {d['val_n']}, test {d['test_n']}.** The test set was frozen. `C` and early stopping used validation macro F1 only. No test score was used to change the protocol.")
    w("")
    w("| species | archive files | sampled | train | val | test | test recording id | capped |")
    w("|---|---:|---:|---:|---:|---:|---|---|")
    for sp, info in d["per_species"].items():
        w(f"| {sp.replace('_',' ')} | {info['archive_files']} | {info['sampled']} | {info['train']} | {info['val']} | {info['test']} | {info['test_ids'][0]} | {info['capped']} |")
    w("")
    w("## Methods")
    w("")
    w("This retrain is the same model family as PROJECT-5SEED.md, not a new architecture. Changes that are only about fitting: inverse-frequency class weights on the neural nets, balanced class weights on logistic regression, at most 30 epochs, early stop after 6 epochs without a gain in validation macro F1, and the same train-only log-mel augmentation (noise 0.15, time roll of at most 3 frames, 3-bin frequency mask). Three seeds, 0, 1, and 2. Adam, learning rate 0.001, weight decay 1e-4, batch 32. CPU.")
    w("")
    w(f"- Majority class: `{mo['majority_label']}`.")
    w(f"- Linear: mean and standard deviation of 32 mel bands. `C` on validation from {{0.01, 0.1, 1, 10, 100}}. Selected **C = {mo['linear_C']}** (val macro F1 {mo['linear_val_macro_f1']:.6f}, val accuracy {mo['linear_val_accuracy']:.6f}).")
    w(f"- Plain CNN, one pass, **{mo['plain_parameters']} parameters.**")
    w(f"- CTM-style, **{mo['ctm_parameters']} parameters**, {mo['ctm_ticks']} ticks, D={mo['d_model']}, history {mo['memory']}, sync subset {mo['k_sync']}.")
    w("")
    w("## Results")
    w("")
    w(f"Frozen test n={d['test_n']}.")
    w("")
    pa, pf, ca, cf = s["plain_acc"], s["plain_f1"], s["ctm_acc"], s["ctm_f1"]
    def fmt(vals):
        return ", ".join(f"{v:.6f}" for v in vals)
    w("| model | params | test accuracy mean | per seed | test macro F1 mean | per seed |")
    w("|---|---:|---:|---|---:|---|")
    w(f"| Majority | 0 | {mo['majority_test_accuracy']:.6f} | — | {mo['majority_test_macro_f1']:.6f} | — |")
    w(f"| Linear | — | {mo['linear_test_accuracy']:.6f} | — | {mo['linear_test_macro_f1']:.6f} | — |")
    w(f"| Plain CNN | {mo['plain_parameters']} | {pa['mean']:.6f} | {fmt(pa['values'])} | {pf['mean']:.6f} | {fmt(pf['values'])} |")
    w(f"| CTM-style | {mo['ctm_parameters']} | {ca['mean']:.6f} | {fmt(ca['values'])} | {cf['mean']:.6f} | {fmt(cf['values'])} |")
    w("")
    w(f"Correct counts: majority {mo['majority_test_correct']}/{d['test_n']}, linear {mo['linear_test_correct']}/{d['test_n']}.")
    w("")
    for r in m["seeds"]:
        b = r["bootstrap_ctm_minus_plain"]
        w(f"- Seed {r['seed']}. CNN {r['plain']['test_correct']}/{d['test_n']} = {r['plain']['test_accuracy']:.6f}, macro F1 {r['plain']['test_macro_f1']:.6f} (best val epoch {r['plain']['best_epoch']}, ran {r['plain']['epochs_ran']}, {r['plain']['seconds']:.1f}s). CTM-style {r['ctm_style']['test_correct_most_certain']}/{d['test_n']} = {r['ctm_style']['test_accuracy_most_certain_tick']:.6f}, macro F1 {r['ctm_style']['test_macro_f1_most_certain_tick']:.6f}, final-tick accuracy {r['ctm_style']['test_accuracy_final_tick']:.6f} (best val epoch {r['ctm_style']['best_epoch']}, ran {r['ctm_style']['epochs_ran']}, {r['ctm_style']['seconds']:.1f}s).")
        w(f"  Bootstrap CTM-style minus CNN, {b['n_boot']} resamples: mean {b['diff_mean']:.6f}, 95% [{b['diff_ci95'][0]:.6f}, {b['diff_ci95'][1]:.6f}], includes 0: **{b['includes_zero']}**.")
    w("")
    diff = s["diff"]
    w(f"Point differences by seed: {fmt(diff['values'])}. Mean {diff['mean']:.6f}.")
    zeros = s["bootstrap_includes_zero"]
    if all(zeros) or any(zeros):
        w("")
        w("**The CTM-style model did not significantly beat the plain CNN.** At least one seed's 95% bootstrap interval on the accuracy difference includes 0, or every interval does.")
    elif all(v > 0 for v in diff["values"]):
        w("")
        w("No seed interval includes 0 and every difference is positive, so the CTM-style model beat the CNN on these three seeds. That is not the full Darlow model.")
    else:
        w("")
        w("The CTM-style model did not win against the plain CNN on this test set.")
    w("")
    w("### Versus the 3,840-clip run")
    w("")
    w("Previous frozen test (n=641), five-seed means for the nets, from metrics_5seed.json: linear accuracy 0.430577, macro F1 0.426535; plain CNN mean accuracy 0.365367, macro F1 0.364671; CTM-style mean accuracy 0.337285, macro F1 0.322283. The user-facing rounded figures were 0.431, 0.365, and 0.337.")
    w("")
    def rose(new, old):
        return "rose" if new > old + 1e-6 else ("fell" if new < old - 1e-6 else "unchanged")
    w(f"- Linear accuracy {rose(mo['linear_test_accuracy'], PREV['linear_acc'])}: {PREV['linear_acc']:.6f} -> {mo['linear_test_accuracy']:.6f} (delta {mo['linear_test_accuracy']-PREV['linear_acc']:+.6f}). Macro F1 {PREV['linear_f1']:.6f} -> {mo['linear_test_macro_f1']:.6f} (delta {mo['linear_test_macro_f1']-PREV['linear_f1']:+.6f}).")
    w(f"- Plain CNN mean accuracy {rose(pa['mean'], PREV['cnn_acc'])}: {PREV['cnn_acc']:.6f} -> {pa['mean']:.6f} (delta {pa['mean']-PREV['cnn_acc']:+.6f}). Macro F1 {PREV['cnn_f1']:.6f} -> {pf['mean']:.6f} (delta {pf['mean']-PREV['cnn_f1']:+.6f}).")
    w(f"- CTM-style mean accuracy {rose(ca['mean'], PREV['ctm_acc'])}: {PREV['ctm_acc']:.6f} -> {ca['mean']:.6f} (delta {ca['mean']-PREV['ctm_acc']:+.6f}). Macro F1 {PREV['ctm_f1']:.6f} -> {cf['mean']:.6f} (delta {cf['mean']-PREV['ctm_f1']:+.6f}).")
    w("")
    w("These are different test sets (different nights and a larger n), so the delta is not a paired test. It only says whether the headline accuracy on the new frozen test is higher than the headline accuracy on the old one.")
    w("")
    w("### Per-species recall")
    w("")
    w("| species | test clips | majority | linear | plain CNN mean | CTM-style mean |")
    w("|---|---:|---:|---:|---:|---:|")
    for sp, info in d["per_species"].items():
        w(f"| {sp.replace('_',' ')} | {info['test']} | {mo['majority_recall'][sp]:.6f} | {mo['linear_recall'][sp]:.6f} | {s['plain_recall'][sp]['mean']:.6f} | {s['ctm_recall'][sp]['mean']:.6f} |")
    scores = []
    for sp, info in d["per_species"].items():
        scores.append((sp, (mo['linear_recall'][sp] + s['plain_recall'][sp]['mean'] + s['ctm_recall'][sp]['mean']) / 3))
    scores.sort(key=lambda z: z[1])
    w("")
    w("Lowest average recall: " + ", ".join(f"{sp.replace('_',' ')} ({v:.3f})" for sp, v in scores[:4]) + ".")
    w("")
    fr = s["frac_last"]
    w(f"Fraction of test clips whose most certain tick is the last tick: {fmt(fr['values'])}, mean {fr['mean']:.6f}.")
    if fr["mean"] >= 0.8:
        w("The most certain tick is usually the last one. Adaptive compute did not emerge.")
    else:
        w("The most certain tick is not always the last tick. There was no early-exit rule, so this is not a measured saving in compute.")
    w("")
    w("## Conclusion")
    w("")
    if all(zeros) or any(zeros):
        conclusion = f"The CTM-style model did not significantly beat the plain CNN (mean accuracy {ca['mean']:.6f} vs {pa['mean']:.6f}, n={d['test_n']})."
    elif all(v > 0 for v in diff["values"]):
        conclusion = f"The CTM-style model beat the plain CNN on every seed (mean accuracy {ca['mean']:.6f} vs {pa['mean']:.6f})."
    else:
        conclusion = f"The CTM-style model did not win (mean accuracy {ca['mean']:.6f} vs {pa['mean']:.6f})."
    conclusion += f" Linear accuracy was {mo['linear_test_accuracy']:.6f}."
    w(conclusion)
    w("")
    w("## Artifacts")
    w("")
    w("- `/workspace/insect-sound/PROJECT-20K.md`")
    w("- `/workspace/insect-sound/metrics_20k.json`")
    w("- `/workspace/insect-sound/test_predictions_20k.csv`")
    w("- `/workspace/insect-sound/manifest_20k.json`")
    w("")
    open("/workspace/insect-sound/PROJECT-20K.md","w").write("\n".join(a)+"\n")

if __name__ == "__main__":
    main()
