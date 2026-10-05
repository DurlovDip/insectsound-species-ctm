#!/usr/bin/env python3
"""Frozen EfficientAT mn10_as embeddings + linear/MLP probe on the quarter split.

The backbone is frozen. Hyperparameters are chosen on validation macro F1 only.
The frozen test set is scored once, after selection.
"""
import hashlib
import json
import os
import struct
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score
from sklearn.preprocessing import StandardScaler

ROOT = "/workspace/insect-sound"
RAW = os.path.join(ROOT, "raw")
MANIFEST = os.path.join(ROOT, "manifest_quarter.json")
CKPT = os.path.join(ROOT, "weights", "mn10_as_mAP_471.pt")
EMB_PATH = os.path.join(ROOT, "embeddings_mn10_as.f32")
PROG_PATH = os.path.join(ROOT, "embeddings_mn10_as_progress.json")
METRICS_PATH = os.path.join(ROOT, "metrics_pretrain.json")
REPORT_PATH = os.path.join(ROOT, "PROJECT-PRETRAIN.md")
QUARTER_METRICS = os.path.join(ROOT, "metrics_quarter.json")
EFFICIENTAT = "/tmp/EfficientAT"

SPECIES = [
    "Aphidoletes_aphidimyza", "Bombus_terrestris", "Bradysia_difformis",
    "Coccinella_septempunctata", "Episyrphus_balteatus", "Halyomorpha_halys",
    "Myzus_persicae", "Nezara_viridula", "Palomena_prasina",
    "Rhaphigaster_nebulos", "Trialeurodes_vaporariorum", "Tuta_absoluta",
]
LAB = {s: i for i, s in enumerate(SPECIES)}
BATCH = 32
SR_IN = 16000
SR_MODEL = 32000
N_SAMPLES = 40000


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
    if mono.shape[0] != N_SAMPLES:
        raise RuntimeError(f"len {mono.shape[0]} {path}")
    return mono


def macro_f1(y, p):
    return float(f1_score(y, p, average="macro", labels=list(range(12)), zero_division=0))


def recalls(y, p):
    r = recall_score(y, p, average=None, labels=list(range(12)), zero_division=0)
    return {SPECIES[i]: float(r[i]) for i in range(12)}


def load_records():
    man = json.load(open(MANIFEST))
    files = man["files"]
    if len(files) != 41495:
        raise RuntimeError(f"manifest files {len(files)} != 41495")
    y = np.empty(len(files), np.int64)
    split = []
    rec_ids = []
    bases = []
    for i, r in enumerate(files):
        if r["species"] not in LAB:
            raise RuntimeError("unknown species " + r["species"])
        y[i] = LAB[r["species"]]
        split.append(r["split"])
        rec_ids.append(r["recording_id"])
        bases.append(r["base"])
    split = np.asarray(split)
    by = {}
    for s, rid in zip(split.tolist(), rec_ids):
        by.setdefault(rid, set()).add(s)
    overlap = [rid for rid, ss in by.items() if len(ss) > 1]
    if overlap:
        raise RuntimeError(f"recording ids in multiple splits: {overlap[:5]}")
    counts = {s: int((split == s).sum()) for s in ("train", "val", "test")}
    if counts != {"train": 28426, "val": 6226, "test": 6843}:
        raise RuntimeError(f"unexpected split counts {counts}")
    return bases, y, split, counts


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_frontend():
    sys.path.insert(0, EFFICIENTAT)
    os.chdir(EFFICIENTAT)  # helpers/utils.py opens a relative metadata CSV
    # get_model prints the module; keep the log readable.
    import io
    from contextlib import redirect_stdout
    from models.mn.model import get_model
    from models.preprocess import AugmentMelSTFT

    if not os.path.isfile(CKPT):
        raise RuntimeError("checkpoint missing: " + CKPT)
    buf = io.StringIO()
    with redirect_stdout(buf):
        model = get_model(
            width_mult=1.0,
            pretrained_name="mn10_as",
            strides=(2, 2, 2, 2),
            head_type="mlp",
            num_classes=527,
        )
    # Confirm the loaded file is the one we hashed, not a partial download.
    downloaded = os.path.join(EFFICIENTAT, "resources", "mn10_as_mAP_471.pt")
    if sha256(downloaded) != sha256(CKPT):
        raise RuntimeError("checkpoint hash mismatch between resources/ and weights/")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    mel = AugmentMelSTFT(
        n_mels=128, sr=SR_MODEL, win_length=800, hopsize=320, n_fft=1024,
        freqm=0, timem=0, fmax=15000,
    )
    mel.eval()
    return model, mel


@torch.no_grad()
def embed_batch(model, mel, waves):
    # waves: (B, 40000) float32 at 16 kHz. Model front-end is 32 kHz.
    x = torch.from_numpy(np.ascontiguousarray(waves))
    x = F.interpolate(x.unsqueeze(1), size=N_SAMPLES * 2, mode="linear", align_corners=False).squeeze(1)
    spec = mel(x)  # (B, 128, T)
    _logits, feats = model(spec.unsqueeze(1))
    if feats.dim() == 1:
        feats = feats.unsqueeze(0)
    return feats.float().cpu().numpy()


def extract(bases):
    n = len(bases)
    emb = np.memmap(EMB_PATH, dtype=np.float32, mode="w+" if not os.path.exists(EMB_PATH) else "r+", shape=(n, 960))
    # shape is unknown until first forward if we guess wrong; probe one clip first.
    done = 0
    if os.path.isfile(PROG_PATH):
        done = int(json.load(open(PROG_PATH))["n_done"])
    if done > n:
        raise RuntimeError(f"progress {done} > n {n}")
    if done == n and os.path.isfile(EMB_PATH):
        emb = np.memmap(EMB_PATH, dtype=np.float32, mode="r", shape=(n, 960))
        print(f"embeddings already complete n={n}", flush=True)
        return np.array(emb), 960, 0.0
    model, mel = build_frontend()
    # confirm dim on one real clip before committing the memmap width
    one = embed_batch(model, mel, read_channel0(os.path.join(RAW, bases[0]))[None, :])
    dim = int(one.shape[1])
    if dim != 960:
        raise RuntimeError(f"unexpected embedding dim {dim}")
    if done == 0:
        # rewrite a fresh memmap so a partial wrong file cannot be reused
        emb = np.memmap(EMB_PATH, dtype=np.float32, mode="w+", shape=(n, dim))
    else:
        emb = np.memmap(EMB_PATH, dtype=np.float32, mode="r+", shape=(n, dim))
        print(f"resuming extraction at {done}/{n}", flush=True)
    t0 = time.time()
    i = done
    while i < n:
        j = min(i + BATCH, n)
        waves = np.stack([read_channel0(os.path.join(RAW, bases[k])) for k in range(i, j)])
        emb[i:j] = embed_batch(model, mel, waves)
        i = j
        if (i // BATCH) % 50 == 0 or i == n:
            emb.flush()
            json.dump({"n_done": i, "dim": dim, "n": n}, open(PROG_PATH, "w"))
            rate = (i - done) / max(time.time() - t0, 1e-6)
            print(f"extract {i}/{n} {rate:.1f} clips/s", flush=True)
    emb.flush()
    json.dump({"n_done": n, "dim": dim, "n": n}, open(PROG_PATH, "w"))
    elapsed = time.time() - t0
    out = np.array(emb)
    del model, mel
    return out, dim, elapsed


class MLP(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, 12))

    def forward(self, x):
        return self.net(x)


def fit_mlp(xtr, ytr, xva, yva, hidden, class_w):
    torch.manual_seed(0)
    model = MLP(xtr.shape[1], hidden)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    w = torch.tensor(class_w, dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=w)
    xt = torch.from_numpy(xtr)
    yt = torch.from_numpy(ytr)
    xv = torch.from_numpy(xva)
    best_f1, best_state, best_ep, stall = -1.0, None, 0, 0
    bs = 256
    for ep in range(1, 41):
        model.train()
        perm = torch.randperm(xt.shape[0])
        for s in range(0, xt.shape[0], bs):
            idx = perm[s:s + bs]
            opt.zero_grad()
            loss = loss_fn(model(xt[idx]), yt[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(xv).argmax(1).numpy()
        f1 = macro_f1(yva, pred)
        if f1 > best_f1 + 1e-6:
            best_f1 = f1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_ep = ep
            stall = 0
        else:
            stall += 1
            if stall >= 5:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, best_ep, best_f1


def predict_mlp(model, x):
    model.eval()
    with torch.no_grad():
        logits = []
        t = torch.from_numpy(x)
        for s in range(0, t.shape[0], 1024):
            logits.append(model(t[s:s + 1024]))
        return torch.cat(logits, 0).argmax(1).numpy()


def main():
    t_all = time.time()
    bases, y, split, counts = load_records()
    print("split", counts, flush=True)
    emb, dim, extract_s = extract(bases)
    if emb.shape != (len(bases), dim):
        raise RuntimeError(f"emb shape {emb.shape}")
    if not np.isfinite(emb).all():
        raise RuntimeError("non-finite embeddings")

    tr = split == "train"
    va = split == "val"
    te = split == "test"
    scaler = StandardScaler()
    xtr = scaler.fit_transform(emb[tr]).astype(np.float32)
    xva = scaler.transform(emb[va]).astype(np.float32)
    xte = scaler.transform(emb[te]).astype(np.float32)
    ytr, yva, yte = y[tr], y[va], y[te]
    counts_c = np.bincount(ytr, minlength=12).astype(np.float64)
    class_w = counts_c.sum() / (12.0 * np.maximum(counts_c, 1.0))

    candidates = []
    for C in (0.01, 0.1, 1.0, 10.0, 100.0):
        t0 = time.time()
        clf = LogisticRegression(
            C=C, class_weight="balanced", solver="lbfgs", max_iter=400,
        )
        clf.fit(xtr, ytr)
        pred_va = clf.predict(xva)
        row = {
            "kind": "logistic",
            "C": C,
            "hidden": None,
            "val_macro_f1": macro_f1(yva, pred_va),
            "val_accuracy": float((pred_va == yva).mean()),
            "val_correct": int((pred_va == yva).sum()),
            "n_iter": int(np.max(clf.n_iter_)),
            "seconds": time.time() - t0,
            "estimator": clf,
        }
        candidates.append(row)
        print(
            f"logistic C={C} val_f1={row['val_macro_f1']:.6f} val_acc={row['val_accuracy']:.6f} "
            f"iters={row['n_iter']} {row['seconds']:.1f}s",
            flush=True,
        )

    for h in (64, 128):
        t0 = time.time()
        model, best_ep, _ = fit_mlp(xtr, ytr, xva, yva, h, class_w)
        pred_va = predict_mlp(model, xva)
        row = {
            "kind": "mlp",
            "C": None,
            "hidden": h,
            "best_epoch": best_ep,
            "val_macro_f1": macro_f1(yva, pred_va),
            "val_accuracy": float((pred_va == yva).mean()),
            "val_correct": int((pred_va == yva).sum()),
            "seconds": time.time() - t0,
            "estimator": model,
        }
        candidates.append(row)
        print(
            f"mlp h={h} val_f1={row['val_macro_f1']:.6f} val_acc={row['val_accuracy']:.6f} "
            f"best_ep={best_ep} {row['seconds']:.1f}s",
            flush=True,
        )

    def rank(r):
        # Locked before any test score: higher val macro F1, then val accuracy,
        # then prefer logistic over MLP, then smaller C / smaller hidden.
        kind_rank = 0 if r["kind"] == "logistic" else 1
        complexity = r["C"] if r["kind"] == "logistic" else float(r["hidden"])
        return (-r["val_macro_f1"], -r["val_accuracy"], kind_rank, complexity)

    chosen = min(candidates, key=rank)
    print("chosen", {k: chosen[k] for k in chosen if k != "estimator"}, flush=True)

    if chosen["kind"] == "logistic":
        pred = chosen["estimator"].predict(xte)
        head_desc = f"logistic regression, C={chosen['C']}, class_weight=balanced, lbfgs"
        head_params = int(chosen["estimator"].coef_.size + chosen["estimator"].intercept_.size)
    else:
        pred = predict_mlp(chosen["estimator"], xte)
        head_desc = (
            f"MLP Linear({dim}->{chosen['hidden']})-ReLU-Linear({chosen['hidden']}->12), "
            f"Adam lr 0.001, weight decay 1e-4, inverse-frequency class weights, "
            f"early stop on val macro F1 (patience 5, max 40), best epoch {chosen['best_epoch']}"
        )
        head_params = sum(p.numel() for p in chosen["estimator"].parameters())

    correct = int((pred == yte).sum())
    n_test = int(yte.shape[0])
    acc = correct / n_test
    f1 = macro_f1(yte, pred)
    rec = recalls(yte, pred)

    quarter = json.load(open(QUARTER_METRICS))
    cnn_mean = float(quarter["summary"]["plain_acc"]["mean"])
    beats = bool(acc > cnn_mean)

    ckpt_hash = sha256(CKPT)
    n_backbone = 4876831  # counted at load time in the smoke test; recompute below if model still around
    # recompute from the checkpoint tensor sizes so the report is not a remembered constant
    state = torch.load(CKPT, map_location="cpu", weights_only=True)
    n_backbone = int(sum(v.numel() for v in state.values()))

    val_table = []
    for r in candidates:
        val_table.append({k: r[k] for k in r if k != "estimator"})

    metrics = {
        "model_name": "EfficientAT mn10_as",
        "checkpoint": "mn10_as_mAP_471.pt",
        "checkpoint_sha256": ckpt_hash,
        "checkpoint_url": "https://github.com/fschmid56/EfficientAT/releases/download/v0.0.1/mn10_as_mAP_471.pt",
        "citation": "Schmid, Koutini, Widmer, Efficient Large-scale Audio Tagging via Transformer-to-CNN Knowledge Distillation, ICASSP 2023",
        "backbone_parameters_frozen": n_backbone,
        "embedding": "adaptive average pool of the last conv map, before the AudioSet MLP head",
        "embedding_dim": dim,
        "frozen": True,
        "preprocess": {
            "channel": 0,
            "input_sr": SR_IN,
            "model_sr": SR_MODEL,
            "resample": "torch.nn.functional.interpolate linear, align_corners=False, 40000 -> 80000",
            "mel": "AugmentMelSTFT eval mode, n_mels=128, n_fft=1024, win=800, hop=320, fmax=15000, no freq/time mask",
        },
        "n_clips": int(len(bases)),
        "split_counts": counts,
        "split_source": MANIFEST,
        "recording_id_overlap": 0,
        "subsampled": False,
        "extract_seconds": extract_s,
        "standardize": "StandardScaler fit on train embeddings only",
        "selection_rule": "max validation macro F1; ties: higher val accuracy, then logistic over MLP, then smaller C or hidden",
        "val_grid": val_table,
        "selected_head": head_desc,
        "selected_head_parameters": head_params,
        "test_n": n_test,
        "test_correct": correct,
        "test_accuracy": acc,
        "test_macro_f1": f1,
        "test_recall": rec,
        "cnn_mean_accuracy": cnn_mean,
        "beats_cnn_mean": beats,
        "comparison_baselines": {
            "majority_accuracy": quarter["models"]["majority_test_accuracy"],
            "majority_correct": quarter["models"]["majority_test_correct"],
            "majority_macro_f1": quarter["models"]["majority_test_macro_f1"],
            "linear_accuracy": quarter["models"]["linear_test_accuracy"],
            "linear_correct": quarter["models"]["linear_test_correct"],
            "linear_macro_f1": quarter["models"]["linear_test_macro_f1"],
            "plain_cnn_mean_accuracy": quarter["summary"]["plain_acc"]["mean"],
            "plain_cnn_mean_macro_f1": quarter["summary"]["plain_f1"]["mean"],
            "ctm_mean_accuracy": quarter["summary"]["ctm_acc"]["mean"],
            "ctm_mean_macro_f1": quarter["summary"]["ctm_f1"]["mean"],
            "test_n": quarter["data"]["test_n"],
        },
        "elapsed_seconds": time.time() - t_all,
    }
    json.dump(metrics, open(METRICS_PATH, "w"), indent=2)

    b = metrics["comparison_baselines"]
    lines = []
    a = lines.append
    a("# InsectSound1000 quarter subset, frozen pretrained audio probe")
    a("")
    a("## Problem")
    a("")
    a("Same twelve-way species task and the same frozen recording-id split as PROJECT-QUARTER.md. A public AudioSet-pretrained network is frozen. One embedding is taken from each clip. Only a linear classifier or a small MLP is trained. Nothing in the protocol was chosen using the test set.")
    a("")
    a("## Model")
    a("")
    a(f"EfficientAT MobileNetV3 `mn10_as` (Schmid, Koutini, Widmer, ICASSP 2023), checkpoint `mn10_as_mAP_471.pt` from the v0.0.1 GitHub release. SHA256 `{ckpt_hash}`. Backbone parameters frozen: **{n_backbone}**. Embedding: adaptive average pool of the last convolutional map, **{dim}** dimensions, before the AudioSet classification head. The AudioSet head is not used.")
    a("")
    a("Clips are channel 0 of the 4-channel 2.5 s 16 kHz float32 WAVs. The published front-end expects 32 kHz, so each waveform is linearly resampled from 40,000 to 80,000 samples (`interpolate`, `align_corners=False`). Mel front-end is EfficientAT `AugmentMelSTFT` in eval mode: 128 mels, n_fft 1024, window 800, hop 320, fmax 15000, no frequency or time masking. CPU. No subsampling: all 41,495 manifest clips.")
    a("")
    a(f"Extraction wall time this run: {extract_s:.1f}s (0 if embeddings were already cached).")
    a("")
    a("## Split")
    a("")
    a(f"Loaded from `manifest_quarter.json`. Train {counts['train']}, val {counts['val']}, test {counts['test']}. Recording-id overlap across splits: 0. Test n={n_test}. Embeddings are standardized with a `StandardScaler` fit on train only.")
    a("")
    a("## Head selection (validation only)")
    a("")
    a("Rule, fixed before the test score: highest validation macro F1. Ties break toward higher validation accuracy, then logistic over MLP, then smaller `C` or hidden size.")
    a("")
    a("| head | val macro F1 | val accuracy | val correct |")
    a("|---|---:|---:|---:|")
    for r in candidates:
        if r["kind"] == "logistic":
            name = f"logistic C={r['C']}"
        else:
            name = f"MLP hidden={r['hidden']} (best epoch {r['best_epoch']})"
        a(f"| {name} | {r['val_macro_f1']:.6f} | {r['val_accuracy']:.6f} | {r['val_correct']}/{counts['val']} |")
    a("")
    a(f"Selected: **{head_desc}**. Trainable head parameters: {head_params}.")
    a("")
    a("## Frozen test")
    a("")
    a(f"Full test set, n={n_test}. Not a subsample.")
    a("")
    a("| model | test accuracy | correct | test macro F1 |")
    a("|---|---:|---:|---:|")
    a(f"| Majority | {b['majority_accuracy']:.6f} | {b['majority_correct']}/{b['test_n']} | {b['majority_macro_f1']:.6f} |")
    a(f"| Linear log-mel | {b['linear_accuracy']:.6f} | {b['linear_correct']}/{b['test_n']} | {b['linear_macro_f1']:.6f} |")
    a(f"| Plain CNN mean | {b['plain_cnn_mean_accuracy']:.6f} | — | {b['plain_cnn_mean_macro_f1']:.6f} |")
    a(f"| CTM-style mean | {b['ctm_mean_accuracy']:.6f} | — | {b['ctm_mean_macro_f1']:.6f} |")
    a(f"| Frozen mn10_as + selected head | {acc:.6f} | {correct}/{n_test} | {f1:.6f} |")
    a("")
    a(f"The frozen pretrained head {'beats' if beats else 'does not beat'} the plain CNN mean accuracy ({acc:.6f} vs {cnn_mean:.6f}).")
    a("")
    a("### Per-species recall (selected head, test)")
    a("")
    a("| species | recall |")
    a("|---|---:|")
    for sp in SPECIES:
        a(f"| {sp.replace('_', ' ')} | {rec[sp]:.6f} |")
    a("")
    a(f"Numbers in this file were written by `pretrain_probe.py` from `metrics_pretrain.json`. Baseline rows were copied from `metrics_quarter.json` at write time. Total wall time {metrics['elapsed_seconds']:.1f}s.")
    a("")
    open(REPORT_PATH, "w").write("\n".join(lines))
    print("WROTE", REPORT_PATH, "acc", acc, "correct", correct, "f1", f1, "beats", beats, flush=True)


if __name__ == "__main__":
    main()
