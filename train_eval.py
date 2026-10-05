#!/usr/bin/env python3
"""Small CPU experiment: log-mel linear baseline, plain CNN, iterative GRU head."""
import json, os, struct, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, recall_score

ROOT = "/workspace/insect-sound"
MANIFEST = os.path.join(ROOT, "manifest.json")
RAW = os.path.join(ROOT, "raw")
SEED = 0
N_FFT = 512
HOP = 256
N_MELS = 40
SR = 16000
STEPS = 6
MAX_EPOCHS = 30
PATIENCE = 5
LR = 1e-3
BATCH = 32

SPECIES = [
    "Aphidoletes_aphidimyza", "Bombus_terrestris", "Bradysia_difformis",
    "Coccinella_septempunctata", "Episyrphus_balteatus", "Halyomorpha_halys",
    "Myzus_persicae", "Nezara_viridula", "Palomena_prasina",
    "Rhaphigaster_nebulos", "Trialeurodes_vaporariorum", "Tuta_absoluta",
]
LAB = {s: i for i, s in enumerate(SPECIES)}

def read_channel0(path):
    raw = open(path, "rb").read()
    if raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise RuntimeError("not wav " + path)
    off = 12
    data = None
    while off + 8 <= len(raw):
        cid = raw[off:off + 4]
        sz = struct.unpack_from("<I", raw, off + 4)[0]
        if cid == b"data":
            data = raw[off + 8:off + 8 + sz]
            break
        off += 8 + sz + (sz & 1)
    if data is None:
        raise RuntimeError("no data chunk " + path)
    x = np.frombuffer(data, dtype="<f4")
    if x.size % 4 != 0:
        raise RuntimeError("channel mismatch " + path)
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
    bins = np.floor((n_fft + 1) * hz / sr).astype(int)
    bins = np.clip(bins, 0, n_freq - 1)
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
    # normalize filters to unit peak
    return fb

def logmel_batch(waves, fb, window):
    # waves: N, T tensor
    spec = torch.stft(
        waves, n_fft=N_FFT, hop_length=HOP, win_length=N_FFT,
        window=window, return_complex=True, center=False,
    )
    power = spec.abs().pow(2)  # N, F, frames
    mel = torch.matmul(fb, power)  # N, M, frames
    return torch.log(mel + 1e-6)

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 48, 3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.fc = nn.Linear(48 * 4 * 4, 96)

    def forward(self, x):
        h = self.conv(x).flatten(1)
        return torch.relu(self.fc(h))

class PlainCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = Encoder()
        self.head = nn.Sequential(
            nn.Linear(96, 140),
            nn.ReLU(inplace=True),
            nn.Linear(140, 140),
            nn.ReLU(inplace=True),
            nn.Linear(140, 12),
        )

    def forward(self, x):
        return self.head(self.enc(x))

class IterativeHead(nn.Module):
    """Same conv encoder as PlainCNN, then a 6-step GRU over the embedding.
    Not a Continuous Thought Machine; a small recurrent/iterative head.
    """
    def __init__(self, steps=STEPS):
        super().__init__()
        self.enc = Encoder()
        self.steps = steps
        self.step_emb = nn.Embedding(steps, 8)
        self.gru = nn.GRU(input_size=96 + 8, hidden_size=64, num_layers=1, batch_first=True)
        self.head = nn.Linear(64, 12)

    def forward(self, x):
        z = self.enc(x)  # B, 96
        b = z.shape[0]
        t = torch.arange(self.steps, device=z.device)
        e = self.step_emb(t)  # T, 8
        e = e.unsqueeze(0).expand(b, -1, -1)
        zseq = z.unsqueeze(1).expand(-1, self.steps, -1)
        seq = torch.cat([zseq, e], dim=-1)
        out, _ = self.gru(seq)
        return self.head(out[:, -1, :])

def nparams(m):
    return sum(p.numel() for p in m.parameters())

def macro_f1_np(y, p):
    return float(f1_score(y, p, average="macro", labels=list(range(12)), zero_division=0))

def per_species_recall(y, p):
    r = recall_score(y, p, average=None, labels=list(range(12)), zero_division=0)
    return {SPECIES[i]: float(r[i]) for i in range(12)}

def bootstrap(y, preds, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y)
    names = list(preds)
    acc = {k: [] for k in names}
    f1s = {k: [] for k in names}
    diff = []  # iterative - plain
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yy = y[idx]
        accs = {}
        for k, p in preds.items():
            pp = p[idx]
            accs[k] = float((pp == yy).mean())
            acc[k].append(accs[k])
            f1s[k].append(macro_f1_np(yy, pp))
        diff.append(accs["iterative"] - accs["plain_cnn"])
    def ci(a):
        a = np.asarray(a)
        return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    return {
        "n_boot": n_boot,
        "accuracy_ci95": {k: ci(v) for k, v in acc.items()},
        "macro_f1_ci95": {k: ci(v) for k, v in f1s.items()},
        "acc_iterative_minus_plain_ci95": ci(diff),
        "acc_diff_mean": float(np.mean(diff)),
    }

def train_model(model, train_loader, val_x, val_y, name):
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    best_state = None
    best_f1 = -1.0
    best_epoch = -1
    stall = 0
    history = []
    t0 = time.time()
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total = 0.0
        n = 0
        correct = 0
        for xb, yb in train_loader:
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(yb)
            n += len(yb)
            correct += int((logits.argmax(1) == yb).sum().item())
        model.eval()
        with torch.no_grad():
            vlogits = model(val_x)
            vpred = vlogits.argmax(1).cpu().numpy()
            vy = val_y.cpu().numpy()
            vf1 = macro_f1_np(vy, vpred)
            vacc = float((vpred == vy).mean())
        history.append({
            "epoch": epoch,
            "train_loss": total / n,
            "train_acc": correct / n,
            "val_acc": vacc,
            "val_macro_f1": vf1,
        })
        print(f"{name} epoch {epoch} train_acc={correct/n:.3f} val_acc={vacc:.3f} val_macro_f1={vf1:.3f}", flush=True)
        if vf1 > best_f1 + 1e-4:
            best_f1 = vf1
            best_epoch = epoch
            stall = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stall += 1
            if stall >= PATIENCE:
                print(f"{name} early stop at epoch {epoch}, best epoch {best_epoch} val_macro_f1 {best_f1:.4f}", flush=True)
                break
    model.load_state_dict(best_state)
    return {
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_f1,
        "epochs_ran": history[-1]["epoch"],
        "history": history,
        "seconds": time.time() - t0,
    }

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    manifest = json.load(open(MANIFEST))
    print("loading audio", len(manifest), flush=True)
    waves = []
    labels = []
    splits = []
    files = []
    species = []
    for rec in manifest:
        path = os.path.join(RAW, rec["base"])
        waves.append(read_channel0(path))
        labels.append(LAB[rec["species"]])
        splits.append(rec["split"])
        files.append(rec["base"])
        species.append(rec["species"])
    waves = np.stack(waves)
    labels = np.asarray(labels, dtype=np.int64)
    splits = np.asarray(splits)
    print("waves", waves.shape, "dtype", waves.dtype, flush=True)

    fb = torch.from_numpy(mel_filterbank(N_FFT, N_MELS, SR))
    window = torch.hann_window(N_FFT)
    specs = []
    bs = 64
    with torch.no_grad():
        for i in range(0, len(waves), bs):
            w = torch.from_numpy(waves[i:i + bs])
            specs.append(logmel_batch(w, fb, window).numpy())
    specs = np.concatenate(specs, axis=0)  # N, M, T
    print("specs", specs.shape, flush=True)
    np.savez_compressed(
        os.path.join(ROOT, "logmel_cache.npz"),
        specs=specs.astype(np.float16),
        labels=labels,
        splits=splits,
        files=np.array(files),
    )

    # per-clip mean/std of mel bands
    mean = specs.mean(axis=2)
    std = specs.std(axis=2)
    feats = np.concatenate([mean, std], axis=1).astype(np.float64)  # N, 80

    idx = {s: np.where(splits == s)[0] for s in ("train", "val", "test")}
    y = {s: labels[idx[s]] for s in idx}

    # majority: smallest class index among train modes (tie -> class 0 given equal counts)
    train_counts = np.bincount(y["train"], minlength=12)
    majority_label = int(np.argmax(train_counts))  # first max
    maj_pred = np.full(len(y["test"]), majority_label, dtype=np.int64)

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(feats[idx["train"]])
    Xva = scaler.transform(feats[idx["val"]])
    Xte = scaler.transform(feats[idx["test"]])
    # C chosen on val only, small grid, not on test
    best_C, best_val = None, -1.0
    val_grid = []
    for C in (0.01, 0.1, 1.0, 10.0):
        clf_c = LogisticRegression(C=C, max_iter=2000, solver="lbfgs")
        clf_c.fit(Xtr, y["train"])
        vp = clf_c.predict(Xva)
        vf1 = macro_f1_np(y["val"], vp)
        val_grid.append({"C": C, "val_macro_f1": vf1, "val_acc": float((vp == y["val"]).mean())})
        print("linear C", C, "val_macro_f1", vf1, flush=True)
        if vf1 > best_val:
            best_val = vf1
            best_C = C
    clf = LogisticRegression(C=best_C, max_iter=2000, solver="lbfgs")
    clf.fit(Xtr, y["train"])
    lin_pred = clf.predict(Xte).astype(np.int64)
    print("linear chosen C", best_C, flush=True)

    # tensors, global train normalization of logmel
    tr = specs[idx["train"]]
    mu = tr.mean()
    sd = tr.std()
    specs_n = ((specs - mu) / (sd + 1e-6)).astype(np.float32)
    def tensor_split(s):
        x = torch.from_numpy(specs_n[idx[s]][:, None, :, :])
        yy = torch.from_numpy(labels[idx[s]])
        return x, yy

    xtr, ytr = tensor_split("train")
    xva, yva = tensor_split("val")
    xte, yte = tensor_split("test")

    g = torch.Generator()
    g.manual_seed(SEED)
    loader = DataLoader(TensorDataset(xtr, ytr), batch_size=BATCH, shuffle=True, generator=g)

    plain = PlainCNN()
    iterative = IterativeHead()
    pc = nparams(plain)
    ic = nparams(iterative)
    print("params plain", pc, "iterative", ic, flush=True)

    torch.manual_seed(SEED)
    info_plain = train_model(plain, loader, xva, yva, "plain_cnn")
    # rebuild loader so second model sees the same order
    g2 = torch.Generator()
    g2.manual_seed(SEED)
    loader2 = DataLoader(TensorDataset(xtr, ytr), batch_size=BATCH, shuffle=True, generator=g2)
    torch.manual_seed(SEED)
    info_iter = train_model(iterative, loader2, xva, yva, "iterative")

    plain.eval(); iterative.eval()
    with torch.no_grad():
        plain_pred = plain(xte).argmax(1).cpu().numpy().astype(np.int64)
        iter_pred = iterative(xte).argmax(1).cpu().numpy().astype(np.int64)
        plain_val = plain(xva).argmax(1).cpu().numpy().astype(np.int64)
        iter_val = iterative(xva).argmax(1).cpu().numpy().astype(np.int64)

    def pack(name, pred, val_pred=None):
        out = {
            "test_accuracy": float((pred == y["test"]).mean()),
            "test_macro_f1": macro_f1_np(y["test"], pred),
            "per_species_recall": per_species_recall(y["test"], pred),
        }
        if val_pred is not None:
            out["val_accuracy"] = float((val_pred == y["val"]).mean())
            out["val_macro_f1"] = macro_f1_np(y["val"], val_pred)
        return out

    results = {
        "data": {
            "source": "InsectSound1000 official OpenAgrar ZIP (not a fallback)",
            "doi": "10.5073/20231024-173119-0",
            "paper": "10.1038/s41597-024-03301-4",
            "zip_url": "https://www.openagrar.de/servlets/MCRFileNodeServlet/openagrar_derivate_00055778/InsectSound1000.zip",
            "n_files_full_archive_from_central_directory": 165982,
            "subset_clips_per_species": 64,
            "n_species": 12,
            "channel_used": 0,
            "audio": "2.5 s, 16 kHz, IEEE float32, originally 4 channels; channel 0 only",
            "split_counts": {s: int((splits == s).sum()) for s in ("train", "val", "test")},
            "split_per_species": {},
            "split_rule": "file-level, stratified by species, ~70/15/15, seeds sample=0 split=1",
            "test_frozen": True,
        },
        "features": {
            "n_fft": N_FFT, "hop": HOP, "n_mels": N_MELS, "sr": SR,
            "logmel_shape": list(specs.shape[1:]),
            "baseline_features": "mean and std of each of 40 log-mel bands (80-d), StandardScaler fit on train",
            "baseline_C_grid_val": val_grid,
            "baseline_C_selected_on_val": best_C,
            "logmel_train_mean": float(mu),
            "logmel_train_std": float(sd),
        },
        "models": {
            "majority": {
                "label_index": majority_label,
                "label_name": SPECIES[majority_label],
                "train_class_counts": train_counts.tolist(),
                "note": "Train counts are equal, so argmax picks class index 0.",
                **pack("majority", maj_pred),
            },
            "linear_logmel": pack("linear", lin_pred, clf.predict(Xva).astype(np.int64)),
            "plain_cnn": {
                "parameters": pc,
                "description": "Small 2D CNN on log-mel, one forward pass",
                **info_plain,
                **pack("plain", plain_pred, plain_val),
            },
            "iterative_gru_head": {
                "parameters": ic,
                "description": "Same encoder as plain CNN plus a 6-step GRU head with a learned step embedding. Not a Continuous Thought Machine.",
                "steps": STEPS,
                **info_iter,
                **pack("iterative", iter_pred, iter_val),
            },
        },
    }
    # per species split counts
    for sname in SPECIES:
        results["data"]["split_per_species"][sname] = {
            sp: int(((np.array(species) == sname) & (splits == sp)).sum())
            for sp in ("train", "val", "test")
        }

    preds = {
        "majority": maj_pred,
        "linear_logmel": lin_pred,
        "plain_cnn": plain_pred,
        "iterative": iter_pred,
    }
    boot = bootstrap(y["test"], preds, n_boot=1000, seed=0)
    results["bootstrap_test"] = boot
    results["comparison"] = {
        "test_acc_iterative_minus_plain": results["models"]["iterative_gru_head"]["test_accuracy"] - results["models"]["plain_cnn"]["test_accuracy"],
        "test_macro_f1_iterative_minus_plain": results["models"]["iterative_gru_head"]["test_macro_f1"] - results["models"]["plain_cnn"]["test_macro_f1"],
        "test_acc_plain_minus_linear": results["models"]["plain_cnn"]["test_accuracy"] - results["models"]["linear_logmel"]["test_accuracy"],
        "test_n": int(len(y["test"])),
    }

    # drop bulky history from printed file? keep it, it's small
    with open(os.path.join(ROOT, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    # predictions csv
    test_i = idx["test"]
    with open(os.path.join(ROOT, "test_predictions.csv"), "w") as f:
        f.write("file,species,y_true,pred_majority,pred_linear,pred_plain_cnn,pred_iterative\n")
        for local, gi in enumerate(test_i):
            f.write(",".join([
                files[gi],
                species[gi],
                str(int(labels[gi])),
                str(int(maj_pred[local])),
                str(int(lin_pred[local])),
                str(int(plain_pred[local])),
                str(int(iter_pred[local])),
            ]) + "\n")

    torch.save({"plain": plain.state_dict(), "iterative": iterative.state_dict()}, os.path.join(ROOT, "models.pt"))
    print(json.dumps({
        "test_n": results["comparison"]["test_n"],
        "majority_acc": results["models"]["majority"]["test_accuracy"],
        "majority_f1": results["models"]["majority"]["test_macro_f1"],
        "linear_acc": results["models"]["linear_logmel"]["test_accuracy"],
        "linear_f1": results["models"]["linear_logmel"]["test_macro_f1"],
        "plain_acc": results["models"]["plain_cnn"]["test_accuracy"],
        "plain_f1": results["models"]["plain_cnn"]["test_macro_f1"],
        "plain_params": pc,
        "iter_acc": results["models"]["iterative_gru_head"]["test_accuracy"],
        "iter_f1": results["models"]["iterative_gru_head"]["test_macro_f1"],
        "iter_params": ic,
        "diff_acc": results["comparison"]["test_acc_iterative_minus_plain"],
        "diff_ci": boot["acc_iterative_minus_plain_ci95"],
    }, indent=2), flush=True)

if __name__ == "__main__":
    main()
