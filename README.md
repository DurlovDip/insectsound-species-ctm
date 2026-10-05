# Insect Species Classification from Sound (InsectSound1000)

Course project for **CSE 5017 Applied Machine Learning**, University of Dhaka (instructor: Dr. Md. Mosaddek Khan), by **Dip Durlov**.

This project classifies 2.5-second, 16 kHz clips from the public InsectSound1000 dataset (Branding et al., *Scientific Data* 2024) into 12 insect species. Whole recording nights (recording ids) are held out, so no night appears in more than one of train, validation, and test. Four approaches are compared on the same split: (1) a linear baseline (logistic regression) on the average loudness of 32 log-mel bands (per-band mean and standard deviation over time), (2) a one-pass plain CNN (38,508 parameters), (3) a CTM-style model (40,179 parameters), which is a small version of the Continuous Thought Machine idea (Darlow et al., NeurIPS 2025) with private neuron histories, a synchronization readout and six internal ticks, not the full model, and (4) a frozen EfficientAT `mn10_as` AudioSet-pretrained network with a small trained head. On the 41,495-clip "quarter" set (test: 6,843 clips), the frozen pretrained model with a small MLP head scored highest (47.5%). The plain CNN came next (44.1% mean over 3 seeds), then the linear baseline (41.9%). The CTM-style model scored 38.0%.

## Main results (quarter set, frozen test n = 6,843)

All numbers below come from `results/metrics_quarter.json`, `results/metrics_pretrain.json` and `results/metrics_pretrain_ctm.json` (reports: `results/PROJECT-QUARTER.md`, `results/PROJECT-PRETRAIN.md`, `results/PROJECT-PRETRAIN-CTM.md`).

| Model | Trainable params | Test accuracy | Test macro F1 |
|---|---:|---:|---:|
| Majority class (*Coccinella septempunctata*) | 0 | 3.2% (218/6,843) | 0.005 |
| Linear (log-mel band mean/std, logistic regression, C = 0.1) | n/a | 41.9% (2,868/6,843) | 0.405 |
| Plain CNN, one pass, mean of 3 seeds | 38,508 | 44.1% (seeds: 42.7 / 44.2 / 45.3%) | 0.444 |
| CTM-style, 6 ticks, mean of 3 seeds | 40,179 | 38.0% (seeds: 40.5 / 35.6 / 37.8%) | 0.340 |
| Frozen EfficientAT `mn10_as` + MLP head (960→64→12) | 62,284 (head only) | **47.5% (3,253/6,843)** | 0.446 |
| Frozen EfficientAT `mn10_as` + CTM-style head, mean of 3 seeds | 21,959 (head only) | 38.0% | 0.353 |

Exact values: majority 0.031857, linear 0.419114, CNN mean 0.440645, CTM-style mean 0.379609, frozen pretrained + MLP 0.475376, pretrained + CTM-style head mean 0.380389.

## Honest notes

- **Night-level hold-out.** In the quarter, 5-seed and session runs the split uses whole recording ids. For each species, the recording id whose file count is closest to 15% of that species' files is used only for test, and the next id closest to 15% only for validation. Model selection (C, early stopping, head choice) used validation macro F1 only. The test set was scored once and never used for tuning. Because there are only a few nights per species, test accuracy reflects how well a model transfers to a new night, and it varies a lot by species (see the per-species recall tables in the reports). The first small run (`train_eval.py`, `results/RESULTS.md`) used a per-file split and is kept only for the record.
- **Channel 0 only.** The InsectSound1000 WAVs have 4 channels. Every model here uses channel 0 only.
- **CTM-style is a small version, not the full model.** It borrows the main ideas of Continuous Thought Machines (private per-neuron weights on a short pre-activation history, a synchronization readout, several internal ticks, a loss taken at the lowest-loss tick and the most certain tick, and a test decision at the most certain tick). It leaves out the U-Net synapse, the large width, the ResNet backbone and other parts of the paper's model. It is not a reproduction of Darlow et al.
- **Extra ticks did not beat one pass.** In every seed the CTM-style model scored below the one-pass CNN of about the same size (point differences −0.022, −0.086 and −0.075; for each seed the bootstrap 95% interval excludes 0). On frozen pretrained embeddings, the CTM-style head also scored below a plain MLP head.
- The frozen-backbone result shows how much a pretrained network helps. It does not show that the CTM idea works. The backbone (about 4.9 M parameters, pretrained on AudioSet) was not trained here.
- Everything ran on a CPU, with 3 seeds for the quarter-set networks. The spread across seeds is large (CNN 42.7–45.3%, CTM-style 35.6–40.5%).

## Repository layout

```
.
├── README.md, LICENSE, requirements.txt, .gitignore
├── download_subset.py      # 768-clip subset (64/species) via HTTP range requests on the official ZIP -> manifest.json
├── expand_subset.py        # larger subset with recording-id split -> manifest_5seed.json
├── expand_10x.py           # ~10x subset -> manifest_10x.json (intermediate; no report built on it here)
├── expand_20k.py           # ~20k subset -> manifest_20k.json
├── expand_quarter.py       # one quarter of the archive (41,495 clips) -> manifest_quarter.json
├── train_eval.py           # first run: linear, plain CNN, small GRU "iterative" head, per-file split -> RESULTS.md
├── session_train.py        # recording-id split, linear vs CNN vs CTM-style, 2 seeds -> PROJECT.md
├── train_5seed.py          # same comparison on manifest_5seed.json, 5 seeds -> PROJECT-5SEED.md
├── train_20k.py            # same comparison on manifest_20k.json (its outputs are not included here)
├── train_quarter.py        # main run on manifest_quarter.json, 3 seeds -> PROJECT-QUARTER.md
├── pretrain_probe.py       # frozen EfficientAT mn10_as embeddings + logistic/MLP head -> PROJECT-PRETRAIN.md
├── pretrain_ctm_head.py    # CTM-style head on the same frozen embeddings -> PROJECT-PRETRAIN-CTM.md
├── results/                # PROJECT*.md and RESULTS.md reports, metrics*.json, session_metrics.json,
│                           # test prediction CSVs (one row per test clip, true label and predictions)
├── manifests/              # manifest*.json, session_manifest.json: clip file names, species, split
│                           # (and the ZIP member offsets used for range download); no audio
└── figures/                # spectrogram-bombus.png, spectrogram-halyomorpha.png
```

No audio, cached features, embeddings, trained model files (`*.pt`) or third-party checkpoints are included.

## How to reproduce

**Requirements.** Python 3 (developed with 3.13), plus `numpy`, `torch` and `scikit-learn` (`pip install -r requirements.txt`). Only `pretrain_probe.py` also needs the EfficientAT code and its dependencies (`torchaudio`, `torchvision`), as described below.

**Paths.** The scripts hardcode their data folder as `ROOT = "/workspace/insect-sound"` (or as `INDEX`/`OUT`/`MANIFEST` constants near the top of the download scripts, plus a few artifact paths further down in some scripts). Change these so they point at your own data folder. The scripts expect a flat layout: the WAV clips in `<ROOT>/raw/`, and the manifest and metrics JSON files directly in `<ROOT>/` (copy them there from `manifests/` and `results/` in this repo). Reports, metrics and models are written back to `<ROOT>/`. The training logic does not depend on the path.

**1. Get the data.** InsectSound1000 is available from the data DOI https://doi.org/10.5073/20231024-173119-0 (OpenAgrar, one ZIP of about 97 GB with 165,982 WAV files) or from Kaggle: https://www.kaggle.com/datasets/hesi0ne/insectsound1000. This repo does not include any audio.

- *Option A (recommended, exact same clips):* download or extract the archive, and copy the clips listed in `manifests/manifest_quarter.json` (field `base` = file name) into `<ROOT>/raw/`. Then copy `manifests/manifest_quarter.json` to `<ROOT>/`. Do **not** rerun `expand_quarter.py` in this case. It rebuilds and overwrites the manifest, and it prefers files that are already on disk, so a rerun can give a different selection.
- *Option B (how the clips were originally fetched):* `download_subset.py` and `expand_*.py` fetch single ZIP members from the OpenAgrar URL with HTTP range requests, so the whole archive is never downloaded. They need two things that are not in this repo: a one-line-per-member JSON index of the ZIP central directory at `<ROOT>/cd/index.jsonl` (fields `n` name, `m` method, `c` compressed size, `u` uncompressed size, `o` local-header offset), and an OpenAgrar session cookie file at `/tmp/oa_cookies.txt`.

**2. Run order** (each script writes its report and metrics into `<ROOT>/`):

1. `python download_subset.py` → `python train_eval.py` (first per-file run, `RESULTS.md`)
2. `python session_train.py` (recording-id split on the 768 clips, `PROJECT.md`)
3. `python expand_subset.py` → `python train_5seed.py` (`PROJECT-5SEED.md`)
4. *(optional)* `python expand_10x.py`, `python expand_20k.py` → `python train_20k.py`
5. `python expand_quarter.py` (or Option A above) → `python train_quarter.py` (main log-mel results, `PROJECT-QUARTER.md`, `metrics_quarter.json`)
6. `python pretrain_probe.py` (frozen EfficientAT embeddings + head, `PROJECT-PRETRAIN.md`; needs `metrics_quarter.json`)
7. `python pretrain_ctm_head.py` (CTM-style head on the cached embeddings, `PROJECT-PRETRAIN-CTM.md`; needs `metrics_pretrain.json` and `metrics_quarter.json`)

**3. Pretrained checkpoint (only for step 6).** The EfficientAT checkpoint `mn10_as_mAP_471.pt` is third-party and is not included. It comes from the EfficientAT **v0.0.1** GitHub release: https://github.com/fschmid56/EfficientAT/releases/download/v0.0.1/mn10_as_mAP_471.pt. `pretrain_probe.py` gets it as follows:

- It imports the EfficientAT code from a clone at `/tmp/EfficientAT` (`git clone https://github.com/fschmid56/EfficientAT /tmp/EfficientAT`; the `EFFICIENTAT` constant sets the path). It then calls EfficientAT's `get_model(pretrained_name="mn10_as", ...)`. That call downloads the checkpoint from the v0.0.1 release into `/tmp/EfficientAT/resources/` through `torch.hub` if the file is not already there.
- The script also expects its own copy at `<ROOT>/weights/mn10_as_mAP_471.pt`, which you must download yourself from the URL above. It stops if that file is missing, or if its SHA-256 differs from the copy in `resources/`. The checkpoint used for the reported results had SHA-256 `0bd7dc2443af498c289a2e739f02ebb515d6aa3fd3ab9db539c86123ae368a4e`.

The 960-d embeddings are cached in `<ROOT>/embeddings_mn10_as.f32` (about 160 MB, not included), and `pretrain_ctm_head.py` reuses them.

## Citations

- **Dataset:** Branding, J., von Hörsten, D., Böckmann, E., Wegener, J. K., & Hartung, E. (2024). InsectSound1000: An insect sound dataset for deep learning based acoustic insect recognition. *Scientific Data*, 11, 475. https://doi.org/10.1038/s41597-024-03301-4. Data: https://doi.org/10.5073/20231024-173119-0. Kaggle mirror: https://www.kaggle.com/datasets/hesi0ne/insectsound1000
- **Continuous Thought Machines:** Darlow, L., Regan, C., Risi, S., Seely, J., & Jones, L. (2025). Continuous Thought Machines. *NeurIPS 2025*. arXiv:2505.05522. https://arxiv.org/abs/2505.05522
- **EfficientAT:** Schmid, F., Koutini, K., & Widmer, G. (2023). Efficient Large-Scale Audio Tagging via Transformer-to-CNN Knowledge Distillation. *ICASSP 2023*. Code and checkpoints: https://github.com/fschmid56/EfficientAT

## License

The code and results in this repository are released under the MIT License (see `LICENSE`). The InsectSound1000 data and the EfficientAT checkpoint are third-party and keep their own licenses. Neither is redistributed here.
