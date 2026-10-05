# InsectSound1000 quarter subset, frozen pretrained audio probe

## Problem

Same twelve-way species task and the same frozen recording-id split as PROJECT-QUARTER.md. A public AudioSet-pretrained network is frozen. One embedding is taken from each clip. Only a linear classifier or a small MLP is trained. Nothing in the protocol was chosen using the test set.

## Model

EfficientAT MobileNetV3 `mn10_as` (Schmid, Koutini, Widmer, ICASSP 2023), checkpoint `mn10_as_mAP_471.pt` from the v0.0.1 GitHub release. SHA256 `0bd7dc2443af498c289a2e739f02ebb515d6aa3fd3ab9db539c86123ae368a4e`. Backbone parameters frozen: **4876831** (feature extractor 2971664 plus the unused AudioSet classifier 1905167; BatchNorm buffers not counted). Embedding: adaptive average pool of the last convolutional map, **960** dimensions, before the AudioSet classification head. The AudioSet head is not used.

Clips are channel 0 of the 4-channel 2.5 s 16 kHz float32 WAVs. The published front-end expects 32 kHz, so each waveform is linearly resampled from 40,000 to 80,000 samples (`interpolate`, `align_corners=False`). Mel front-end is EfficientAT `AugmentMelSTFT` in eval mode: 128 mels, n_fft 1024, window 800, hop 320, fmax 15000, no frequency or time masking. CPU. No subsampling: all 41,495 manifest clips.

Extraction wall time this run: 229.7s (0 if embeddings were already cached).

## Split

Loaded from `manifest_quarter.json`. Train 28426, val 6226, test 6843. Recording-id overlap across splits: 0. Test n=6843. Embeddings are standardized with a `StandardScaler` fit on train only.

## Head selection (validation only)

Rule, fixed before the test score: highest validation macro F1. Ties break toward higher validation accuracy, then logistic over MLP, then smaller `C` or hidden size.

| head | val macro F1 | val accuracy | val correct |
|---|---:|---:|---:|
| logistic C=0.01 | 0.486315 | 0.600064 | 3736/6226 |
| logistic C=0.1 | 0.488615 | 0.602152 | 3749/6226 |
| logistic C=1.0 | 0.473086 | 0.586251 | 3650/6226 |
| logistic C=10.0 | 0.469290 | 0.581915 | 3623/6226 |
| logistic C=100.0 | 0.468723 | 0.581272 | 3619/6226 |
| MLP hidden=64 (best epoch 8) | 0.508380 | 0.615002 | 3829/6226 |
| MLP hidden=128 (best epoch 3) | 0.504820 | 0.623675 | 3883/6226 |

Selected: **MLP Linear(960->64)-ReLU-Linear(64->12), Adam lr 0.001, weight decay 1e-4, inverse-frequency class weights, early stop on val macro F1 (patience 5, max 40), best epoch 8**. Trainable head parameters: 62284.

Logistic `C=0.01` stopped at 289 lbfgs iterations. `C` in {0.1, 1, 10, 100} hit `max_iter=400` without converging, so those four validation scores are incomplete fits. They were still below the MLP on validation macro F1. The selected head is the MLP, not a non-converged logistic model.

## Frozen test

Full test set, n=6843. Not a subsample.

| model | test accuracy | correct | test macro F1 |
|---|---:|---:|---:|
| Majority | 0.031857 | 218/6843 | 0.005146 |
| Linear log-mel | 0.419114 | 2868/6843 | 0.405081 |
| Plain CNN mean | 0.440645 | — | 0.443585 |
| CTM-style mean | 0.379609 | — | 0.339988 |
| Frozen mn10_as + selected head | 0.475376 | 3253/6843 | 0.446051 |

The frozen pretrained head beats the plain CNN mean accuracy (0.475376 vs 0.440645).

### Per-species recall (selected head, test)

| species | recall |
|---|---:|
| Aphidoletes aphidimyza | 0.872945 |
| Bombus terrestris | 0.574380 |
| Bradysia difformis | 0.652449 |
| Coccinella septempunctata | 0.752294 |
| Episyrphus balteatus | 0.918295 |
| Halyomorpha halys | 0.280788 |
| Myzus persicae | 0.097087 |
| Nezara viridula | 0.184939 |
| Palomena prasina | 0.336957 |
| Rhaphigaster nebulos | 0.404922 |
| Trialeurodes vaporariorum | 0.252294 |
| Tuta absoluta | 0.277228 |

Numbers in this file were written by `pretrain_probe.py` from `metrics_pretrain.json`. Baseline rows were copied from `metrics_quarter.json` at write time. Total wall time 533.2s.
