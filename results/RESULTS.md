# InsectSound1000 small subset: plain CNN vs a light iterative head

## Problem frame

12-way species classification on short clips from InsectSound1000 (Branding et al., Scientific Data 2024, https://doi.org/10.1038/s41597-024-03301-4; dataset DOI https://doi.org/10.5073/20231024-173119-0).

This is a small CPU experiment in the style of a two-system comparison plus a simple baseline. The second neural model is a **small recurrent/iterative head** (6 GRU steps on the encoder embedding). It is **not** a Continuous Thought Machine and is **not** a reproduction of Darlow et al.

Question: on a frozen file-level test split, does that iterative head beat a plain CNN of similar size by enough to matter, relative to a linear baseline?

## Data

Real InsectSound1000, not a fallback. The official OpenAgrar release is a single ZIP (`InsectSound1000.zip`, derivate `openagrar_derivate_00055778`, 97,478,572,235 bytes, 165,982 WAV members in the ZIP central directory, all deflate). Individual members were fetched with HTTP range requests and inflated. The full archive was not downloaded.

Subset: **64 clips per species × 12 species = 768 files**. Each file is a 2.5 s, 16 kHz, 32-bit float, 4-channel WAV (640,058 bytes). Models used **channel 0 only**. Clips were spaced across the few recording-ids in each species (3–10 ids per species in the full set), not taken as one contiguous run.

Split is **by file**, stratified by species, seeds `sample=0`, `split=1`:

| split | clips | per species |
|---|---:|---:|
| train | 540 | 45 |
| val | 120 | 10 |
| test | 108 | 9 |

Test was frozen: model selection used validation only (logistic regression `C` grid; neural nets early-stopped on validation macro F1). Test was scored once.

Caveat: this is **not** a recording-night or individual hold-out. Every one of the 108 test files shares its recording-id with at least one training file of the same species. Absolute accuracy is likely optimistic for a new night or new insects. The same-file window leak the split was meant to avoid does not occur: each WAV is entirely in one split.

## Method

Log-mel spectrogram: n_fft=512, hop=256, n_mels=40, Hann window, 20–8000 Hz mel filterbank, natural log of power. Spectrogram shape per clip: 40 × 155.

- **Majority baseline.** Training counts are exactly 45 per class, so `argmax` selects class index 0 (`Aphidoletes_aphidimyza`). No features.
- **Linear baseline.** 80 features = mean and standard deviation of each of 40 log-mel bands. `StandardScaler` fit on train. `LogisticRegression` (lbfgs). `C` chosen on validation macro F1 from {0.01, 0.1, 1, 10}; selected **C=10.0** (val macro F1 0.501108).
- **Plain CNN.** One forward pass. Three conv blocks (16, 32, 48 channels, 3×3, batch norm, ReLU, two 2×2 pools), adaptive average pool to 4×4, linear to 96-d, then an MLP 96→140→140→12. **127700 parameters.**
- **Iterative head.** The same encoder, then a learned 8-d step embedding concatenated with the 96-d vector and fed to a 1-layer GRU (hidden 64) for **6 steps**. Class logits come from the last GRU state. **126156 parameters.**

Neural nets: Adam lr=0.001, weight decay 1e-4, batch 32, cross-entropy, max 30 epochs, early stop patience 5 on validation macro F1, seed 0. Best validation checkpoint restored before the test pass. CPU only (no CUDA). Plain CNN best epoch 14 (ran 19 epochs, 8.00 s). Iterative head best epoch 11 (ran 16 epochs, 6.63 s).

## Results (frozen test, n=108)

Correct counts are out of 108. Macro F1 uses labels 0–11 with zero_division=0. Bootstrap is 1000 resamples of the test set (seed 0); intervals are 2.5 and 97.5 percentiles.

| model | params | test correct | test accuracy | accuracy 95% bootstrap interval | test macro F1 | macro F1 95% bootstrap interval |
|---|---:|---:|---:|---|---:|---|
| Majority class | 0 | 9 | 0.083333 | [0.037037, 0.138889] | 0.012821 | [0.005952, 0.020325] |
| Linear log-mel mean/std | — | 52 | 0.481481 | [0.388889, 0.574074] | 0.499824 | [0.413054, 0.566276] |
| Plain CNN | 127700 | 48 | 0.444444 | [0.351852, 0.527778] | 0.403512 | [0.323517, 0.466408] |
| Iterative GRU head (6 steps) | 126156 | 51 | 0.472222 | [0.379630, 0.564815] | 0.468877 | [0.381509, 0.535668] |

Accuracy difference, iterative minus plain CNN: **0.027778** (51−48 = 3 clips). Bootstrap 95% interval **[-0.064815, 0.120370]** (bootstrap mean 0.029167). The interval includes 0.

Macro F1 difference, iterative minus plain CNN: **0.065365**.

Plain CNN minus linear accuracy: **-0.037037** (48−52 = −4 clips). The three non-majority accuracy intervals overlap heavily. Do not treat any pairwise gap as significant.

Per-species recall on the test set (correct / 9). Pentatomid bugs are the hard group for these tiny models.

| species | majority | linear | plain CNN | iterative |
|---|---:|---:|---:|---:|
| Aphidoletes_aphidimyza | 1.000000 | 0.666667 | 0.333333 | 0.666667 |
| Bombus_terrestris | 0.000000 | 0.666667 | 0.777778 | 0.777778 |
| Bradysia_difformis | 0.000000 | 0.888889 | 0.888889 | 0.888889 |
| Coccinella_septempunctata | 0.000000 | 0.555556 | 0.777778 | 0.666667 |
| Episyrphus_balteatus | 0.000000 | 0.666667 | 0.666667 | 0.666667 |
| Halyomorpha_halys | 0.000000 | 0.555556 | 0.000000 | 0.000000 |
| Myzus_persicae | 0.000000 | 0.333333 | 0.000000 | 0.111111 |
| Nezara_viridula | 0.000000 | 0.000000 | 0.222222 | 0.222222 |
| Palomena_prasina | 0.000000 | 0.222222 | 0.111111 | 0.555556 |
| Rhaphigaster_nebulos | 0.000000 | 0.222222 | 0.000000 | 0.222222 |
| Trialeurodes_vaporariorum | 0.000000 | 0.666667 | 0.555556 | 0.666667 |
| Tuta_absoluta | 0.000000 | 0.333333 | 1.000000 | 0.222222 |

Validation scores at the selected checkpoint (not used as the reported test result):

- Linear: val accuracy 0.483333, val macro F1 0.501108
- Plain CNN: val accuracy 0.483333, val macro F1 0.446348
- Iterative: val accuracy 0.433333, val macro F1 0.439885

## Conclusion

No. The iterative head did not beat the plain CNN by enough to matter. It was right on 3 more test clips (accuracy 0.472222 vs 0.444444) and its macro F1 was higher by 0.065365, but the test set has only 108 clips and the 95% bootstrap interval on the accuracy difference is about −0.065 to +0.120 and includes zero. Both networks also landed at or below the linear model on mel-band mean and standard deviation (52/108, accuracy 0.481481, macro F1 0.499824). With this subset size, this training budget, and this file-level (not night-level) split, a light GRU head is not a meaningful upgrade over a single forward pass, and neither neural net clearly beat a linear baseline.

## Artifacts

- `/workspace/insect-sound/raw/` — 768 extracted WAVs
- `/workspace/insect-sound/manifest.json` — file, species, split, ZIP offsets
- `/workspace/insect-sound/metrics.json` — full numeric record
- `/workspace/insect-sound/test_predictions.csv` — 108 test rows
- `/workspace/insect-sound/models.pt` — selected state dicts
- `/workspace/insect-sound/train_eval.py`, `download_subset.py`

