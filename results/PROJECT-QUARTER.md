# InsectSound1000, one quarter of the archive, three seeds

## Problem

Twelve-way species classification on a larger slice of InsectSound1000 (Branding et al., Scientific Data 2024; dataset DOI 10.5073/20231024-173119-0). Same three systems as the 3,840-clip run: a linear log-mel baseline, a one-pass CNN, and a CTM-style model (private neuron histories, synchronization readout, six internal ticks, loss at the best tick and the most certain tick, test decision at the most certain tick). Not the full Darlow et al. model (arXiv:2505.05522).

## Data and split

Real OpenAgrar ZIP members, channel 0 only. **41495 clips.** Target is 165982/4 = 41495 clips, spread as evenly as archive sizes allow. Myzus persicae (3208), Trialeurodes vaporariorum (1062), and Tuta absoluta (633) are smaller than an equal share, so those species use every clip and the rest are filled to 4065 or 4066. Log-mel n_fft 512, hop 512, 32 bands, shape [32, 78].

Whole recording ids. For each species the id whose full-archive file count is closest to 15% is entirely test; the remaining id closest to 15% is entirely validation; all other ids are train. Clips stay inside those ids. Files already on disk were kept first; any remaining slots were filled by evenly spaced files from the same recording id, up to each species' target (4,065 or 4,066, or every archive file when the species is smaller). No id is in two splits.

Split sizes: **train 28426, val 6226, test 6843.** The test set was frozen. `C` and early stopping used validation macro F1 only. No test score was used to change the protocol.

| species | archive files | sampled | train | val | test | test recording id | used every archive file |
|---|---:|---:|---:|---:|---:|---|---|
| Aphidoletes aphidimyza | 14065 | 4066 | 2962 | 435 | 669 | 202252-18-31 | False |
| Bombus terrestris | 18291 | 4066 | 1735 | 1363 | 968 | 2023427-16-31 | False |
| Bradysia difformis | 11394 | 4066 | 2508 | 925 | 633 | 2022331-13-52 | False |
| Coccinella septempunctata | 14682 | 4066 | 3656 | 192 | 218 | 2023418-15-24 | False |
| Episyrphus balteatus | 16868 | 4066 | 3026 | 477 | 563 | 2023614-16-27 | False |
| Halyomorpha halys | 19671 | 4066 | 2698 | 759 | 609 | 2022421-16-34 | False |
| Myzus persicae | 3208 | 3208 | 2441 | 252 | 515 | 202276-16-47 | True |
| Nezara viridula | 20323 | 4066 | 2852 | 311 | 903 | 2022511-18-10 | False |
| Palomena prasina | 27340 | 4065 | 3075 | 438 | 552 | 202298-17-57 | False |
| Rhaphigaster nebulos | 18445 | 4065 | 2263 | 908 | 894 | 2023104-16-43 | False |
| Trialeurodes vaporariorum | 1062 | 1062 | 752 | 92 | 218 | 2022914-15-32 | True |
| Tuta absoluta | 633 | 633 | 458 | 74 | 101 | 2022714-16-8 | True |

## Methods

This retrain is the same model family as PROJECT-5SEED.md, not a new architecture. Changes that are only about fitting: inverse-frequency class weights on the neural nets, balanced class weights on logistic regression, at most 16 epochs, early stop after 4 epochs without a gain in validation macro F1, batch 64, and the same train-only log-mel augmentation (noise 0.15, time roll of at most 3 frames, 3-bin frequency mask). Three seeds, 0, 1, and 2. Adam, learning rate 0.001, weight decay 1e-4, batch 64. CPU.

- Majority class: `Coccinella_septempunctata`.
- Linear: mean and standard deviation of 32 mel bands. `C` on validation from {0.01, 0.1, 1, 10, 100}. Selected **C = 0.1** (val macro F1 0.398888, val accuracy 0.484741).
- Plain CNN, one pass, **38508 parameters.**
- CTM-style, **40179 parameters**, 6 ticks, D=80, history 8, sync subset 14.

## Results

Frozen test n=6843.

| model | params | test accuracy mean | per seed | test macro F1 mean | per seed |
|---|---:|---:|---|---:|---|
| Majority | 0 | 0.031857 | — | 0.005146 | — |
| Linear | — | 0.419114 | — | 0.405081 | — |
| Plain CNN | 38508 | 0.440645 | 0.426713, 0.442058, 0.453164 | 0.443585 | 0.441856, 0.432797, 0.456104 |
| CTM-style | 40179 | 0.379609 | 0.405085, 0.355984, 0.377758 | 0.339988 | 0.364747, 0.323763, 0.331456 |

Correct counts: majority 218/6843, linear 2868/6843.

- Seed 0. CNN 2920/6843 = 0.426713, macro F1 0.441856 (best val epoch 13, ran 16, 112.9s). CTM-style 2772/6843 = 0.405085, macro F1 0.364747, final-tick accuracy 0.394564 (best val epoch 6, ran 10, 146.0s).
  Bootstrap CTM-style minus CNN, 1000 resamples: mean -0.021471, 95% [-0.031569, -0.011252], includes 0: **False**.
- Seed 1. CNN 3025/6843 = 0.442058, macro F1 0.432797 (best val epoch 8, ran 12, 82.6s). CTM-style 2436/6843 = 0.355984, macro F1 0.323763, final-tick accuracy 0.414438 (best val epoch 3, ran 7, 102.6s).
  Bootstrap CTM-style minus CNN, 1000 resamples: mean -0.086082, 95% [-0.096891, -0.075694], includes 0: **False**.
- Seed 2. CNN 3101/6843 = 0.453164, macro F1 0.456104 (best val epoch 14, ran 16, 101.9s). CTM-style 2585/6843 = 0.377758, macro F1 0.331456, final-tick accuracy 0.310390 (best val epoch 2, ran 6, 92.0s).
  Bootstrap CTM-style minus CNN, 1000 resamples: mean -0.075512, 95% [-0.086808, -0.063565], includes 0: **False**.

Point differences by seed: -0.021628, -0.086073, -0.075406. Mean -0.061036.

The CTM-style model did not win against the plain CNN on this test set.

### Versus the 3,840-clip run

Previous frozen test (n=641), five-seed means for the nets, from metrics_5seed.json: linear accuracy 0.430577, macro F1 0.426535; plain CNN mean accuracy 0.365367, macro F1 0.364671; CTM-style mean accuracy 0.337285, macro F1 0.322283. The user-facing rounded figures were 0.431, 0.365, and 0.337.

- Linear accuracy fell: 0.430577 -> 0.419114 (delta -0.011463). Macro F1 0.426535 -> 0.405081 (delta -0.021454).
- Plain CNN mean accuracy rose: 0.365367 -> 0.440645 (delta +0.075278). Macro F1 0.364671 -> 0.443585 (delta +0.078914).
- CTM-style mean accuracy rose: 0.337285 -> 0.379609 (delta +0.042324). Macro F1 0.322283 -> 0.339988 (delta +0.017705).

These are different test sets (different nights and a larger n), so the delta is not a paired test. It only says whether the headline accuracy on the new frozen test is higher than the headline accuracy on the old one.

### Per-species recall

| species | test clips | majority | linear | plain CNN mean | CTM-style mean |
|---|---:|---:|---:|---:|---:|
| Aphidoletes aphidimyza | 669 | 0.000000 | 0.895366 | 0.926258 | 0.850025 |
| Bombus terrestris | 968 | 0.000000 | 0.403926 | 0.498623 | 0.439394 |
| Bradysia difformis | 633 | 0.000000 | 0.710900 | 0.612428 | 0.477620 |
| Coccinella septempunctata | 218 | 1.000000 | 0.779817 | 0.782875 | 0.304281 |
| Episyrphus balteatus | 563 | 0.000000 | 0.824156 | 0.891652 | 0.811723 |
| Halyomorpha halys | 609 | 0.000000 | 0.146141 | 0.250684 | 0.127531 |
| Myzus persicae | 515 | 0.000000 | 0.025243 | 0.026537 | 0.007767 |
| Nezara viridula | 903 | 0.000000 | 0.043189 | 0.114064 | 0.170543 |
| Palomena prasina | 552 | 0.000000 | 0.302536 | 0.265097 | 0.288647 |
| Rhaphigaster nebulos | 894 | 0.000000 | 0.327740 | 0.287099 | 0.302013 |
| Trialeurodes vaporariorum | 218 | 0.000000 | 0.811927 | 0.657492 | 0.489297 |
| Tuta absoluta | 101 | 0.000000 | 0.158416 | 0.366337 | 0.062706 |

Lowest average recall: Myzus persicae (0.020), Nezara viridula (0.109), Halyomorpha halys (0.175), Tuta absoluta (0.196).

Fraction of test clips whose most certain tick is the last tick: 0.397779, 0.295631, 0.404647, mean 0.366019.
The most certain tick is not always the last tick. There was no early-exit rule, so this is not a measured saving in compute.

## Conclusion

The CTM-style model did not win (mean accuracy 0.379609 vs 0.440645). Linear accuracy was 0.419114.

## Artifacts

- `/workspace/insect-sound/PROJECT-QUARTER.md`
- `/workspace/insect-sound/metrics_quarter.json`
- `/workspace/insect-sound/test_predictions_quarter.csv`
- `/workspace/insect-sound/manifest_quarter.json`

