# InsectSound1000, five seeds, held-out recording nights

## Problem

Twelve-way species classification of 2.5-second clips from InsectSound1000 (Branding et al., Scientific Data 2024, https://doi.org/10.1038/s41597-024-03301-4; data https://doi.org/10.5073/20231024-173119-0). The comparison is a linear baseline, a one-pass CNN, and a small CTM-style network. CTM-style means the ideas in Darlow, Regan, Risi, Seely, and Jones (arXiv:2505.05522, NeurIPS 2025): private weights on each neuron's pre-activation history, synchronization of post-activations as the vector that produces the output, several internal ticks, and a loss that uses both the lowest-loss tick and the most certain tick. It is not the full Sakana model.

## Data and split

Real OpenAgrar ZIP members, channel 0 only. **3840 clips** (target 320 per species). Log-mel n_fft 512, hop 512, 32 bands, shape [32, 78].

Whole recording ids. For each species the id whose full-archive file count is closest to 15% is entirely test; the remaining id closest to 15% is entirely validation; all other ids are train. Clips are then subsampled evenly inside those ids up to 320 per species. No id is in two splits.

Split sizes: **train 2650, val 549, test 641.** Test was frozen. Logistic `C` and early stopping used validation macro F1 only. The test protocol was fixed before any test score was computed.

| species | sampled | train | val | test | test recording id |
|---|---:|---:|---:|---:|---|
| Aphidoletes aphidimyza | 320 | 233 | 34 | 53 | 202252-18-31 |
| Bombus terrestris | 320 | 137 | 107 | 76 | 2023427-16-31 |
| Bradysia difformis | 320 | 197 | 73 | 50 | 2022331-13-52 |
| Coccinella septempunctata | 320 | 288 | 15 | 17 | 2023418-15-24 |
| Episyrphus balteatus | 320 | 238 | 38 | 44 | 2023614-16-27 |
| Halyomorpha halys | 320 | 212 | 60 | 48 | 2022421-16-34 |
| Myzus persicae | 320 | 243 | 25 | 52 | 202276-16-47 |
| Nezara viridula | 320 | 224 | 25 | 71 | 2022511-18-10 |
| Palomena prasina | 320 | 242 | 35 | 43 | 202298-17-57 |
| Rhaphigaster nebulos | 320 | 178 | 72 | 70 | 2023104-16-43 |
| Trialeurodes vaporariorum | 320 | 226 | 28 | 66 | 2022914-15-32 |
| Tuta absoluta | 320 | 232 | 37 | 51 | 2022714-16-8 |

Bombus terrestris and Bradysia difformis have only three recording ids in the archive, so one id is train, one is val, and one is test. Their clip shares follow the archive session sizes, not a 70/15/15 cut.

## Methods

Seeds 0, 1, 2, 3, and 4. Adam, learning rate 0.001, weight decay 1e-4, batch 32, at most 20 epochs, stop after 5 epochs without a validation macro-F1 gain. Train-only augmentation for both nets: Gaussian noise with standard deviation 0.15 on the normalized log-mel, a time roll of at most 3 frames, and a 3-bin frequency mask. Inverse-frequency class weights for the nets. Logistic regression uses balanced class weights. CPU.

- Majority class: `Coccinella_septempunctata`.
- Linear: mean and standard deviation of 32 mel bands. `C` selected on val from {0.01, 0.1, 1, 10, 100}. Selected **C = 10.0** (val macro F1 0.427158).
- Plain CNN, one forward pass, **38508 parameters.**
- CTM-style, **40179 parameters**, 6 ticks, D=80, history length 8, sync on 14 neurons (upper triangle). Output is a linear map of the synchronization vector. Test decision is the class at the most certain tick. Loss is the mean of cross-entropy at the best tick and at the most certain tick.

Simplified versus the paper: one GLU synapse MLP instead of a U-Net, a shallow private GLU instead of a wide deep neuron-level MLP, and no ResNet. The two nets are the same order of size; the exact counts are above.

## Results

Frozen test n=641.

| model | params | test accuracy mean | per seed | test macro F1 mean | per seed |
|---|---:|---:|---|---:|---|
| Majority | 0 | 0.026521 | — | 0.004306 | — |
| Linear | — | 0.430577 | — | 0.426535 | — |
| Plain CNN | 38508 | 0.365367 | 0.343214, 0.369735, 0.390016, 0.361934, 0.361934 | 0.364671 | 0.357611, 0.347995, 0.410699, 0.351905, 0.355147 |
| CTM-style | 40179 | 0.337285 | 0.340094, 0.326053, 0.326053, 0.357254, 0.336973 | 0.322283 | 0.295279, 0.313456, 0.335381, 0.336744, 0.330556 |

Correct counts: majority 17/641, linear 276/641.

- Seed 0. CNN 220/641 = 0.343214, macro F1 0.357611 (val epoch 19, ran 20, 13.4s). CTM-style 218/641 = 0.340094, macro F1 0.295279, final-tick accuracy 0.360374 (val epoch 9, ran 14, 27.7s).
  Bootstrap CTM-style minus CNN, 1000 resamples: mean -0.003406, 95% [-0.040562, 0.034321], includes 0: **True**.
- Seed 1. CNN 237/641 = 0.369735, macro F1 0.347995 (val epoch 2, ran 7, 5.0s). CTM-style 209/641 = 0.326053, macro F1 0.313456, final-tick accuracy 0.294852 (val epoch 5, ran 10, 19.3s).
  Bootstrap CTM-style minus CNN, 1000 resamples: mean -0.043184, 95% [-0.078003, -0.007800], includes 0: **False**.
- Seed 2. CNN 250/641 = 0.390016, macro F1 0.410699 (val epoch 17, ran 20, 13.7s). CTM-style 209/641 = 0.326053, macro F1 0.335381, final-tick accuracy 0.291732 (val epoch 7, ran 12, 26.9s).
  Bootstrap CTM-style minus CNN, 1000 resamples: mean -0.063752, 95% [-0.095164, -0.031201], includes 0: **False**.
- Seed 3. CNN 232/641 = 0.361934, macro F1 0.351905 (val epoch 2, ran 7, 4.3s). CTM-style 229/641 = 0.357254, macro F1 0.336744, final-tick accuracy 0.288612 (val epoch 9, ran 14, 25.3s).
  Bootstrap CTM-style minus CNN, 1000 resamples: mean -0.004424, 95% [-0.043682, 0.032761], includes 0: **True**.
- Seed 4. CNN 232/641 = 0.361934, macro F1 0.355147 (val epoch 5, ran 10, 5.9s). CTM-style 216/641 = 0.336973, macro F1 0.330556, final-tick accuracy 0.347894 (val epoch 6, ran 11, 20.0s).
  Bootstrap CTM-style minus CNN, 1000 resamples: mean -0.023796, 95% [-0.054602, 0.006240], includes 0: **True**.

Point differences (CTM-style minus CNN) by seed: -0.003120, -0.043682, -0.063963, -0.004680, -0.024961. Mean -0.028081.

At least one seed's 95% interval includes 0, and the seeds do not agree. **The CTM-style model did not significantly win.**

### Per-species recall

Mean over five seeds for the neural nets. Denominator is that species' test clips.

| species | test clips | majority | linear | plain CNN | CTM-style |
|---|---:|---:|---:|---:|---:|
| Aphidoletes aphidimyza | 53 | 0.000000 | 0.867925 | 0.871698 | 0.905660 |
| Bombus terrestris | 76 | 0.000000 | 0.368421 | 0.442105 | 0.476316 |
| Bradysia difformis | 50 | 0.000000 | 0.800000 | 0.500000 | 0.348000 |
| Coccinella septempunctata | 17 | 1.000000 | 0.882353 | 0.752941 | 0.317647 |
| Episyrphus balteatus | 44 | 0.000000 | 0.886364 | 0.804545 | 0.781818 |
| Halyomorpha halys | 48 | 0.000000 | 0.125000 | 0.291667 | 0.195833 |
| Myzus persicae | 52 | 0.000000 | 0.153846 | 0.003846 | 0.000000 |
| Nezara viridula | 71 | 0.000000 | 0.070423 | 0.059155 | 0.067606 |
| Palomena prasina | 43 | 0.000000 | 0.279070 | 0.232558 | 0.255814 |
| Rhaphigaster nebulos | 70 | 0.000000 | 0.328571 | 0.122857 | 0.091429 |
| Trialeurodes vaporariorum | 66 | 0.000000 | 0.696970 | 0.487879 | 0.496970 |
| Tuta absoluta | 51 | 0.000000 | 0.156863 | 0.235294 | 0.203922 |

Lowest average recall: Myzus persicae (0.053), Nezara viridula (0.066), Rhaphigaster nebulos (0.181), Tuta absoluta (0.199).

### Certainty across ticks

Fraction of test clips whose most certain tick is the last tick: 0.673947, 0.358814, 0.664587, 0.549142, 0.563183, mean 0.561934.
- Seed 0 mean certainty by tick: 0.010, 0.080, 0.242, 0.246, 0.375, 0.487. Chosen-tick histogram [0, 0, 61, 97, 51, 432].
- Seed 1 mean certainty by tick: 0.008, 0.017, 0.181, 0.292, 0.313, 0.359. Chosen-tick histogram [0, 0, 84, 145, 182, 230].
- Seed 2 mean certainty by tick: 0.002, 0.048, 0.115, 0.208, 0.375, 0.444. Chosen-tick histogram [0, 0, 0, 65, 150, 426].
- Seed 3 mean certainty by tick: 0.134, 0.068, 0.213, 0.422, 0.395, 0.492. Chosen-tick histogram [3, 0, 4, 177, 105, 352].
- Seed 4 mean certainty by tick: 0.003, 0.022, 0.160, 0.377, 0.408, 0.438. Chosen-tick histogram [0, 0, 0, 131, 149, 361].

The most certain tick is not always the last tick. That is only a description of where certainty peaked. There was no early-exit rule and no measured compute saving.

## Conclusion

The CTM-style model did not significantly beat the plain CNN on the frozen night hold-out (n=641): mean accuracy 0.337285 versus 0.365367. The linear baseline scored 0.430577 accuracy and 0.426535 macro F1.

## Artifacts

- `/workspace/insect-sound/PROJECT-5SEED.md`
- `/workspace/insect-sound/metrics_5seed.json`
- `/workspace/insect-sound/test_predictions_5seed.csv`
- `/workspace/insect-sound/manifest_5seed.json`

