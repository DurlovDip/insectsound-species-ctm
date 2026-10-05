# Insect sounds, held-out nights: a small CTM-style model vs a plain CNN

## Problem

Can a small model that thinks in internal ticks tell 12 insect species apart from a 2.5-second microphone clip, when it is not allowed to train on the same recording night as the test clip?

The data are InsectSound1000 (Branding et al., Scientific Data 2024, https://doi.org/10.1038/s41597-024-03301-4; dataset https://doi.org/10.5073/20231024-173119-0). The neural comparison follows the idea in Continuous Thought Machines (Darlow, Regan, Risi, Seely, Jones; Sakana AI; arXiv:2505.05522, NeurIPS 2025), at a scale that fits a CPU. This is not the full Darlow model. It only borrows the experimental shape: a simple baseline, a one-pass network, and a second network of similar size whose extra machinery is internal time.

## Data

Real InsectSound1000, not a fallback. Clips already on disk from the official OpenAgrar ZIP (`openagrar_derivate_00055778`), fetched earlier by HTTP range and inflate. **No further download.** Each clip is one WAV: 2.5 s, 16 kHz, 32-bit float, 4 channels. Models use **channel 0 only**.

Subset size: **64 clips × 12 species = 768 files.** Log-mel: n_fft 512, hop 512, 32 mels, shape 32 × 78.

## Split

Recording-id hold-out. Within each species, the recording id whose on-disk clip count is closest to 15% of that species' 64 clips is the entire test session; the remaining id closest to 15% is the entire validation session; all other recording ids are train. No recording id appears in more than one split. With only 3 ids (Bombus, Bradysia) this cannot be 70/15/15 of clips.

Counts: **train 485, val 142, test 141.** Recording-id overlap between train and test: **0.** The test set was frozen. `C` for logistic regression and early stopping for the networks used validation macro F1 only.

| species | ids | train clips | val clips | test clips | test recording id |
|---|---:|---:|---:|---:|---|
| Aphidoletes aphidimyza | 5 | 39 | 13 | 12 | 202253-16-31 |
| Bombus terrestris | 3 | 22 | 21 | 21 | 2023510-16-19 |
| Bradysia difformis | 3 | 22 | 21 | 21 | 2022330-13-52 |
| Coccinella septempunctata | 7 | 40 | 12 | 12 | 2023418-15-24 |
| Episyrphus balteatus | 6 | 44 | 10 | 10 | 2023515-16-42 |
| Halyomorpha halys | 7 | 45 | 9 | 10 | 2022425-18-24 |
| Myzus persicae | 5 | 39 | 13 | 12 | 2022224-14-57 |
| Nezara viridula | 8 | 48 | 8 | 8 | 2022511-18-10 |
| Palomena prasina | 10 | 50 | 7 | 7 | 2022811-16-28 |
| Rhaphigaster nebulos | 6 | 44 | 10 | 10 | 2023104-16-43 |
| Trialeurodes vaporariorum | 6 | 44 | 10 | 10 | 2022912-16-11 |
| Tuta absoluta | 8 | 48 | 8 | 8 | 2022627-17-3 |

Species with only three recording ids in this subset (Bombus terrestris, Bradysia difformis) cannot yield a 70/15/15 clip split. One id each goes to train, val, and test, so those species contribute about a third of their clips to test. Coccinella has two tiny ids (1 and 2 clips); the rule did not pick those as the test night, because a larger id was closer to 15%.

This is a harder protocol than the earlier file-level split of the same 768 clips, where every test file shared a recording id with train. That earlier run (linear 52/108, CNN 48/108, GRU 51/108) is not repeated here and is not the result below.

## Methods

All three scored models see the same log-mel. Train-set class weights (inverse frequency) are used for the neural nets. Logistic regression uses `class_weight='balanced'`. Two training seeds, 0 and 1. Adam, learning rate 0.001, weight decay 1e-4, batch 32, at most 12 epochs, early stop after 3 epochs without a validation macro-F1 improvement. CPU only.

- **Majority.** Most frequent training class (`Palomena_prasina`). Because the session split is not balanced, this is not exactly 1/12.
- **Linear baseline.** Mean and standard deviation of each of 32 mel bands (64 features), standard-scaled on train. `C` chosen on validation from {0.01, 0.1, 1, 10, 100}. Selected **C = 0.1**.
- **Plain CNN.** One forward pass. Conv 16 then 32, two pools, adaptive pool to 4×4, linear 512→64→12. **38508 parameters.**
- **CTM-style.** **28081 parameters.** 6 internal ticks. Width D=64. Each neuron keeps a length-8 FIFO of pre-activations and applies its **own** weights (a private GLU) to that history. A shared synapse MLP mixes an attention readout with the post-activation vector to produce the next pre-activation. Synchronization is a decayed upper-triangle of pairwise products of post-activations, on the first 12 neurons for the output and the last 12 for the attention query. Class logits are a linear map of the **output synchronization vector**, not of the neuron activations themselves. Certainty is 1 minus entropy divided by log(12).

Loss for the CTM-style model, as in the paper: half the cross-entropy at the tick with the lowest loss, plus half the cross-entropy at the tick with the highest certainty. At test time the label is not available, so the reported decision is the class at the **most certain tick**. Final-tick accuracy is reported only as a diagnostic.

What was left out of the Sakana CTM on purpose: the U-Net synapse, deep wide neuron-level MLPs, ImageNet-scale width (D=2048, M=25, T=75), learnable positional embeddings, and a ResNet backbone. The name in this report is **CTM-style**, not Continuous Thought Machine.

## Results

Frozen test set, n=141. Neural numbers are two seeds. Spread is max minus min.

| model | params | test accuracy | test macro F1 | correct / n |
|---|---:|---|---|---|
| Majority | 0 | 0.049645 | 0.007883 | 7/141 |
| Linear log-mel | — | 0.319149 | 0.265281 | 45/141 |
| Plain CNN | 38508 | mean 0.375887 (seeds 0.326241, 0.425532) | mean 0.314393 (seeds 0.258975, 0.369811) | see seeds |
| CTM-style, most certain tick | 28081 | mean 0.361702 (seeds 0.397163, 0.326241) | mean 0.291941 (seeds 0.302929, 0.280952) | see seeds |

Seed detail:

- Seed 0. Plain CNN 46/141 = 0.326241, macro F1 0.258975 (best val epoch 8, ran 11, 1.4s). CTM-style most-certain 56/141 = 0.397163, macro F1 0.302929; final-tick accuracy 0.397163 (best val epoch 9, ran 12, 4.0s).
  Bootstrap of CTM-style minus plain accuracy, 1000 resamples of this seed's test predictions: mean 0.070383, 95% interval [-0.007092, 0.148936]. Includes 0: **True**.
- Seed 1. Plain CNN 60/141 = 0.425532, macro F1 0.369811 (best val epoch 8, ran 11, 1.3s). CTM-style most-certain 46/141 = 0.326241, macro F1 0.280952; final-tick accuracy 0.276596 (best val epoch 8, ran 11, 3.7s).
  Bootstrap of CTM-style minus plain accuracy, 1000 resamples of this seed's test predictions: mean -0.098957, 95% interval [-0.177305, -0.028369]. Includes 0: **False**.

Point difference (CTM-style minus plain), two seeds: 0.070922 and -0.099291, mean -0.014184.

Seed 0's interval includes 0, so that seed's gain for the CTM-style model is not significant. Seed 1's interval lies entirely below 0, which means the plain CNN was ahead on that seed, not the CTM-style model. **The CTM-style model did not significantly win.** The two seeds also disagree on which network is better (spread of plain accuracy 0.099291; spread of CTM-style accuracy 0.070922).

### Per-species recall

Recall is correct predictions divided by the number of test clips of that species. The table is the mean of the two seeds for the neural nets. Low recall is a hard species under this night hold-out, not a claim about the full 166k-file corpus.

| species | test clips | majority | linear | plain CNN mean | CTM-style mean |
|---|---:|---:|---:|---:|---:|
| Aphidoletes aphidimyza | 12 | 0.000000 | 1.000000 | 1.000000 | 0.750000 |
| Bombus terrestris | 21 | 0.000000 | 0.190476 | 0.571429 | 0.833333 |
| Bradysia difformis | 21 | 0.000000 | 0.142857 | 0.119048 | 0.047619 |
| Coccinella septempunctata | 12 | 0.000000 | 0.916667 | 0.750000 | 0.333333 |
| Episyrphus balteatus | 10 | 0.000000 | 0.200000 | 0.400000 | 0.300000 |
| Halyomorpha halys | 10 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| Myzus persicae | 12 | 0.000000 | 0.083333 | 0.083333 | 0.083333 |
| Nezara viridula | 8 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| Palomena prasina | 7 | 1.000000 | 0.428571 | 0.142857 | 0.428571 |
| Rhaphigaster nebulos | 10 | 0.000000 | 0.000000 | 0.050000 | 0.000000 |
| Trialeurodes vaporariorum | 10 | 0.000000 | 0.800000 | 0.550000 | 0.600000 |
| Tuta absoluta | 8 | 0.000000 | 0.125000 | 0.687500 | 0.812500 |

Hardest species by average recall of linear, CNN, and CTM-style: Halyomorpha halys (0.000), Nezara viridula (0.000), Rhaphigaster nebulos (0.017), Myzus persicae (0.083). Pentatomid bugs (Nezara, Palomena, Halyomorpha, Rhaphigaster) are the usual cluster to check, but the ranking above is whatever this test set actually produced.

### Did certainty move, and was the sure tick always the last one?

Fraction of test clips whose most certain tick is the last tick (tick 5): seeds 0.985816 and 0.780142, mean 0.882979.

- Seed 0. Mean certainty by tick: 0.002, 0.009, 0.053, 0.127, 0.242, 0.353. Histogram of chosen ticks [0, 0, 0, 0, 2, 139]. Mean attention entropy by tick (nats, over the 24 pooled spectrogram tokens): 3.176, 3.176, 3.176, 3.176, 3.176, 3.176.
- Seed 1. Mean certainty by tick: 0.002, 0.103, 0.092, 0.124, 0.218, 0.277. Histogram of chosen ticks [0, 1, 0, 0, 30, 110]. Mean attention entropy by tick (nats, over the 24 pooled spectrogram tokens): 3.172, 3.172, 3.172, 3.172, 3.171, 3.171.

Mean certainty does rise across the six ticks (near 0 at tick 0, about 0.35 and 0.28 at the last tick). The chosen tick is still the last one for 139/141 clips in seed 0 and 110/141 in seed 1. **Adaptive compute did not emerge:** the model becomes more sure as it is forced to keep ticking, and it does not peak early. Attention entropy stays at about 3.17 nats on every tick. A uniform distribution over the 24 spectrogram tokens has entropy ln(24) = 3.178, so the attention weights barely leave uniform and do not move across ticks. The synchronization readout is doing the work, if anything is; the attention query is not retargeting the clip.

## Conclusion

On this frozen night hold-out the CTM-style model did not significantly beat the plain CNN. Mean test accuracy was 0.361702 versus 0.375887 (mean difference -0.014184); one bootstrap interval includes 0 and the other shows the CNN ahead. A linear model on mel-band mean and standard deviation scored 0.319149 accuracy and 0.265281 macro F1, below both nets but in the same uncertain range. Certainty increased over ticks, yet the most certain tick was almost always the last one, and attention stayed nearly uniform, so this small CTM-style net did not show adaptive compute.

## Artifacts

- `/workspace/insect-sound/PROJECT.md` — this report
- `/workspace/insect-sound/session_metrics.json` — every number above
- `/workspace/insect-sound/session_test_predictions.csv` — test predictions for both seeds
- `/workspace/insect-sound/session_manifest.json` — recording-id split
- `/workspace/insect-sound/session_models_seed0.pt`, `session_models_seed1.pt`

