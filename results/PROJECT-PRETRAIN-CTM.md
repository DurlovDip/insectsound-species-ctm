# InsectSound1000 quarter subset, CTM-style head on frozen mn10_as embeddings

## Problem

Same frozen recording-id split and the same frozen EfficientAT `mn10_as` embeddings as PROJECT-PRETRAIN.md. The backbone is not trained. Only a small CTM-style head is trained. This is the same simplified tick model used on log-mel spectrograms in PROJECT-QUARTER.md (private neuron history, synchronization readout, loss at the lowest-loss tick and the most certain tick, test decision at the most certain tick). It is not the full Darlow et al. model.

## Data

Embeddings reused from `/workspace/insect-sound/embeddings_mn10_as.f32`: 41495 clips, 960 dimensions, extraction not repeated. Split from `manifest_quarter.json`: train 28426, val 6226, test 6843. Recording-id overlap: 0. `StandardScaler` fit on train embeddings only. No subsampling. Full test n=6843.

## Head

Each 960-d vector is split into contiguous tokens (no learned audio front-end). Each tick: synchronization of a neuron subset builds an attention query over those tokens; a shared GLU synapse updates a private pre-activation history; a per-neuron SuperLinear+GLU writes the next state; a second synchronization vector is the class logit. Certainty is one minus the normalized softmax entropy. Loss is the mean of cross-entropy at the lowest-loss tick and cross-entropy at the highest-certainty tick. Inverse-frequency class weights. Adam, learning rate 0.001, weight decay 1e-4. Early stop on validation macro F1 of the most-certain-tick prediction, patience 5, at most 30 epochs. CPU.

Architecture chosen on seed 0 validation macro F1 only, before any test score. Ties would go to fewer parameters, then fewer ticks.

| config | tokens | D | d_in | memory | ticks | sync | params | seed 0 val macro F1 | seed 0 val accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| small | 12 | 32 | 32 | 4 | 4 | 8 | 13221 | 0.383041 | 0.468358 |
| mid ** | 16 | 48 | 40 | 8 | 6 | 10 | 21959 | 0.403781 | 0.525538 |
| wide | 16 | 64 | 48 | 8 | 6 | 12 | 33553 | 0.379476 | 0.468358 |

Selected: **mid**. Head parameters: **21959**. Tokens 16 (width 60), D=48, d_in=40, memory 8, ticks 6, sync subset 10.

## Frozen test

Scored only for the selected head, seeds 0, 1, and 2. Seed 0 is the selection run, not a retrain. Not a subsample.

| model | test accuracy | correct | test macro F1 |
|---|---:|---:|---:|
| Majority | 0.031857 | 218/6843 | 0.005146 |
| Linear log-mel | 0.419114 | 2868/6843 | 0.405081 |
| Plain CNN mean | 0.440645 | — | 0.443585 |
| CTM-style on spectrograms, mean | 0.379609 | — | 0.339988 |
| Frozen mn10_as + MLP 960→64→12 | 0.475376 | 3253/6843 | 0.446051 |
| Frozen mn10_as + CTM-style head, mean of 3 seeds | 0.380389 | — | 0.352799 |

- Seed 0: 2749/6843 = 0.401724, macro F1 0.373414. Val macro F1 0.403781 at epoch 14 (ran 19, 51.1s). Most-certain tick is the last tick on 0.457986 of test clips. Tick counts [0, 682, 723, 1262, 1042, 3134]. Beats MLP accuracy: **False**.
- Seed 1: 2473/6843 = 0.361391, macro F1 0.329058. Val macro F1 0.373612 at epoch 4 (ran 9, 23.4s). Most-certain tick is the last tick on 0.432705 of test clips. Tick counts [0, 569, 1085, 1414, 814, 2961]. Beats MLP accuracy: **False**.
- Seed 2: 2587/6843 = 0.378051, macro F1 0.355926. Val macro F1 0.375226 at epoch 5 (ran 10, 28.5s). Most-certain tick is the last tick on 0.391495 of test clips. Tick counts [127, 764, 1275, 839, 1159, 2679]. Beats MLP accuracy: **False**.

The CTM-style head does not beat the frozen MLP. Mean accuracy 0.380389 versus MLP 0.475376 (3253/6843).

### Per-species recall, mean over the 3 seeds

| species | recall mean |
|---|---:|
| Aphidoletes aphidimyza | 0.770802 |
| Bombus terrestris | 0.478994 |
| Bradysia difformis | 0.567141 |
| Coccinella septempunctata | 0.633028 |
| Episyrphus balteatus | 0.826525 |
| Halyomorpha halys | 0.270936 |
| Myzus persicae | 0.014239 |
| Nezara viridula | 0.115172 |
| Palomena prasina | 0.339372 |
| Rhaphigaster nebulos | 0.146160 |
| Trialeurodes vaporariorum | 0.197248 |
| Tuta absoluta | 0.237624 |

Numbers were written by `pretrain_ctm_head.py` from `metrics_pretrain_ctm.json`. Baseline rows were read from `metrics_pretrain.json` and `metrics_quarter.json`. Wall time 152.0s.
