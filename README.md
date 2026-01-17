# CV Action Understanding

Lightweight notebook lab for human action recognition (HAR) experiments.

## Overview
- Focus: computer vision models for human action recognition.
- HAR = Human Action Recognition.
- Notebook filenames encode date, runtime, and GPU used (e.g., `...-21min-GPU-2T4s`).

## Highlights
- Summary table: see `NOTEBOOK_EXPERIMENTS_SUMMARY.md#overview`.

| Notebook | Runtime | GPU | Major changes | Runtime variation (why) |
| --- | --- | --- | --- | --- |
| original-har-2024-36min-GPU-P100.ipynb | 36 min | 1× P100 | Baseline: in‑memory NumPy load, one‑hot labels, EfficientNetB7 + Flatten + Dense(512), no val split/augmentation | Slower: eager Python/PIL load + no `tf.data` pipeline; larger head; single GPU |
| version1-har-20260117-36min-GPU-P100.ipynb | 36 min | 1× P100 | Exploratory cells only; training pipeline unchanged | Same as baseline: only adds diagnostics |
| version2-har-enhancement-20260111-28min-GPU-2T4s.ipynb | 28 min | 2× T4 | `tf.data` pipeline + stratified split, augmentation + preprocess, GAP head, sparse loss, callbacks, multi‑GPU | Faster: `tf.data` + smaller head + 2× GPU; augmentation adds work but input pipeline parallelism helps |
| verion3-har-enhancement-20260111-22min-GPU-P100.ipynb | 22 min | 1× P100 | Same enhancements as v2 on single P100 | Faster than baseline: `tf.data` + smaller head; P100 strong single‑GPU throughput |
| version4-har-enhancement-20260111-21min-GPU-2T4s.ipynb | 21 min | 2× T4 | v2 enhancements + mixed precision (float16 policy, float32 output) | Fastest: mixed precision + 2× GPU + `tf.data` pipeline |

## What’s here
- Jupyter notebooks tracking incremental changes from an open-source baseline.
- `NOTEBOOK_EXPERIMENTS_SUMMARY.md` for a side-by-side experiment summary.

## Attribution
The original notebook is sourced from Kaggle: https://www.kaggle.com/code/kirollosashraf/human-action-recognition-har

## Data expectations
The notebooks currently read data from the Kaggle-style path:

```
../input/human-action-recognition-har-dataset/Human Action Recognition/
```

If you run locally, update paths to your dataset location.

## How to use
- Open a notebook and run top-to-bottom.
- Compare versions via the summary file.
- Use filenames to track runtime/GPU differences.

## Notes
- Versions introduce changes such as `tf.data` pipelines, augmentation, mixed precision, and multi-GPU training.
- Keep future notebooks aligned with the naming convention for clarity.
