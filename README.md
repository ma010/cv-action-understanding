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

## Notebook workflow (major steps)
Each notebook follows the same end-to-end flow, with later versions optimizing parts of the pipeline:

1. **Data discovery & labeling**
   - Read the dataset folders/files and build a table of image paths + labels.
   - Map class names to numeric labels (one-hot in the baseline, integer labels in enhanced versions).
2. **Train/validation split**
   - Baseline: trains on all data with no validation split.
   - Enhanced versions: stratified train/validation split to measure generalization.
3. **Input pipeline**
   - Baseline: load all images into RAM with PIL, then convert to NumPy arrays.
   - Enhanced versions: stream with `tf.data` (`read_file` → `decode_jpeg` → `resize`) and use `map`/`prefetch` for throughput.
4. **Augmentation & preprocessing**
   - Baseline: no augmentation.
   - Enhanced versions: random flips/rotations/zoom/contrast plus EfficientNet preprocessing.
5. **Model definition**
   - All versions use EfficientNetB7 as the backbone.
   - Baseline head: `Flatten → Dense(512) → Dense(num_classes)`.
   - Enhanced head: `GlobalAveragePooling2D → Dropout → Dense(num_classes)` (lighter/faster).
6. **Training configuration**
   - Optimizer: Adam (all versions).
   - Loss: categorical cross-entropy (baseline) vs sparse categorical cross-entropy (enhanced).
   - Metrics: accuracy plus optional top‑3 accuracy in enhanced versions.
   - Callbacks: early stopping + checkpointing in enhanced versions.
7. **Acceleration options**
   - Enhanced versions add multi‑GPU (`MirroredStrategy`).
   - Version 4 adds mixed precision with float32 output logits.
8. **Evaluation & inference**
   - Report training/validation metrics.
   - Run sample predictions and map numeric outputs back to class names.

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
