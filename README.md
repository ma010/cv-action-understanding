# CV Action Understanding

Lightweight notebook lab for human action recognition (HAR) experiments.

## Overview
- Focus: computer vision models for human action recognition.
- HAR = Human Action Recognition.
- Notebook filenames encode date, runtime, and GPU used (e.g., `...-21min-GPU-2T4s`).

## What’s here
- Jupyter notebooks tracking incremental changes from an open-source baseline.
- `NOTEBOOK_EXPERIMENTS_SUMMARY.md` for a side-by-side experiment summary.

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
