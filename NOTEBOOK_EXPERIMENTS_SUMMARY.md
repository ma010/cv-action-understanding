# Human Action Recognition (HAR) Notebook Experiment Summary

## Overview
This summary compares the original notebook with subsequent variants. Each variant is described by filename (date, runtime, GPU) and the code changes relative to the original. Emphasis is placed on experimental changes and speed‑impacting modifications (data pipeline, model head, mixed precision, multi‑GPU).

Note: HAR = Human Action Recognition.

| Notebook | Runtime | GPU | Major changes |
| --- | --- | --- | --- |
| original-har-2024-36min-GPU-P100.ipynb | 36 min | 1× P100 | Baseline: in‑memory NumPy load, one‑hot labels, EfficientNetB7 + Flatten + Dense(512), no val split/augmentation |
| version1-har-20260117-36min-GPU-P100.ipynb | 36 min | 1× P100 | Exploratory cells only; training pipeline unchanged |
| version2-har-enhancement-20260111-28min-GPU-2T4s.ipynb | 28 min | 2× T4 | tf.data pipeline + stratified split, augmentation + preprocess, GAP head, sparse loss, callbacks, multi‑GPU |
| verion3-har-enhancement-20260111-22min-GPU-P100.ipynb | 22 min | 1× P100 | Same enhancements as v2 on single P100 |
| version4-har-enhancement-20260111-21min-GPU-2T4s.ipynb | 21 min | 2× T4 | v2 enhancements + mixed precision (float16 policy, float32 output) |

## Baseline (Original)
**File:** `original-har-2024-36min-GPU-P100.ipynb`  
**Runtime/GPU (from filename):** 36 min on 1× P100

**Key experiment setup**
- Loads all training images into memory via PIL, resizes to 160×160, converts to NumPy arrays.
- Labels are one‑hot encoded with `to_categorical`.
- Model: EfficientNetB7 backbone (`include_top=False`, `pooling="avg"`, ImageNet weights) + `Flatten` + Dense(512) + Dense(15 softmax).
- Optimizer/loss: Adam + `categorical_crossentropy`. 40 epochs.
- No explicit validation split, no callbacks, no augmentation.

**Speed/compute profile**
- Potentially heavy RAM usage (entire dataset in memory).
- Single‑GPU training, no mixed precision.
- Model head includes `Flatten` after pooled features (extra compute vs GAP‑only).

## Version 1 (Exploratory Add‑Ons)
**File:** `version1-har-20260117-36min-GPU-P100.ipynb`  
**Runtime/GPU (from filename):** 36 min on 1× P100

**Changes vs original**
- Adds exploratory/diagnostic cells only:
  - `len(train_data.label.unique())`
  - `print(len(y_train[0]))`
  - `train_data["label"].factorize()` and a `label_map` lookup
  - `tf.config.list_physical_devices('GPU')`
- Training/data pipeline and model remain identical to original.

**Speed impact**
- No meaningful training‑time changes; same model and input pipeline.

## Version 2 (Enhanced Pipeline + Multi‑GPU, T4s)
**File:** `version2-har-enhancement-20260111-28min-GPU-2T4s.ipynb`  
**Runtime/GPU (from filename):** 28 min on 2× T4

**Changes vs original**
- **Data pipeline redesign**:
  - Uses `train_test_split` (stratified) into train/val.
  - Builds `tf.data.Dataset` from filepaths and integer labels.
  - Uses `tf.io.read_file`, `tf.image.decode_jpeg`, `tf.image.resize`, `map(..., num_parallel_calls=AUTOTUNE)`, `batch`, `prefetch`.
  - Uses integer labels (`pd.factorize`) + `sparse_categorical_crossentropy`.
- **Augmentation + preprocessing**:
  - Adds `RandomFlip`, `RandomRotation`, `RandomZoom`, `RandomContrast`.
  - Adds `tf.keras.applications.efficientnet.preprocess_input`.
- **Model head + training**:
  - Replaces `Flatten + Dense(512)` with `GlobalAveragePooling2D + Dropout(0.2) + Dense(NUM_CLASSES)`.
  - Adds `SparseTopKCategoricalAccuracy(k=3)`.
  - Adds `EarlyStopping` and `ModelCheckpoint` (best weights).
- **Distributed training**:
  - Uses `tf.distribute.MirroredStrategy()` (multi‑GPU).
- **Inference cleanup**:
  - Uses `class_names` to show label names; prints probability and label.
  - Adds more test images.

**Speed‑impacting changes**
- Streaming input pipeline (no full in‑memory load) + `prefetch/parallel map`.
- Multi‑GPU MirroredStrategy (2× T4).
- Smaller head (GAP + Dropout) vs Flatten + Dense(512).
- Removes one‑hot labels (sparse loss) for reduced memory/compute overhead.

## Version 3 (Same Enhancements, P100)
**File:** `verion3-har-enhancement-20260111-22min-GPU-P100.ipynb`  
**Runtime/GPU (from filename):** 22 min on 1× P100

**Changes vs original**
- Same code changes as Version 2 (tf.data pipeline, stratified split, augmentation, preprocess_input, GAP head, callbacks, sparse labels, top‑3 metric, MirroredStrategy, inference label mapping).

**Speed‑impacting changes**
- Same as Version 2, but runs on P100 (single GPU).

## Version 4 (Enhanced + Mixed Precision, T4s)
**File:** `version4-har-enhancement-20260111-21min-GPU-2T4s.ipynb`  
**Runtime/GPU (from filename):** 21 min on 2× T4

**Changes vs Version 2/3**
- **Mixed precision enabled**:
  - `mixed_precision.set_global_policy('mixed_float16')`.
  - Output layer explicitly `dtype="float32"` to keep stable logits/softmax.
- Otherwise matches Version 2/3 enhancements.

**Speed‑impacting changes**
- Mixed precision likely reduces memory bandwidth and speeds up tensor ops on compatible GPUs.
- 2× T4 with MirroredStrategy + mixed precision.

## Cross‑Notebook Summary of Experimental Changes
- **Baseline → Enhanced (v2/v3/v4):**
  - Switched from in‑memory NumPy arrays to `tf.data` streaming with parallelism and prefetch.
  - Added data augmentation + EfficientNet preprocess.
  - Added validation split + early stopping + checkpointing.
  - Changed labels to integer + sparse loss.
  - Simplified head (GAP + Dropout) vs Flatten + Dense(512).
  - Added Top‑3 accuracy metric.
  - Added multi‑GPU distribution strategy.
- **Enhanced → Mixed Precision (v4):**
  - Global mixed‑precision policy with float32 output head.

## Notes on Runtime Differences (from filenames)
- P100 single‑GPU runs: 36 min (original, v1) vs 22 min (v3 with enhanced pipeline).
- T4 multi‑GPU runs: 28 min (v2) vs 21 min (v4 with mixed precision).
- Differences likely driven by the pipeline/architecture changes, GPU count/type, and mixed precision.
