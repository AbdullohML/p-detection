# Pedestrian Detection with YOLO11n

## Project goal

Fine-tune COCO-pretrained YOLO11n for pedestrian detection on Penn-Fudan.
This standalone university project covers preparation, training, validation,
held-out testing, speed measurement, and visual error analysis.

## Research question

Does stronger data augmentation improve YOLO11n pedestrian detection performance on the Penn-Fudan dataset?

## Dataset

[Official Penn-Fudan dataset](https://www.cis.upenn.edu/~jshi/ped_html/)
contains pedestrian instance masks. Zero is background; each nonzero mask
value is one instance. Boxes enclose all foreground pixels with exclusive
upper edges (maximum foreground coordinate + 1), then become normalized YOLO
`0 x_center y_center width height` labels. The only class is person.

The downloaded official masks contain 423 instances, including newly labeled
small or occluded pedestrians noted in the archive readme.

Sorted filenames are shuffled using Python `random.Random(42)`. Train and
validation counts use floor(0.70*N) and floor(0.15*N); test gets the remainder.
Saved split manifests and zero-overlap assertions prevent filename leakage.
Nearby scenes may be correlated: this requested image split does not guarantee
scene independence. See `data/splits/statistics.json` for measured statistics.

## Baseline

YOLO11n, `yolo11n.pt`, COCO pretrained, 640×640 input. Exactly 50 epochs,
batch 16, AdamW, lr0 0.001, weight decay 0.0005, seed 42, AMP requested.
CUDA is selected automatically. See [baseline protocol](docs/baseline.md).

## Second experiment

YOLO11n with stronger color and geometric augmentation. Everything else stays
the same. Exact overrides are in [experiment](docs/experiment.md).

## Installation

Tested with Python 3.13. Run from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The execution environment used existing system packages through
`python -m venv --system-site-packages .venv`; the pinned requirements allow a
fresh environment. CUDA requires a working NVIDIA driver and compatible PyTorch.

## Download dataset

```bash
python scripts/download_dataset.py
```

## Prepare dataset

```bash
python scripts/prepare_dataset.py
```

Inspect `results/figures/dataset_examples/` before training. The conversion
asserts valid image/mask pairs and coordinates and verifies disjoint splits.
The checked-in YAML uses relative paths. Scripts resolve its root and create an
ignored runtime YAML so Ultralytics does not use a machine-specific datasets folder.

## Train baseline

```bash
python scripts/train_baseline.py --smoke
python scripts/train_baseline.py
```

The optional smoke run uses one epoch and cannot be evaluated as a final model.
Existing run directories cause an error to avoid overwriting experiments.
If batch 16 raises CUDA OOM, preserve its error record and move its run directory,
then retry with the largest feasible reduced batch via `--batch`. Use that same
batch for the augmented condition and document the deviation.

## Train second experiment

```bash
python scripts/train_augmented.py
```

## Evaluate

After each full model is selected using validation:

```bash
python scripts/evaluate.py --condition baseline
python scripts/evaluate.py --condition augmented
```

Test images use explicit 640×640 padding (`rect=False`). Reports test mAP@0.5, mAP@0.5:0.95, precision, and recall. Do not tune from test
results. Both use the same test set. JSON records accompany `results/metrics/results.csv`.

## Benchmark inference

```bash
python scripts/benchmark.py --condition baseline
python scripts/benchmark.py --condition augmented
```

Batch 1, FP32, explicit 640×640 padding (`rect=False`), ten warmup predictions, five passes through identical preloaded
test images. Wall-clock timing includes preprocessing, forward pass and NMS;
excludes model loading and disk reads. CUDA synchronizes before and after each
prediction. FPS = 1000 / mean latency in milliseconds. Hardware/software and
individual timings are saved. Run both on the same idle device.

## Failure analysis

```bash
python scripts/analyze_failures.py
```

Confidence 0.25 and one-to-one matching at IoU >= 0.5. See
[visual failure analysis](docs/failure_analysis.md). Visually review saved figures;
never automatically infer occlusion or other causes from matching counts.

## Results

| Model | Condition | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | Latency ms | FPS |
|---|---|---:|---:|---:|---:|---:|---:|
| YOLO11n | Baseline | 0.9586 | 0.8128 | 0.9676 | 0.9063 | 8.1828 | 122.2070 |
| YOLO11n | Strong augmentation | 0.9413 | 0.6066 | 0.8982 | 0.9394 | 8.0861 | 123.6683 |

Measured on the same 26 held-out test images; metrics are fractions.

Stronger augmentation changed mAP@0.5:0.95 by -0.2062. It did not improve overall AP on this split.

Dataset: 170 images; 119 train, 25 validation, 26 test. 423 pedestrian instances; mean 2.488/image, minimum 1, maximum 8.

Baseline: 50 epochs, batch 16, AMP enabled; training command 103.86 s (including startup, plotting, checkpoint saving and Ultralytics' final validation); 106.50 s including the additional best-checkpoint validation. Checkpoint: `runs/baseline/weights/best.pt` (local, ignored by Git).

Augmented: 50 epochs, batch 16, AMP enabled; training command 105.47 s (including startup, plotting, checkpoint saving and Ultralytics' final validation); 108.02 s including the additional best-checkpoint validation. Checkpoint: `runs/augmented/weights/best.pt` (local, ignored by Git).

Hardware: NVIDIA GeForce RTX 3050 Laptop GPU; 11th Gen Intel(R) Core(TM) i5-11400H @ 2.70GHz. Python 3.13.9, PyTorch 2.11.0+cu130, CUDA runtime 13.0, Ultralytics 8.3.228.

The table is generated from `results/metrics/results.csv`. Training/validation records, per-epoch curves, test metrics, and all benchmark timings are saved in `results/`. Precision and recall use Ultralytics' maximum-F1 operating point. The separate FP/FN analysis uses confidence 0.25.

Regenerate the table and README results with `python scripts/report_results.py`.

## Repository structure

```text
configs/             relative dataset YAML
scripts/             download, conversion, training, evaluation, speed, matching
data/raw/           official archive and extracted dataset (ignored)
data/processed/     YOLO images and labels (ignored)
data/splits/        reproducible filename manifests and statistics
docs/               baseline, experiment, visually reviewed failure analysis
results/metrics/    measured CSV/JSON records and comparison
results/figures/    lightweight annotation, training, evaluation, failure figures
results/predictions/ test predictions and matching records
runs/               full training outputs and weights (ignored)
```

## Reproducibility

Seed 42; fixed split manifests; pinned direct dependencies. Each training and
benchmark record includes software versions and hardware. CUDA determinism is
requested, but different devices/library versions can still produce differences.
The best validation checkpoint, rather than the last epoch, is evaluated.
Checkpoints, data copies, virtual environment, and large runs are excluded from Git.

Correctness checks:

```bash
python -m unittest discover -s tests
python -m compileall -q scripts tests
```

See [execution notes](docs/execution.md) for GPU access and the Ultralytics
version compatibility diagnosis.
