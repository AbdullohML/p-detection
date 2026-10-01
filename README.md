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

Reports test mAP@0.5, mAP@0.5:0.95, precision, and recall. Do not tune from test
results. Both use the same test set. JSON records accompany `results/metrics/results.csv`.

## Benchmark inference

```bash
python scripts/benchmark.py --condition baseline
python scripts/benchmark.py --condition augmented
```

Batch 1, FP32, ten warmup predictions, five passes through identical preloaded
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

Full experiment results are pending. No unmeasured metrics are reported.
Evaluation and benchmarking automatically generate `results/metrics/comparison.md`
from `results.csv`. Training records contain elapsed time, best checkpoint,
final epoch metrics, best validation metrics, and hardware/software versions.

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
