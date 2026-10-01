# Execution notes

The managed filesystem sandbox hides GPU driver access: initial `nvidia-smi`
and `torch.cuda.is_available()` failed there. Outside the sandbox, CUDA works
on the NVIDIA GeForce RTX 3050 Laptop GPU (4 GB), driver 580.178.04.

The first diagnostic smoke test used Ultralytics 8.4.171 and PyTorch
2.11.0+cu130. Its AMP check failed and it automatically retried batch 16 at
batch 8 after CUDA OOM. It is preserved in `runs/baseline_smoke_diagnostic`
and `results/metrics/diagnostic_8.4.171_*`; it is not a final experiment.

Before full experiments, Ultralytics was pinned to 8.3.228, whose YOLO11n
AMP check passes on this device. The project still uses the exact same
COCO-pretrained `yolo11n.pt`. No model, dataset, or requested training
hyperparameter was changed to resolve the compatibility issue.

Both final 50-epoch runs completed with batch 16 and AMP enabled, without
reducing batch size. The earlier diagnostic OOM does not apply to the final
configuration.

Before reporting final results, test evaluation, benchmarking, and failure
analysis were corrected to explicitly use `rect=False` for fixed 640×640
padding. Default prediction padding can instead produce rectangular tensors.
The final CSV/JSON and failure figures reflect the explicit square inputs;
earlier provisional measurements are superseded. This protocol correction
was applied equally to both models. No weights, model selection, training
settings, confidence thresholds, or augmentation parameters were tuned using
the test set. Training-time validation retains the Ultralytics defaults.

Training time in JSON measures the complete training command, including
startup, AMP checking, plotting, checkpoint saving, and the library's final
validation. The `time` column in the per-epoch CSV separately records cumulative
epoch-loop time. Best-checkpoint revalidation uses batch 16, whereas
training-time validation uses the library's default doubled validation batch.
Rectangular batch padding and precision differences can change revalidation
metrics slightly; model selection still uses training-time validation fitness.

Smoke and diagnostic figures remain local and are ignored; their measured
metric records are retained. Full-run curves, augmentation examples, test
figures and failure visualizations are tracked.
