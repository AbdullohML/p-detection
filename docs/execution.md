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
