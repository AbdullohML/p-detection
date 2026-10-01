# Baseline protocol

COCO-pretrained Ultralytics YOLO11n (`yolo11n.pt`), detection, one class `0: person`.
Train only on train images. Validation selects `weights/best.pt` using Ultralytics
fitness (mAP@0.5:0.95). Test evaluation happens after the complete run.

Fixed settings: 640 pixels, 50 epochs, batch 16, AdamW, lr0 0.001,
weight decay 0.0005, seed 42, pretrained enabled, AMP requested.
Other model settings retain the pinned Ultralytics defaults. Workers=2 and
CPU threads=6 control loading/execution, identically for both conditions.
On CUDA, AMP is enabled if Ultralytics' AMP compatibility check passes.
Batch reduction is permitted only after CUDA OOM and must be recorded.
The second condition must use the same successful batch size.

A one-epoch `--smoke` run has a separate directory and is never a final model.
The scripts reject incomplete training for test evaluation. Do not tune settings
from test results. Precision/recall in the main table are Ultralytics' values at
its maximum smoothed F1 confidence; failure analysis instead uses fixed confidence
0.25. These operating points are different and should not be conflated.

Reference: https://docs.ultralytics.com/modes/train/
