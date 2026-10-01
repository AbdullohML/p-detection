# Failure analysis

Pending full training and test predictions. No visual categories or causes
have been assigned yet.

Run `python scripts/analyze_failures.py` after completing both experiments.
It uses confidence 0.25, NMS IoU 0.7, and matching IoU >= 0.5. Predictions
are processed in descending confidence. Each prediction may match at most
one unmatched ground-truth box, and each ground-truth box is used once.
Unmatched predictions are FP; unmatched ground truth is FN.

Figures: green = matched GT, yellow = FN GT, cyan = matched prediction,
red = FP prediction; prediction labels include confidence. Inspect at least
three saved cases visually before documenting likely causes. Model failures
and missing/incomplete dataset annotations must be distinguished.
