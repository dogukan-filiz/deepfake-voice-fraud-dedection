# full_run_002 — Local Evaluation Report
**Date:** 2026-06-04
**Model:** full_run_002 (base + head override)
**Base:** models/ssl_aasist/weights.pth
**Head:** training_runs/full_run_002/best_head.pth
**Threshold:** 0.49 | **CALL_CHANNEL_MODE:** false

## Results
| Metric | Value |
|--------|-------|
| Accuracy | 1.0 (100%) |
| Macro F1 | 1.0 |
| AUC-ROC | 1.0 |
| Real Recall | 1.0 (50/50) |
| Fake Recall | 1.0 (50/50) |
| False Positives | 0 |
| False Negatives | 0 |

## Confusion Matrix
|  | Pred REAL | Pred FAKE |
|--|-----------|-----------|
| True REAL | 50 (TN) | 0 (FP) |
| True FAKE | 0 (FN) | 50 (TP) |

## Score Distribution
| Class | p_real mean | p_real min | p_real max |
|-------|------------|------------|------------|
| Real  | 0.9576 | 0.8125 | 0.9961 |
| Fake  | 0.0833 | 0.0119 | 0.4791 |

## Verdict
Perfect separation on digital domain. All 50 real files scored > 0.81, all 50 fake files scored < 0.48.
Model is production-ready for digital audio deepfake detection.