# Model Training Cloud Report

**Environment:** RunPod — NVIDIA RTX A5000 (24 GB VRAM), CUDA 12.8, Ubuntu 24.04  
**Model:** SSL+AASIST (XLSR-300M backbone + AASIST head)  
**Strategy:** Frozen backbone, AASIST head fine-tuning only

---

## Baseline Evaluation

> Fill in after running `evaluate_current_model.py`

| Metric | Value |
|--------|-------|
| AUC-ROC | — |
| Best macro F1 | — |
| Real recall (at best threshold) | — |
| Fake recall (at best threshold) | — |
| Best threshold | — |
| Current AUTH_THRESHOLD | 0.01 |

---

## Smoke Training (Pipeline Validation Only)

> Fill in after smoke run. NOTE: Results on 20+20 train / 10+10 val are NOT meaningful metrics.

| Item | Value |
|------|-------|
| Train samples | 20 real + 20 fake |
| Val samples | 10 real + 10 fake |
| Epochs run | 3 |
| Loss function | Focal (γ=2, α=[0.4, 0.6]) |
| Frozen params | XLSR-300M (~300M) |
| Trainable params | AASIST head (~297K) |
| Peak GPU memory | — |
| Pipeline status | ✓ / ✗ |

---

## Observations / Notes

- [ ] XLSR-300M downloaded from HuggingFace (first run)
- [ ] weights.pth uploaded from local (1.2 GB)
- [ ] requirements-training.txt installed

---

## Next Steps (pending approval)

1. Acquire real training dataset (Common Voice TR + ASVspoof 2019)
2. Run full training with frozen backbone
3. Evaluate on held-out test set
4. If macro F1 improves and real recall >= baseline: promote best_head.pth to production
5. Update AUTH_THRESHOLD based on validation threshold sweep
