# Shuaa improvement campaign

This campaign follows [the research recommendations](improvement-plan.md). Its outcomes are not known before execution. The reference is the independently evaluated first ensemble: validation accuracy 85.0716%, Macro-F1 73.7819%.

## Ordered comparisons

| Configuration | Change | Initial learning rate |
|---|---|---:|
| `convnextv2_in22k` | ConvNeXt V2 Tiny weights fine-tuned on ImageNet-22K then 1K | 0.0001 |
| `convnextv2_in22k_lr3e5` | Lower learning rate | 0.00003 |
| `convnextv2_in22k_smoothing` | Label smoothing 0.05 | 0.0001 |
| `convnextv2_in22k_effective_number` | Effective-number class weights from training counts | 0.0001 |
| `convnextv2_in22k_mixup` | Mixup alpha 0.1 | 0.0001 |
| `dinov2_small_frozen` | DINOv2 ViT-S/14 frozen encoder with a trainable 12-class head | 0.001 |

Each trial has a 1.25-hour allocation, up to 35 epochs, and early stopping. The strongest completed configuration may be repeated with seed 43; initial trials use seed 42. The overall campaign budget is 11 hours, leaving room around the 8.75-hour maximum trial allocation for overhead. Export and parity checking happen afterward within the notebook session limit. Actual completed epochs and stopping reasons are recorded; a time-limited trial is not evidence of full convergence.

All trials use AdamW and the existing train/validation split. Compare the best single model and equal-probability mean of the top two models. Final test remains untouched. The current published result stays available regardless of whether this campaign improves it. Larger Base models, DINO block fine-tuning, and test-time augmentation are deferred until these results identify whether their extra compute is justified.

## Execution and collection

The reviewed notebook source commit must be pinned. Set `SHUAA_CAMPAIGN=improvements` and `SHUAA_BUDGET_HOURS=11`; use a separate Kaggle notebook, `turki101/shuaa-improvements`, so the first campaign remains available.

After completion, collect into a separate directory:

```powershell
python -X utf8 scripts/collect_kaggle.py --kernel turki101/shuaa-improvements --output outputs/kaggle-improvements
```

Review the measured report and parity evidence before passing `--publish`. Independent CPU validation then produces the website result. Refer to the run record for observed launch status; this document describes the experimental design, not a completion claim.
