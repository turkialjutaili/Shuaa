# Shuaa: healthy versus defective on validation

The published Shuaa models predict one of 12 classes. For this secondary analysis, `No-Anomaly` means **healthy** and each of the other 11 classes means **defective**. A model's top-ranked 12-class prediction is collapsed to one of these two labels. This is a measured view of the existing classifier, not a separately trained binary model or a new final-test score.

The [shared validation set](../benchmark/split.json) contains 3,001 images: 1,500 healthy and 1,501 defective. Counts below come from the [independent CPU evaluation records](../results/predictions/) and can also be reconstructed from each 12-class confusion matrix in the [published leaderboard](../public/data/leaderboard.json).

| Model | Correct / total | Binary accuracy | Defect recall | Healthy recall | Healthy → defective | Defective → healthy |
|---|---:|---:|---:|---:|---:|---:|
| Leading ConvNeXt V2 Tiny ensemble (`turki-training-20260922-205413`) | 2,881 / 3,001 | **96.00%** | 93.47% | 98.53% | 22 | 98 |
| Earlier ConvNeXt V2 Tiny ensemble (`turki-training-20260918-024537`) | 2,869 / 3,001 | 95.60% | 93.20% | 98.00% | 30 | 102 |
| Fine-tuned DINOv2 ViT-S/14 (`turki-training-20260923-032000`) | 2,850 / 3,001 | 94.97% | 93.27% | 96.67% | 50 | 101 |

For the leader, the binary confusion matrix is:

| Actual / predicted | Healthy | Defective |
|---|---:|---:|
| Healthy | 1,478 | 22 |
| Defective | 98 | 1,403 |

Accuracy is `(1,478 + 1,403) / 3,001`. Defect recall is `1,403 / 1,501`; healthy recall is `1,478 / 1,500`. Defect precision is `1,403 / (1,403 + 22) = 98.46%`. The binary score exceeds 95% on this validation split, while 98 truly defective images were missed. The distinction matters if missing a defect is costly.

These observations use the same validation split that guided model selection. They do not establish performance on a new solar farm or the reserved final-test split. The 12-class competition ranking remains based on its original accuracy and Macro-F1; this binary grouping is an additional diagnostic.
