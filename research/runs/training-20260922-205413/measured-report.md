# Shuaa measured training report

Source commit: `0e5d7475fe369d0079f4412d6910b29854fcf1a9`. Split hash: `043313273c135d80e4a298717c19253145ec1156eb99dc69441b78dcbd4139f5`.

Campaign status: **complete**. These are validation results; test was not evaluated. Time-limited runs may have stopped before convergence.

| Run | Epochs | Stop reason | Validation accuracy | Validation Macro-F1 |
|---|---:|---|---:|---:|
| convnextv2_in22k | 31 | time_budget_reached | 0.856048 | 0.743053 |
| convnextv2_in22k_lr3e5 | 19 | early_stopped | 0.835721 | 0.698820 |
| convnextv2_in22k_smoothing | 31 | time_budget_reached | 0.850716 | 0.729388 |
| convnextv2_in22k_effective_number | 31 | time_budget_reached | 0.851716 | 0.735672 |
| convnextv2_in22k_mixup | 31 | time_budget_reached | 0.856714 | 0.734397 |
| dinov2_small_frozen | 29 | early_stopped | 0.742752 | 0.548778 |
| convnextv2_in22k_mixup_seed43 | 31 | time_budget_reached | 0.857381 | 0.745995 |

Selected: `probability_mean` using convnextv2_in22k_mixup_seed43, convnextv2_in22k_mixup.

Full methodology and limitations: `research/methodology.md`. All per-epoch logs and resumable checkpoints are in `runs/`. ONNX parity report: `model.parity.json`.
