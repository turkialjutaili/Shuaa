# Shuaa measured training report

Source commit: `80e8b79713ff66adf27d5281fea71825e4b1363b`. Split hash: `043313273c135d80e4a298717c19253145ec1156eb99dc69441b78dcbd4139f5`.

Campaign status: **complete**. These are validation results; test was not evaluated. Time-limited runs may have stopped before convergence.

| Run | Epochs | Stop reason | Validation accuracy | Validation Macro-F1 |
|---|---:|---|---:|---:|
| dinov2_small_last4 | 25 | completed | 0.830390 | 0.698027 |
| dinov2_small_last4_seed43 | 16 | early_stopped | 0.823392 | 0.690581 |

Selected: `single` using dinov2_small_last4.

Full methodology and limitations: `research/methodology.md`. All per-epoch logs and resumable checkpoints are in `runs/`. ONNX parity report: `model.parity.json`.
