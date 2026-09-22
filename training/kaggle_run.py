"""Kaggle entry point: verified dataset, bounded campaign, ONNX export, provenance."""
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone

def main():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Enable a Kaggle GPU accelerator before starting the campaign")
    from benchmark.dataset import prepare
    from training.export import PAPERS, export
    from training.train import atomic_json
    repo = Path(__file__).resolve().parents[1]
    os.chdir(repo)
    output = Path(os.environ.get("SHUAA_OUTPUT", "/kaggle/working/shuaa-output"))
    output.mkdir(parents=True, exist_ok=True)
    runs = output / "runs"
    previous = os.environ.get("SHUAA_PREVIOUS_OUTPUT")
    if previous:
        prior = Path(previous) / "runs"
        if not prior.is_dir():
            raise ValueError("SHUAA_PREVIOUS_OUTPUT must contain runs from a previous kernel")
        shutil.copytree(prior, runs, dirs_exist_ok=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    manifest = prepare(repo)
    started = datetime.now(timezone.utc).isoformat()
    campaign_name = os.environ.get("SHUAA_CAMPAIGN", "legacy")
    subprocess.run([sys.executable, "-m", "training.campaign", "--dataset", str(repo / "data/infrared"), "--split", str(repo / "benchmark/split.json"), "--output", str(runs), "--budget-hours", os.environ.get("SHUAA_BUDGET_HOURS", "11"), "--campaign", campaign_name], check=True)
    campaign = json.loads((runs / "campaign.json").read_text())
    if "selected" not in campaign:
        raise RuntimeError("No trained model available for export")
    selected = campaign["selected"]
    run_names = selected["runs"]
    model_names = [campaign["runs"][name]["config"]["model"] for name in run_names]
    release = os.environ.get("SHUAA_RELEASE_TAG", "training-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"))
    metadata = dict(id=f"turki-{release}", participant="turki", model_name=" + ".join(model_names), paper_url=PAPERS[model_names[0]], code_url=f"https://github.com/turkialjutaili/Shuaa/tree/{commit}", source_commit=commit, checkpoint_url=f"https://github.com/turkialjutaili/Shuaa/releases/download/{release}/model.onnx")
    # Release publication happens separately after this validation-parity gate passes.
    export([runs / name / "best.pt" for name in run_names], repo / "data/infrared", repo / "benchmark/split.json", output / "model.onnx", metadata)
    provenance = dict(source_commit=commit, dataset=manifest["dataset"], split_hash=manifest["split_hash"], started_at=started, finished_at=datetime.now(timezone.utc).isoformat(), python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0), release_tag=release, campaign=campaign_name, campaign_status=campaign["status"], selected=selected, papers=[PAPERS[name] for name in model_names], test_evaluated=False)
    atomic_json(output / "provenance.json", provenance)
    (output / "pip-freeze.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True), encoding="utf-8")
    table = "\n".join(f"| {name} | {r['epochs_completed']} | {r['status']} | {r['best_validation']['accuracy']:.6f} | {r['best_validation']['macro_f1']:.6f} |" for name, r in campaign["runs"].items())
    report = f"# Shuaa measured training report\n\nSource commit: `{commit}`. Split hash: `{manifest['split_hash']}`.\n\nCampaign status: **{campaign['status']}**. These are validation results; test was not evaluated. Time-limited runs may have stopped before convergence.\n\n| Run | Epochs | Stop reason | Validation accuracy | Validation Macro-F1 |\n|---|---:|---|---:|---:|\n{table}\n\nSelected: `{selected['kind']}` using {', '.join(run_names)}.\n\nFull methodology and limitations: `research/methodology.md`. All per-epoch logs and resumable checkpoints are in `runs/`. ONNX parity report: `model.parity.json`.\n"
    (output / "measured-report.md").write_text(report, encoding="utf-8")
    print(report)

if __name__ == "__main__":
    main()
