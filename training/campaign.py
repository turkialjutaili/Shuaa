"""Run the legacy or bounded next validation campaign without touching test data."""
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
from training.train import atomic_json, config_from, score, train

def rank(result):
    metrics = result["best_validation"]
    return metrics["accuracy"], metrics["macro_f1"]

def choose_ensemble(first, second):
    a, b = np.load(first), np.load(second)
    if not np.array_equal(a["ids"], b["ids"]) or not np.array_equal(a["labels"], b["labels"]):
        raise ValueError("Cannot ensemble predictions for different validation samples")
    probabilities = (a["probabilities"] + b["probabilities"]) / 2
    return score(a["labels"], probabilities.argmax(1))

CAMPAIGNS = {
    "legacy": ["efficientnetv2_ce", "convnextv2_ce", "efficientnetv2_weighted", "convnextv2_weighted"],
    # Ordered one-change trials. DINO is kept within the same bounded allocation so it cannot be
    # starved by an open-ended ConvNeXt run.
    "improvements": [
        "convnextv2_in22k", "convnextv2_in22k_lr3e5", "convnextv2_in22k_smoothing",
        "convnextv2_in22k_effective_number", "convnextv2_in22k_mixup", "dinov2_small_frozen",
    ],
    "dinov2_finetune": ["dinov2_small_last4"],
}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="benchmark/split.json")
    parser.add_argument("--output", default="runs")
    parser.add_argument("--budget-hours", type=float, default=11)
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--device", default=None)
    parser.add_argument("--campaign", choices=sorted(CAMPAIGNS), default=os.environ.get("SHUAA_CAMPAIGN", "legacy"))
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.budget_hours <= 0:
        raise ValueError("budget-hours must be positive")
    start = time.monotonic()
    configs = CAMPAIGNS[args.campaign]
    results = {}
    # A session may end at any epoch; --resume reuses each run's checkpoint.
    for name in configs:
        config = config_from(Path(args.config_dir) / f"{name}.json")
        run = output / name
        result_path = run / "result.json"
        existing = json.loads(result_path.read_text()) if result_path.exists() else None
        if existing and existing["status"] in {"completed", "early_stopped", "time_budget_reached"}:
            result = existing
        else:
            remaining = args.budget_hours * 3600 - (time.monotonic() - start)
            if remaining < config["time_budget_hours"] * 3600 + 300:
                print(json.dumps(dict(run=name, status="skipped_insufficient_campaign_budget", remaining_seconds=remaining)), flush=True)
                break
            print(json.dumps(dict(run=name, status="starting", allocation_hours=config["time_budget_hours"])), flush=True)
            result = train(config, args.dataset, args.split, run, resume=True, device_name=args.device)
            print(json.dumps(dict(run=name, status=result["status"], epochs_completed=result["epochs_completed"], best_validation=result["best_validation"])), flush=True)
        if result["best_validation"] is not None:
            results[name] = result
    if not results:
        atomic_json(output / "campaign.json", dict(status="no_completed_validation", test_evaluated=False))
        return
    winner = max(results, key=lambda name: rank(results[name]))
    repeat_name = winner + "_seed43"
    repeat = output / repeat_name
    config = results[winner]["config"] | {"seed": 43}
    existing = json.loads((repeat / "result.json").read_text()) if (repeat / "result.json").exists() else None
    if len(results) == len(configs):
        if existing:
            repeat_result = existing
        elif args.budget_hours * 3600 - (time.monotonic() - start) > config["time_budget_hours"] * 3600 + 300:
            repeat_result = train(config, args.dataset, args.split, repeat, resume=True, device_name=args.device)
        else:
            repeat_result = None
        if repeat_result and repeat_result["best_validation"] is not None:
            results[repeat_name] = repeat_result
    ordered = sorted(results, key=lambda name: rank(results[name]), reverse=True)
    selection = dict(kind="single", runs=[ordered[0]], validation=results[ordered[0]]["best_validation"])
    if len(ordered) >= 2:
        ensemble = choose_ensemble(output / ordered[0] / "validation_predictions.npz", output / ordered[1] / "validation_predictions.npz")
        if (ensemble["accuracy"], ensemble["macro_f1"]) > rank(results[ordered[0]]):
            selection = dict(kind="probability_mean", runs=ordered[:2], validation=ensemble)
    core_complete = all(name in results for name in configs)
    atomic_json(output / "campaign.json", dict(campaign=args.campaign, status="complete" if core_complete else "partial",
                core_runs=configs, optional_repeat=repeat_name if repeat_name in results else None,
                runs=results, selected=selection, test_evaluated=False))
    print(json.dumps(selection, indent=2))

if __name__ == "__main__":
    main()
