"""Export probability-output ONNX and verify actual-image CPU parity before producing metadata."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import numpy as np
import torch
from torch import nn
from training.train import CLASSES, SolarDataset, atomic_json, load_split, make_model

PAPERS = {
    "tf_efficientnetv2_s.in1k": "https://proceedings.mlr.press/v139/tan21a.html",
    "convnextv2_tiny.fcmae_ft_in1k": "https://arxiv.org/abs/2301.00808",
    "convnextv2_tiny.fcmae_ft_in22k_in1k": "https://arxiv.org/abs/2301.00808",
    "vit_small_patch14_dinov2.lvd142m": "https://arxiv.org/abs/2304.07193",
}

class ProbabilityModel(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)
    def forward(self, images):
        return torch.stack([model(images).softmax(1) for model in self.models]).mean(0)

def training_component(checkpoint, state):
    config = state["config"]
    epochs_completed = len(state["history"])
    result_path = Path(checkpoint).parent / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result["config"] != config or result["split_file_sha256"] != state["split_file_sha256"]:
            raise ValueError("Run result differs from checkpoint")
        if result["epochs_completed"] < epochs_completed:
            raise ValueError("Run result predates the best checkpoint")
        epochs_completed = result["epochs_completed"]
    weighting = config.get("class_weighting") or ("inverse_frequency" if config.get("weighted_loss") else "none")
    return dict(run_id=Path(checkpoint).parent.name, model_name=config["model"], seed=config["seed"],
                epochs_completed=epochs_completed, configured_epochs=config["epochs"], optimizer="AdamW",
                initial_learning_rate=config["lr"], weighted_loss=weighting != "none")

def export(checkpoints, dataset, split_path, output, metadata=None):
    import onnx
    import onnxruntime as ort
    manifest = load_split(split_path)
    states = [torch.load(path, map_location="cpu", weights_only=False) for path in checkpoints]
    models = []
    sizes = set()
    for state in states:
        if state["classes"] != CLASSES or state["split_file_sha256"] != manifest["file_sha256"]:
            raise ValueError("Checkpoint classes or split differ from benchmark")
        model = make_model(state["config"], pretrained=False)
        model.load_state_dict(state["model"])
        models.append(model.eval())
        sizes.add(state["config"]["image_size"])
    if len(sizes) != 1:
        raise ValueError("Ensemble requires matching input sizes")
    size = sizes.pop()
    model = ProbabilityModel(models).eval()
    valid = SolarDataset(dataset, manifest, "validation", size)
    count = min(8, len(valid))
    if count < 2:
        raise ValueError("At least two validation samples are required for dynamic-batch parity")
    images = torch.stack([valid[i][0] for i in range(count)])
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, images[:1], str(output), input_names=["images"], output_names=["probabilities"], dynamic_axes={"images": {0: "batch"}, "probabilities": {0: "batch"}}, opset_version=17, dynamo=False)
    onnx.checker.check_model(str(output))
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    max_error = 0.
    for batch in (images[:1], images):
        with torch.inference_mode():
            reference = model(batch).numpy()
        actual = session.run(["probabilities"], {"images": batch.numpy()})[0]
        np.testing.assert_allclose(actual, reference, rtol=1e-3, atol=1e-5)
        np.testing.assert_allclose(actual.sum(1), np.ones(len(batch)), atol=1e-5)
        if actual.shape != (len(batch), len(CLASSES)) or not np.isfinite(actual).all() or (actual < 0).any():
            raise ValueError("Invalid ONNX probability output")
        if not np.array_equal(actual.argmax(1), reference.argmax(1)):
            raise ValueError("ONNX changes predicted labels on parity samples")
        max_error = max(max_error, float(np.abs(actual - reference).max()))
    sha256 = hashlib.sha256(output.read_bytes()).hexdigest()
    atomic_json(output.with_suffix(".parity.json"), dict(status="passed", checkpoint_sha256=sha256, max_absolute_error=max_error, tested_batch_sizes=[1, count], split_hash=manifest["split_hash"], test_evaluated=False))
    if metadata is not None:
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]{2,79}", metadata["id"]):
            raise ValueError("Invalid submission id")
        if not re.fullmatch(r"[0-9a-f]{40}", metadata["source_commit"]):
            raise ValueError("source_commit must be a full Git commit hash")
        if not metadata["checkpoint_url"].startswith("https://github.com/turkialjutaili/Shuaa/releases/download/"):
            raise ValueError("Use a Shuaa GitHub Release asset URL")
        components = [training_component(path, state) for path, state in zip(checkpoints, states)]
        training = dict(ensemble_method="probability_mean" if len(states) > 1 else "single_model", components=components)
        submission = dict(schema_version=1, **metadata, checkpoint_sha256=sha256, split_hash=manifest["split_hash"], preprocess=dict(input_size=size, color_mode="RGB", resize="bilinear", mean=[.485, .456, .406], std=[.229, .224, .225]), classes=CLASSES, training=training)
        from benchmark.evaluation import validate_submission
        validate_submission(submission, manifest["split_hash"])
        atomic_json(output.with_suffix(".submission.json"), submission)
    return sha256

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True, help="Repeat for an approved probability-mean ensemble")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="benchmark/split.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--id")
    parser.add_argument("--participant", choices=["turki", "muhannad", "mazen"], default="turki")
    parser.add_argument("--model-name")
    parser.add_argument("--paper-url")
    parser.add_argument("--code-url", default="https://github.com/turkialjutaili/Shuaa")
    parser.add_argument("--source-commit")
    parser.add_argument("--checkpoint-url")
    args = parser.parse_args()
    supplied = [args.id, args.model_name, args.paper_url, args.source_commit, args.checkpoint_url]
    if any(supplied) and not all(supplied):
        parser.error("Metadata requires --id, --model-name, --paper-url, --source-commit and --checkpoint-url together")
    metadata = dict(id=args.id, participant=args.participant, model_name=args.model_name, paper_url=args.paper_url, code_url=args.code_url, source_commit=args.source_commit, checkpoint_url=args.checkpoint_url) if all(supplied) else None
    print(export(args.checkpoint, args.dataset, args.split, args.output, metadata))

if __name__ == "__main__":
    main()
