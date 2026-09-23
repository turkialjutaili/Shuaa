import json
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn
from benchmark.dataset import manifest_hash
from training.train import CLASSES, DEFAULTS, SolarDataset, class_weights, config_from, effective_number_weights, load_split, make_model, preprocess, set_trainable, train

class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
        self.head = nn.Linear(4, 12)
    def get_classifier(self):
        return self.head
    def forward(self, x):
        return self.head(self.features(x).flatten(1))

def fixture(tmp_path):
    samples = []
    for partition in ("train", "validation", "test"):
        for label in range(12):
            name = f"{partition}-{label}.png"
            pixels = np.full((24, 40), 10 + label * 15, np.uint8)
            Image.fromarray(pixels).save(tmp_path / name)
            samples.append(dict(id=name, path=name, label=label, group=name, split=partition))
    manifest = dict(schema_version=1, classes=CLASSES, samples=samples, seed=42, dataset={})
    manifest["split_hash"] = manifest_hash(manifest)
    path = tmp_path / "split.json"
    path.write_text(json.dumps(manifest))
    return path

def test_preprocess_matches_benchmark(tmp_path):
    from benchmark.evaluation import preprocess_image
    image = Image.fromarray(np.arange(24 * 40, dtype=np.uint8).reshape(24, 40))
    path = tmp_path / "image.png"
    image.save(path)
    np.testing.assert_allclose(preprocess(image, 224).numpy(), preprocess_image(path, 224), atol=1e-7)

def test_preprocess_triplicate_and_full_resize(tmp_path):
    image = Image.fromarray(np.arange(24 * 40, dtype=np.uint8).reshape(24, 40))
    result = preprocess(image, 32)
    assert result.shape == (3, 32, 32)
    assert result.dtype == torch.float32
    reconstructed = result.numpy() * np.array([.229,.224,.225])[:,None,None] + np.array([.485,.456,.406])[:,None,None]
    np.testing.assert_allclose(reconstructed[0], reconstructed[1], atol=1e-7)
    np.testing.assert_allclose(reconstructed[1], reconstructed[2], atol=1e-7)

def test_train_only_weights_and_forbid_test(tmp_path):
    path = fixture(tmp_path)
    manifest = load_split(path)
    with pytest.raises(ValueError, match="cannot load test"):
        SolarDataset(tmp_path, manifest, "test")
    weights = class_weights([s for s in manifest["samples"] if s["split"] == "train"])
    assert torch.equal(weights, torch.ones(12))
    with pytest.raises(ValueError, match="Every class"):
        class_weights([{"label":0}])

def test_hash_tamper_rejected(tmp_path):
    path = fixture(tmp_path)
    content = json.loads(path.read_text())
    content["samples"][0]["label"] = 3
    path.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="hash mismatch"):
        load_split(path)

def test_unknown_config_rejected(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"epohs":3}')
    with pytest.raises(ValueError, match="Unknown"):
        config_from(path)

def test_effective_number_weights_favor_rare_classes():
    samples = [{"label": label} for label in range(12) for _ in range(1 if label == 0 else 10)]
    weights = effective_number_weights(samples, beta=.9)
    assert weights.shape == (12,)
    assert weights[0] > weights[1]
    assert weights.mean().item() == pytest.approx(1.)

def test_next_models_create_at_224_without_download():
    convnext = make_model(DEFAULTS | {"model":"convnextv2_tiny.fcmae_ft_in22k_in1k", "pretrained":False})
    assert convnext.get_classifier().out_features == len(CLASSES)
    dino = make_model(DEFAULTS | {"model":"vit_small_patch14_dinov2.lvd142m", "model_kwargs":{"img_size":224}, "pretrained":False})
    assert dino.patch_embed.img_size == (224, 224)
    assert dino.get_classifier().out_features == len(CLASSES)
    set_trainable(dino, classifier_only=True)
    assert all(parameter.requires_grad for parameter in dino.get_classifier().parameters())
    assert not dino.patch_embed.proj.weight.requires_grad
    set_trainable(dino, classifier_only=False, last_blocks=4)
    assert all(parameter.requires_grad for parameter in dino.blocks[-1].parameters())
    assert not any(parameter.requires_grad for parameter in dino.blocks[0].parameters())
    assert all(parameter.requires_grad for parameter in dino.get_classifier().parameters())
    assert not dino.patch_embed.proj.weight.requires_grad

def test_cpu_training_checkpoint_and_resume(tmp_path):
    torch.set_num_threads(1)
    path = fixture(tmp_path)
    config = DEFAULTS | dict(epochs=2, warmup_epochs=1, batch_size=7, accumulation=2, image_size=32, pretrained=False, workers=0)
    factory = lambda c: TinyNet()
    output = tmp_path / "run"
    result = train(config, tmp_path, path, output, device_name="cpu", model_factory=factory)
    assert result["epochs_completed"] == 2
    assert result["test_evaluated"] is False
    state = torch.load(output / "last.pt", weights_only=False)
    assert set(state["rng"]) == {"python", "numpy", "torch", "loader", "cuda"}
    assert all(name in state for name in ("optimizer", "scheduler", "scaler", "history"))
    resumed = train(config, tmp_path, path, output, resume=True, device_name="cpu", model_factory=factory)
    assert resumed["best_validation"] == result["best_validation"]
    with pytest.raises(ValueError, match="differs"):
        train(config | {"seed":43}, tmp_path, path, output, resume=True, device_name="cpu", model_factory=factory)

def test_onnx_dynamic_probability_parity(tmp_path, monkeypatch):
    import training.export as exporter
    torch.set_num_threads(1)
    path = fixture(tmp_path)
    config = DEFAULTS | dict(epochs=2, warmup_epochs=1, batch_size=12, image_size=32, pretrained=False, workers=0)
    output = tmp_path / "run"
    train(config, tmp_path, path, output, device_name="cpu", model_factory=lambda c: TinyNet())
    monkeypatch.setattr(exporter, "make_model", lambda config, pretrained=False: TinyNet())
    metadata = dict(id="test-training-export", participant="turki", model_name="TinyNet",
                    paper_url="https://example.com/paper", code_url="https://github.com/turkialjutaili/Shuaa",
                    source_commit="a" * 40,
                    checkpoint_url="https://github.com/turkialjutaili/Shuaa/releases/download/test/model.onnx")
    sha = exporter.export([output / "best.pt"], tmp_path, path, tmp_path / "model.onnx", metadata)
    assert len(sha) == 64
    parity = json.loads((tmp_path / "model.parity.json").read_text())
    assert parity["status"] == "passed"
    assert parity["tested_batch_sizes"] == [1, 8]
    submission = json.loads((tmp_path / "model.submission.json").read_text())
    component = submission["training"]["components"][0]
    assert component["epochs_completed"] == 2
    assert component["configured_epochs"] == 2
    assert component["optimizer"] == "AdamW"

def test_dinov2_real_graph_onnx_smoke(tmp_path):
    import onnx
    import onnxruntime as ort
    from training.export import ProbabilityModel
    torch.set_num_threads(1)
    config = DEFAULTS | {"model":"vit_small_patch14_dinov2.lvd142m", "model_kwargs":{"img_size":224}, "pretrained":False}
    model = ProbabilityModel([make_model(config).eval()]).eval()
    example = torch.zeros(1, 3, 224, 224)
    target = tmp_path / "dino.onnx"
    torch.onnx.export(model, example, str(target), input_names=["images"], output_names=["probabilities"],
                      dynamic_axes={"images":{0:"batch"}, "probabilities":{0:"batch"}}, opset_version=17, dynamo=False)
    onnx.checker.check_model(str(target))
    session = ort.InferenceSession(str(target), providers=["CPUExecutionProvider"])
    for batch_size in (1, 2):
        batch = example.repeat(batch_size, 1, 1, 1)
        actual = session.run(["probabilities"], {"images":batch.numpy()})[0]
        assert actual.shape == (batch_size, len(CLASSES))
        np.testing.assert_allclose(actual.sum(1), np.ones(batch_size), atol=1e-5)

def test_interrupted_epoch_boundary_resume_matches_uninterrupted(tmp_path, monkeypatch):
    import training.train as trainer
    torch.set_num_threads(1)
    path = fixture(tmp_path)
    config = DEFAULTS | dict(epochs=3, warmup_epochs=1, batch_size=5, accumulation=2, image_size=32, pretrained=False, workers=0)
    factory = lambda c: TinyNet()
    train(config, tmp_path, path, tmp_path / "reference", device_name="cpu", model_factory=factory)
    original_save = trainer.atomic_json
    def simulate_interruption(path, value):
        original_save(path, value)
        if Path(path).name == "history.json":
            raise InterruptedError("simulated session end after checkpoint")
    monkeypatch.setattr(trainer, "atomic_json", simulate_interruption)
    with pytest.raises(InterruptedError):
        train(config, tmp_path, path, tmp_path / "interrupted", device_name="cpu", model_factory=factory)
    monkeypatch.setattr(trainer, "atomic_json", original_save)
    train(config, tmp_path, path, tmp_path / "interrupted", resume=True, device_name="cpu", model_factory=factory)
    a = torch.load(tmp_path / "reference/last.pt", weights_only=False)
    b = torch.load(tmp_path / "interrupted/last.pt", weights_only=False)
    assert a["history"] == b["history"]
    for key in a["model"]:
        assert torch.equal(a["model"][key], b["model"][key])
