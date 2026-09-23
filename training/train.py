"""Train ImageNet-pretrained models against the immutable Shuaa split.

python -m training.train --config configs/efficientnetv2_ce.json --dataset data/infrared --split benchmark/split.json --output runs/efficientnetv2_ce
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

CLASSES = ["Cell", "Cell-Multi", "Cracking", "Hot-Spot", "Hot-Spot-Multi", "Shadowing", "Diode", "Diode-Multi", "Vegetation", "Soiling", "Offline-Module", "No-Anomaly"]
MEAN = np.array([.485, .456, .406], dtype=np.float32)[:, None, None]
STD = np.array([.229, .224, .225], dtype=np.float32)[:, None, None]
DEFAULTS = dict(model="tf_efficientnetv2_s.in1k", seed=42, epochs=35, patience=8, warmup_epochs=2,
                batch_size=32, accumulation=2, image_size=224, lr=.0001, weight_decay=.01,
                weighted_loss=False, class_weighting=None, effective_number_beta=.9999,
                label_smoothing=0., mixup_alpha=0., freeze_backbone=False,
                trainable_last_blocks=0, head_lr_multiplier=1., model_kwargs={},
                workers=2, pretrained=True, time_budget_hours=8)

def config_from(path):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    unknown = set(raw) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    c = DEFAULTS | raw
    for key in ("epochs", "patience", "batch_size", "accumulation", "image_size"):
        if not isinstance(c[key], int) or c[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if not 0 <= c["warmup_epochs"] < c["epochs"]:
        raise ValueError("warmup_epochs must be smaller than epochs")
    if c["time_budget_hours"] <= 0 or c["lr"] <= 0 or c["workers"] < 0:
        raise ValueError("Invalid budget, learning rate or workers")
    if c["class_weighting"] not in {None, "none", "inverse_frequency", "effective_number"}:
        raise ValueError("class_weighting must be none, inverse_frequency or effective_number")
    if not 0 <= c["label_smoothing"] < 1 or c["mixup_alpha"] < 0:
        raise ValueError("Invalid label smoothing or mixup alpha")
    if not 0 <= c["effective_number_beta"] < 1:
        raise ValueError("effective_number_beta must be in [0,1)")
    if c["weighted_loss"] and c["class_weighting"] not in {None, "inverse_frequency"}:
        raise ValueError("Legacy weighted_loss conflicts with class_weighting")
    if not isinstance(c["model_kwargs"], dict):
        raise ValueError("model_kwargs must be an object")
    if not isinstance(c["trainable_last_blocks"], int) or c["trainable_last_blocks"] < 0:
        raise ValueError("trainable_last_blocks must be a nonnegative integer")
    if c["freeze_backbone"] and c["trainable_last_blocks"]:
        raise ValueError("Cannot freeze the backbone and fine-tune its last blocks")
    if c["head_lr_multiplier"] <= 0:
        raise ValueError("head_lr_multiplier must be positive")
    return c

def preprocess(image, size=224, augment=False):
    image = image.convert("RGB")
    if augment and random.random() < .5:
        image = ImageOps.mirror(image)
    if augment and random.random() < .5:
        image = ImageOps.flip(image)
    image = image.resize((size, size), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.
    return torch.from_numpy(np.ascontiguousarray((array - MEAN) / STD))

def load_split(path):
    raw = Path(path).read_bytes()
    from benchmark.dataset import load_manifest
    split = load_manifest(path)
    if split.get("classes") != CLASSES:
        raise ValueError("Class order does not match the benchmark")
    samples = split["samples"]
    if not samples or any(s["split"] not in {"train", "validation", "test"} for s in samples):
        raise ValueError("Invalid split manifest")
    paths = [s["path"] for s in samples]
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate sample path")
    grouped = {}
    for sample in samples:
        if not isinstance(sample["label"], int) or not 0 <= sample["label"] < len(CLASSES):
            raise ValueError("Invalid class index")
        group = sample.get("group", sample["path"])
        if group in grouped and grouped[group] != sample["split"]:
            raise ValueError("Duplicate group crosses split boundaries")
        grouped[group] = sample["split"]
    split["file_sha256"] = hashlib.sha256(raw).hexdigest()
    return split

class SolarDataset(Dataset):
    def __init__(self, root, manifest, partition, size=224):
        if partition not in {"train", "validation"}:
            raise ValueError("Training code cannot load test data; use the locked final evaluator")
        self.root = Path(root).resolve()
        self.samples = [s for s in manifest["samples"] if s["split"] == partition]
        self.augment = partition == "train"
        self.size = size
        if not self.samples:
            raise ValueError(f"Empty {partition} partition")
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, index):
        sample = self.samples[index]
        path = (self.root / sample["path"]).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Image path escapes dataset")
        with Image.open(path) as image:
            tensor = preprocess(image, self.size, self.augment)
        return tensor, sample["label"]

def class_weights(samples):
    counts = np.bincount([s["label"] for s in samples], minlength=len(CLASSES))
    if (counts == 0).any():
        raise ValueError("Every class must occur in the training partition")
    return torch.tensor(counts.sum() / (len(CLASSES) * counts), dtype=torch.float32)

def effective_number_weights(samples, beta=.9999):
    counts = np.bincount([s["label"] for s in samples], minlength=len(CLASSES))
    if (counts == 0).any():
        raise ValueError("Every class must occur in the training partition")
    weights = (1. - beta) / (1. - np.power(beta, counts))
    weights /= weights.mean()
    return torch.tensor(weights, dtype=torch.float32)

def score(labels, predictions):
    cm = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)
    np.add.at(cm, (labels, predictions), 1)
    tp = cm.diagonal().astype(float)
    denominator = cm.sum(0) + cm.sum(1)
    f1 = np.divide(2 * tp, denominator, out=np.zeros_like(tp), where=denominator > 0)
    return dict(accuracy=float(tp.sum() / cm.sum()), macro_f1=float(f1.mean()), confusion_matrix=cm.tolist())

def make_model(config, pretrained=None):
    import timm
    return timm.create_model(config["model"], pretrained=config["pretrained"] if pretrained is None else pretrained,
                             num_classes=len(CLASSES), **config.get("model_kwargs", {}))

def set_trainable(model, classifier_only, last_blocks=0):
    if last_blocks and (not hasattr(model, "blocks") or last_blocks > len(model.blocks)):
        raise ValueError("Requested final blocks are unavailable on this model")
    for parameter in model.parameters():
        parameter.requires_grad = not classifier_only
    if classifier_only:
        model.eval()
        classifier = model.get_classifier()
        classifier.train()
        for parameter in classifier.parameters():
            parameter.requires_grad = True
    elif last_blocks:
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad = False
        for block in model.blocks[-last_blocks:]:
            block.train()
            for parameter in block.parameters():
                parameter.requires_grad = True
        for name in ("norm", "fc_norm"):
            layer = getattr(model, name, None)
            if isinstance(layer, nn.Module):
                layer.train()
                for parameter in layer.parameters():
                    parameter.requires_grad = True
        classifier = model.get_classifier()
        classifier.train()
        for parameter in classifier.parameters():
            parameter.requires_grad = True

def apply_mixup(images, target, alpha):
    if alpha <= 0:
        return images, target, target, 1.
    coefficient = float(np.random.beta(alpha, alpha))
    order = torch.randperm(images.shape[0], device=images.device)
    return coefficient * images + (1. - coefficient) * images[order], target, target[order], coefficient

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)

def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)

def save_checkpoint(path, value):
    tmp = Path(str(path) + ".tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)

@torch.inference_mode()
def validate(model, loader, device):
    model.eval()
    probabilities, labels = [], []
    for images, target in loader:
        probabilities.append(model(images.to(device)).softmax(1).cpu().numpy())
        labels.append(target.numpy())
    probabilities, labels = np.concatenate(probabilities), np.concatenate(labels)
    if not np.isfinite(probabilities).all():
        raise ValueError("Non-finite validation probabilities")
    return score(labels, probabilities.argmax(1)), probabilities, labels

def train(config, dataset, split_path, output, resume=False, device_name=None, model_factory=make_model):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    last_path = output / "last.pt"
    if last_path.exists() and not resume:
        raise ValueError("Output already has checkpoint; pass --resume or choose a new directory")
    manifest = load_split(split_path)
    set_seed(config["seed"])
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model_factory(config | {"pretrained": False} if resume and last_path.exists() else config).to(device)
    train_data = SolarDataset(dataset, manifest, "train", config["image_size"])
    valid_data = SolarDataset(dataset, manifest, "validation", config["image_size"])
    inverse_weights = class_weights(train_data.samples)  # Also validates support for unweighted CE.
    weighting = config["class_weighting"] or ("inverse_frequency" if config["weighted_loss"] else "none")
    weights = effective_number_weights(train_data.samples, config["effective_number_beta"]) if weighting == "effective_number" else inverse_weights
    criterion = nn.CrossEntropyLoss(weight=weights.to(device) if weighting != "none" else None,
                                    label_smoothing=config["label_smoothing"])
    generator = torch.Generator().manual_seed(config["seed"])
    train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True, num_workers=config["workers"], generator=generator, worker_init_fn=seed_worker, pin_memory=device.type == "cuda")
    valid_loader = DataLoader(valid_data, batch_size=config["batch_size"], shuffle=False, num_workers=config["workers"])
    set_trainable(model, config["freeze_backbone"] or config["warmup_epochs"] > 0)
    classifier_ids = {id(parameter) for parameter in model.get_classifier().parameters()}
    optimizer = torch.optim.AdamW([
        {"params": [parameter for parameter in model.parameters() if id(parameter) not in classifier_ids], "lr": config["lr"]},
        {"params": list(model.get_classifier().parameters()), "lr": config["lr"] * config["head_lr_multiplier"]},
    ], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"], eta_min=config["lr"] * .01)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    start_epoch, bad_epochs, history, best_key, elapsed_before = 0, 0, [], (-1., -1.), 0.
    if resume and last_path.exists():
        state = torch.load(last_path, map_location="cpu", weights_only=False)  # Only checkpoints created by this run.
        if state["config"] != config or state["split_file_sha256"] != manifest["file_sha256"]:
            raise ValueError("Resume configuration or split differs from checkpoint")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch, bad_epochs, history = state["epoch"] + 1, state["bad_epochs"], state["history"]
        best_key, elapsed_before = tuple(state["best_key"]), state["elapsed_seconds"]
        random.setstate(state["rng"]["python"])
        np.random.set_state(state["rng"]["numpy"])
        torch.set_rng_state(state["rng"]["torch"])
        generator.set_state(state["rng"]["loader"])
        if device.type == "cuda" and state["rng"]["cuda"] is not None:
            torch.cuda.set_rng_state_all(state["rng"]["cuda"])
    started = time.monotonic()
    atomic_json(output / "config.json", config)
    status = "completed"
    for epoch in range(start_epoch, config["epochs"]):
        if elapsed_before + time.monotonic() - started >= config["time_budget_hours"] * 3600:
            status = "time_budget_reached"
            break
        if bad_epochs >= config["patience"]:
            status = "early_stopped"
            break
        classifier_only = config["freeze_backbone"] or epoch < config["warmup_epochs"]
        set_trainable(model, classifier_only, config["trainable_last_blocks"] if not classifier_only else 0)
        if classifier_only:
            pass  # set_trainable also freezes normalization statistics.
        else:
            model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss, count = 0., 0
        for step, (images, target) in enumerate(train_loader):
            images, target = images.to(device), target.to(device)
            images, target_a, target_b, mix = apply_mixup(images, target, config["mixup_alpha"])
            window_start = (step // config["accumulation"]) * config["accumulation"]
            window_batches = min(config["accumulation"], len(train_loader) - window_start)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(images)
                loss = mix * criterion(logits, target_a) + (1. - mix) * criterion(logits, target_b)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            scaler.scale(loss / window_batches).backward()
            if (step + 1) % config["accumulation"] == 0 or step + 1 == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            running_loss += loss.item() * target.numel()
            count += target.numel()
        metrics, probabilities, labels = validate(model, valid_loader, device)
        scheduler.step()
        key = (metrics["accuracy"], metrics["macro_f1"])
        improved = key > best_key
        bad_epochs = 0 if improved else bad_epochs + 1
        if improved:
            best_key = key
        history.append(dict(epoch=epoch, train_loss=running_loss / count, validation=metrics, learning_rate=optimizer.param_groups[0]["lr"]))
        state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), scaler=scaler.state_dict(), epoch=epoch, bad_epochs=bad_epochs, history=history, best_key=best_key, config=config, classes=CLASSES, split_hash=manifest.get("split_hash"), split_file_sha256=manifest["file_sha256"], elapsed_seconds=elapsed_before + time.monotonic() - started,
                     rng=dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(), loader=generator.get_state(), cuda=torch.cuda.get_rng_state_all() if device.type == "cuda" else None))
        save_checkpoint(last_path, state)
        if improved:
            save_checkpoint(output / "best.pt", state)
            np.savez_compressed(output / "validation_predictions.npz", probabilities=probabilities, labels=labels, ids=np.array([str(s["id"]) for s in valid_data.samples]))
        atomic_json(output / "history.json", history)
        print(json.dumps(dict(run=output.name, epoch=epoch + 1, configured_epochs=config["epochs"], accuracy=metrics["accuracy"], macro_f1=metrics["macro_f1"], improved=improved, elapsed_seconds=state["elapsed_seconds"])), flush=True)
    result = dict(status=status, epochs_completed=len(history), best_validation=dict(accuracy=best_key[0], macro_f1=best_key[1]) if history else None, config=config, split_hash=manifest.get("split_hash"), split_file_sha256=manifest["file_sha256"], elapsed_seconds=elapsed_before + time.monotonic() - started, device=str(device), test_evaluated=False)
    atomic_json(output / "result.json", result)
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="benchmark/split.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    train(config_from(args.config), args.dataset, args.split, args.output, args.resume, args.device)

if __name__ == "__main__":
    main()
