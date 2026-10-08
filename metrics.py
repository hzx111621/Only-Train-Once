"""Metric-only inference helpers for the FADNet release checkpoint."""

import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score
from torch.utils.data import DataLoader
from torchvision import transforms

from fadnet import FADNet


FIXED_THRESHOLD = 0.6911


def get_test_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def make_loader(dataset, batch_size=16, workers=0):
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)


def expected_calibration_error(labels, probabilities, bins=10):
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    edges = np.linspace(0, 1, bins + 1)
    result = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        selected = (probabilities > lower) & (probabilities <= upper)
        if selected.any():
            result += abs(probabilities[selected].mean() - labels[selected].mean()) * selected.mean()
    return float(result)


class MetricEvaluator:
    def __init__(self, model_path, device=None, dropout_rate=0.3, activation="softplus"):
        self.model_path = Path(model_path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = FADNet(dropout_rate=dropout_rate, activation=activation).to(self.device)
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        saved_metrics = checkpoint.get("metrics", {})
        self.saved_threshold = float(saved_metrics.get("uncertainty_threshold", FIXED_THRESHOLD))

    @torch.inference_mode()
    def find_best_threshold(self, loader):
        uncertainties, labels = [], []
        for images, batch_labels in loader:
            _, _, uncertainty = self.model(images.to(self.device))
            uncertainties.append(uncertainty.flatten().cpu().numpy())
            labels.append(batch_labels.numpy())
        uncertainties = np.concatenate(uncertainties)
        labels = np.concatenate(labels)
        candidates = np.linspace(float(uncertainties.min()), float(uncertainties.max()), 100)
        scores = [f1_score(labels, np.where(uncertainties > threshold, 0, 1), average="macro", zero_division=0) for threshold in candidates]
        return float(candidates[int(np.argmax(scores))])

    @torch.inference_mode()
    def evaluate(self, loader, threshold=FIXED_THRESHOLD):
        probabilities, uncertainties, labels = [], [], []
        started = time.perf_counter()
        for images, batch_labels in loader:
            _, batch_probabilities, batch_uncertainty = self.model(images.to(self.device))
            probabilities.append(batch_probabilities.cpu().numpy())
            uncertainties.append(batch_uncertainty.flatten().cpu().numpy())
            labels.append(batch_labels.numpy())
        probabilities = np.concatenate(probabilities)
        uncertainties = np.concatenate(uncertainties)
        labels = np.concatenate(labels)
        predictions = np.where(uncertainties > threshold, 0, 1)
        real_mask = labels == 1
        fake_mask = labels == 0
        return {
            "total_samples": int(labels.size),
            "real_samples": int(real_mask.sum()),
            "synthetic_samples": int(fake_mask.sum()),
            "acc": float((predictions == labels).mean()),
            "f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
            "ap": float(average_precision_score(labels, probabilities[:, 1])),
            "ece": expected_calibration_error(labels, probabilities[:, 1]),
            "real_acc": float((predictions[real_mask] == labels[real_mask]).mean()) if real_mask.any() else 0.0,
            "fake_acc": float((predictions[fake_mask] == labels[fake_mask]).mean()) if fake_mask.any() else 0.0,
            "avg_real_unc": float(uncertainties[real_mask].mean()) if real_mask.any() else 0.0,
            "avg_fake_unc": float(uncertainties[fake_mask].mean()) if fake_mask.any() else 0.0,
            "uncertainty_threshold": float(threshold),
            "inference_seconds": round(time.perf_counter() - started, 4),
        }


def print_metrics(name, metrics):
    print(f"\n[{name}]")
    keys = ("total_samples", "real_samples", "synthetic_samples", "acc", "f1", "ap", "ece", "real_acc", "fake_acc", "avg_real_unc", "avg_fake_unc", "uncertainty_threshold", "inference_seconds")
    for key in keys:
        value = metrics[key]
        print(f"{key}: {value:.6f}" if isinstance(value, float) else f"{key}: {value}")


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
