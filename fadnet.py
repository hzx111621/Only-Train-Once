"""FADNet model and image datasets used by the release tests."""

from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from torchvision import models


def image_files(root):
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    # Match the original test scripts: jpg files first, then png files.
    return list(root.glob("**/*.jpg")) + list(root.glob("**/*.png"))


def limited_images(root, max_samples):
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be positive or None")
    files = image_files(root)[:max_samples]
    if not files:
        raise ValueError(f"No jpg/png images found in: {root}")
    return files


class RealFakeDataset(Dataset):
    """A binary dataset: real images have label 1 and synthetic images label 0."""

    def __init__(self, real_path, fake_path, transform=None, max_samples=None):
        self.transform = transform
        self.data = []
        self.labels = []
        for path in limited_images(real_path, max_samples):
            self.data.append(path)
            self.labels.append(1)
        for path in limited_images(fake_path, max_samples):
            self.data.append(path)
            self.labels.append(0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        path = self.data[index]
        label = self.labels[index]
        with Image.open(path) as source:
            image = source.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class MixedRealFakeDataset(Dataset):
    """A real set plus several named synthetic sources."""

    def __init__(self, real_path, fake_items, transform=None, real_max_samples=None, fake_max_samples=None):
        self.transform = transform
        self.data = []
        self.labels = []
        self.source_names = ["Real"] + [name for name, _ in fake_items]
        self.source_labels = []
        for path in limited_images(real_path, real_max_samples):
            self.data.append(path)
            self.labels.append(1)
            self.source_labels.append(0)
        for source_id, (_, fake_path) in enumerate(fake_items, start=1):
            for path in limited_images(fake_path, fake_max_samples):
                self.data.append(path)
                self.labels.append(0)
                self.source_labels.append(source_id)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        path = self.data[index]
        label = self.labels[index]
        with Image.open(path) as source:
            image = source.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class FADNet(nn.Module):
    """The ResNet-50 FADNet architecture used by the supplied checkpoint."""

    def __init__(self, backbone="resnet50", dropout_rate=0.3, feature_dim=512, activation="softplus"):
        super().__init__()
        if backbone != "resnet50":
            raise ValueError("The release checkpoint uses backbone='resnet50'.")
        self.num_classes = 2
        self.backbone_type = backbone
        self.activation_type = activation
        resnet = models.resnet50(weights=None)
        in_features = resnet.fc.in_features
        resnet.fc = nn.Identity()
        self.backbone = resnet
        self.feature_proj = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(1024, feature_dim),
        )
        self.evidence_layer = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, self.num_classes),
        )
        if activation == "softplus":
            self.activation_fn = nn.Softplus()
        elif activation == "relu":
            self.activation_fn = nn.ReLU()
        elif activation == "exp":
            self.activation_fn = lambda value: torch.exp(torch.clamp(value, max=10))
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, images, return_features=False):
        features = self.feature_proj(self.backbone(images))
        normalized_features = torch.nn.functional.normalize(features, dim=1)
        evidence = self.activation_fn(self.evidence_layer(features))
        alpha = evidence + 1
        strength = alpha.sum(dim=1, keepdim=True)
        probabilities = alpha / strength
        uncertainty = self.num_classes / strength
        if return_features:
            return alpha, probabilities, uncertainty, normalized_features
        return alpha, probabilities, uncertainty
