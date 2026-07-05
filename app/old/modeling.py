from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import models, transforms

from app.old.common import DAMAGE_TYPES, DEFAULT_MODEL_PATH


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass
class PredictionResult:
    predicted_types: list[str]
    type_scores: Dict[str, float]
    damage_mask: np.ndarray
    damage_ratio: float
    recommended_steps: list[str]


class OldPhotoDamageDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        image_size: int = 256,
        augment: bool = False,
    ) -> None:
        manifest_path = Path(manifest_path)
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)

        self.samples = [sample for sample in manifest["samples"] if sample["split"] == split]
        self.image_size = image_size
        self.augment = augment
        self.to_tensor = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _maybe_flip(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        if self.augment and torch.rand(1).item() > 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return image, mask

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[index]
        image = Image.open(sample["damaged_path"]).convert("RGB")
        mask = Image.open(sample["mask_path"]).convert("L")
        image, mask = self._maybe_flip(image, mask)

        image_tensor = self.to_tensor(image)
        mask = mask.resize((self.image_size, self.image_size))
        mask_tensor = transforms.ToTensor()(mask)

        labels = torch.tensor([sample["labels"][name] for name in DAMAGE_TYPES], dtype=torch.float32)
        has_mask = torch.tensor(float(mask_tensor.max().item() > 0.0), dtype=torch.float32)

        return {
            "image": image_tensor,
            "labels": labels,
            "mask": mask_tensor,
            "has_mask": has_mask,
        }


class OldPhotoDamageModel(nn.Module):
    def __init__(self, num_classes: int = len(DAMAGE_TYPES), pretrained: bool = True) -> None:
        super().__init__()
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        backbone = models.mobilenet_v3_large(weights=weights)
        self.encoder = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier_head = nn.Sequential(
            nn.Linear(960, 256),
            nn.Hardswish(),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes),
        )
        self.mask_head = nn.Sequential(
            nn.Conv2d(960, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.Hardswish(),
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.Hardswish(),
            nn.Conv2d(64, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(x)
        pooled = self.pool(features).flatten(1)
        logits = self.classifier_head(pooled)
        mask_logits = self.mask_head(features)
        mask_logits = F.interpolate(mask_logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return logits, mask_logits


class DiceBCELoss(nn.Module):
    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits: torch.Tensor, target: torch.Tensor, active_mask: torch.Tensor) -> torch.Tensor:
        bce = self.bce(logits, target)
        probs = torch.sigmoid(logits)
        probs_flat = probs.flatten(1)
        target_flat = target.flatten(1)
        intersection = (probs_flat * target_flat).sum(dim=1)
        dice = 1.0 - ((2.0 * intersection + self.smooth) / (probs_flat.sum(dim=1) + target_flat.sum(dim=1) + self.smooth))

        active_mask = active_mask.view(-1, 1, 1, 1)
        bce = (bce * active_mask).mean(dim=(1, 2, 3))
        loss = bce + dice * active_mask.view(-1)

        valid = active_mask.view(-1) > 0
        if valid.any():
            return loss[valid].mean()
        return torch.zeros((), device=logits.device)


def compute_binary_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    positives = y_true.sum()
    negatives = len(y_true) - positives
    if positives == 0 or negatives == 0:
        return float("nan")

    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1)
    positive_rank_sum = ranks[y_true == 1].sum()
    auc = (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)
    return float(auc)


def classification_report(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    y_pred = (y_score >= threshold).astype(np.int32)
    per_class: Dict[str, Dict[str, float]] = {}
    f1_scores = []
    auc_scores = []

    for index, name in enumerate(DAMAGE_TYPES):
        truth = y_true[:, index]
        pred = y_pred[:, index]
        tp = float(((truth == 1) & (pred == 1)).sum())
        fp = float(((truth == 0) & (pred == 1)).sum())
        fn = float(((truth == 1) & (pred == 0)).sum())

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        auc = compute_binary_auc(truth.astype(np.int32), y_score[:, index])

        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc": auc,
        }
        f1_scores.append(f1)
        if not math.isnan(auc):
            auc_scores.append(auc)

    return {
        "macro_f1": float(np.mean(f1_scores)) if f1_scores else 0.0,
        "macro_auc": float(np.mean(auc_scores)) if auc_scores else float("nan"),
        "per_class": per_class,
    }


def segmentation_report(mask_true: np.ndarray, mask_pred: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    pred = (mask_pred >= threshold).astype(np.uint8)
    truth = (mask_true >= threshold).astype(np.uint8)
    intersection = float((pred & truth).sum())
    union = float((pred | truth).sum())
    pred_sum = float(pred.sum())
    truth_sum = float(truth.sum())
    dice = (2.0 * intersection) / (pred_sum + truth_sum + 1e-8)
    iou = intersection / (union + 1e-8)
    return {"dice": dice, "iou": iou}


def build_transform(image_size: int = 256) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def load_model(weights_path: str | Path = DEFAULT_MODEL_PATH, device: str | torch.device | None = None) -> OldPhotoDamageModel:
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = OldPhotoDamageModel(pretrained=False)
    weights_path = Path(weights_path)
    if weights_path.exists():
        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def recommend_steps(predicted_types: Iterable[str], damage_ratio: float, face_detected: bool = False) -> list[str]:
    steps: list[str] = []
    predicted_types = list(predicted_types)
    if any(name in predicted_types for name in ("noise", "fade", "sepia")):
        steps.append("real_esrgan")
    if any(name in predicted_types for name in ("scratches", "missing_patch", "stains")) or damage_ratio > 0.01:
        steps.append("lama")
    if "blur" in predicted_types:
        steps.append("report_blur")
    if face_detected:
        steps.append("codeformer")
    return steps


@torch.no_grad()
def predict_image(
    model: OldPhotoDamageModel,
    image: Image.Image,
    device: str | torch.device | None = None,
    image_size: int = 256,
    threshold: float = 0.5,
    clean_threshold: float = 0.3,
    face_detected: bool = False,
) -> PredictionResult:
    device = torch.device(device or next(model.parameters()).device)
    transform = build_transform(image_size=image_size)
    tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    logits, mask_logits = model(tensor)
    scores = torch.sigmoid(logits)[0].cpu().numpy()
    mask_probs = torch.sigmoid(mask_logits)[0, 0].cpu().numpy()

    type_scores = {name: float(scores[index]) for index, name in enumerate(DAMAGE_TYPES)}
    predicted_types = [name for name, score in type_scores.items() if score >= threshold]
    # If the classifier is unconfident across the board, treat the image as clean.
    # Suppress the segmentation mask too — it's untrustworthy in this regime
    # (the trained model collapses to damage_ratio≈1.0 on out-of-distribution clean photos).
    image_is_clean = max(type_scores.values()) < clean_threshold
    if image_is_clean:
        predicted_types = []
        damage_mask = np.zeros_like(mask_probs, dtype=np.uint8)
    else:
        damage_mask = (mask_probs >= threshold).astype(np.uint8) * 255
    damage_ratio = float(damage_mask.mean() / 255.0)

    return PredictionResult(
        predicted_types=predicted_types,
        type_scores=type_scores,
        damage_mask=damage_mask,
        damage_ratio=damage_ratio,
        recommended_steps=recommend_steps(predicted_types, damage_ratio, face_detected=face_detected),
    )
