"""Damage severity scoring for routing decisions.

A scalar in [0, 1] that summarizes how degraded an input is. Used to decide:
  - whether to use the diffusion fallback for super-resolution
  - which CodeFormer `w` to apply (low w favors quality on heavy damage,
    high w favors identity preservation on mild damage)

Severity blends three signals:
  - classifier confidence (max score over predicted damage types)
  - damage mask coverage (fraction of pixels flagged by the robust detector,
    falling back to the model's damage_ratio if no robust mask is available)
  - damage-type severity weights (structured defects like scratches /
    missing_patch / stains weigh more than fade/sepia/noise)
"""
from __future__ import annotations

from typing import Optional

from app.old.damage_detector import DamageDetectionResult
from app.old.modeling import PredictionResult


_TYPE_WEIGHTS = {
    "scratches": 0.7,
    "missing_patch": 0.8,
    "stains": 0.65,
    "blur": 0.5,
    "fade": 0.45,
    "sepia": 0.45,
    "noise": 0.4,
}


def compute_severity(
    prediction: PredictionResult,
    damage_detection: Optional[DamageDetectionResult] = None,
) -> float:
    classifier_max = (
        max(prediction.type_scores.values()) if prediction.type_scores else 0.0
    )

    if damage_detection is not None:
        coverage = damage_detection.coverage
    else:
        coverage = prediction.damage_ratio
    # Coverage saturates at 25% — beyond that the image is uniformly degraded
    # and increasing coverage does not change the routing decision.
    coverage_term = min(coverage / 0.25, 1.0)

    type_severity = 0.0
    for damage_type in prediction.predicted_types:
        type_severity = max(type_severity, _TYPE_WEIGHTS.get(damage_type, 0.0))

    score = 0.4 * classifier_max + 0.3 * coverage_term + 0.3 * type_severity
    return float(min(max(score, 0.0), 1.0))


def codeformer_fidelity_weight(severity: float) -> float:
    if severity > 0.7:
        return 0.45
    if severity > 0.4:
        return 0.6
    return 0.75
