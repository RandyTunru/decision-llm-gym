"""Evaluation metrics for decision-model training."""
from __future__ import annotations

import math


def argmax_accuracy(preds: list[int], targets: list[int]) -> float:
    if not preds:
        return float("nan")
    return sum(p == t for p, t in zip(preds, targets)) / len(preds)


def soft_cross_entropy(pred_dists: list[list[float]], target_dists: list[list[float]]) -> float:
    """Mean -sum(target * log pred) over examples (the training objective)."""
    if not pred_dists:
        return float("nan")
    losses = []
    for p, t in zip(pred_dists, target_dists):
        losses.append(-sum(ti * math.log(max(pi, 1e-12)) for ti, pi in zip(t, p)))
    return sum(losses) / len(losses)


def score_mae(pred_levels: list[int], target_expected: list[float]) -> float:
    """Mean |predicted level - expected target level| for score questions."""
    if not pred_levels:
        return float("nan")
    return sum(abs(p - t) for p, t in zip(pred_levels, target_expected)) / len(pred_levels)
