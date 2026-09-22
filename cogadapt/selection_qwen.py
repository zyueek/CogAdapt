"""Selected experimental functions; see provenance/code_excerpts.json.

This review module omits model loading, datasets, and experiment runners.
"""

from __future__ import annotations
from collections.abc import Sequence
import numpy as np


METRIC_WEIGHTS = {
    "confidence": 0.15,
    "choice_change": 0.25,
    "hidden_shift": 0.20,
    "moe_write": 0.20,
    "residual_integration": 0.20,
}


def _standardize(values: np.ndarray) -> np.ndarray:
    scale = float(values.std())
    return (values - values.mean()) / (scale if scale > 1e-9 else 1.0)


def combine_metrics(metrics: dict[str, Sequence[float]]) -> np.ndarray:
    count = len(next(iter(metrics.values())))
    combined = np.zeros(count, dtype=float)
    for name, weight in METRIC_WEIGHTS.items():
        combined += weight * _standardize(np.asarray(metrics[name], dtype=float))
    # V8 depth priors are deliberately mild and cannot swamp an example-specific z score.
    for block in range(count):
        if 31 <= block <= 41:
            combined[block] += 0.20
        if block in {25, 29}:
            combined[block] += 0.10
        if block == 39:
            combined[block] += 0.15
    return combined


def active_block_count(difficulty_z: float, hard_blocks: int = 8) -> int:
    if difficulty_z < -0.5:
        return 4
    if difficulty_z > 0.5:
        return hard_blocks
    return 6


def select_active_blocks(
    scores: Sequence[float], difficulty_z: float, hard_blocks: int = 8
) -> list[int]:
    count = min(active_block_count(difficulty_z, hard_blocks), len(scores))
    selected = np.argsort(np.asarray(scores, dtype=float), kind="stable")[-count:].tolist()
    if difficulty_z > 0.5 and len(scores) > 39 and 39 not in selected:
        selected[0] = 39
    return sorted(set(int(value) for value in selected))
