"""Selected experimental functions; see provenance/code_excerpts.json.

This review module omits model loading, datasets, and experiment runners.
"""

from __future__ import annotations
from collections.abc import Sequence
import numpy as np

LATE_ANCHOR = 38


METRIC_WEIGHTS = {
    "confidence": 0.15,
    "choice_change": 0.25,
    "hidden_shift": 0.20,
    "moe_write": 0.20,
    "residual_integration": 0.20,
}


def _standardize(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros_like(values, dtype=float)
    filled = values.copy()
    filled[~finite] = float(np.nanmedian(filled))
    scale = float(filled.std())
    return (filled - filled.mean()) / (scale if scale > 1e-9 else 1.0)


def combine_metrics(metrics: dict[str, Sequence[float]]) -> np.ndarray:
    count = len(next(iter(metrics.values())))
    combined = np.zeros(count, dtype=float)
    for name, weight in METRIC_WEIGHTS.items():
        combined += weight * _standardize(np.asarray(metrics[name], dtype=float))
    # Relative-depth mapping of the frozen V8 Qwen prior to 47 GLM layers.
    for block in range(count):
        if 30 <= block <= 40:
            combined[block] += 0.20
        if block in {24, 28}:
            combined[block] += 0.10
        if block == LATE_ANCHOR:
            combined[block] += 0.15
    return combined


def active_block_count(difficulty_z: float, hard_blocks: int = 10) -> int:
    if difficulty_z < -0.5:
        return 4
    if difficulty_z > 0.5:
        return hard_blocks
    return 6


def select_active_blocks(
    scores: Sequence[float], difficulty_z: float, hard_blocks: int = 10
) -> list[int]:
    count = min(active_block_count(difficulty_z, hard_blocks), len(scores))
    selected = np.argsort(np.asarray(scores, dtype=float), kind="stable")[-count:].tolist()
    if difficulty_z > 0.5 and len(scores) > LATE_ANCHOR and LATE_ANCHOR not in selected:
        selected[0] = LATE_ANCHOR
    return sorted(set(int(value) for value in selected))
