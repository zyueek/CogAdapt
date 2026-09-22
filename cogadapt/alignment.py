"""Selected experimental functions; see provenance/code_excerpts.json.

This review module omits model loading, datasets, and experiment runners.
"""

from __future__ import annotations
import numpy as np
from scipy.stats import spearmanr


def residualize(values: np.ndarray, controls: np.ndarray) -> np.ndarray:
    """OLS on original values, including an intercept; ranking happens later."""
    design = np.column_stack([np.ones(len(controls)), controls])
    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError("Code-size control design is rank deficient")
    residual = values - design @ np.linalg.lstsq(design, values, rcond=None)[0]
    np.testing.assert_allclose(design.T @ residual, 0, atol=1e-7)
    return residual


def rank_correlation(x, y):
    """Spearman correlation with average tied ranks and paired finite filtering."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.ndim != 1 or x.shape != y.shape:
        raise ValueError("Expected equally sized vectors")
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(spearmanr(x, y).statistic)


def length_controlled_spearman(x, y, controls):
    """The paper heatmap protocol: OLS on values first, then rank correlation.

    This is not the alternative convention that residualizes ranks.
    Use an n-by-2 control array: log1p(character count), nonblank-line count.
    """
    x, y, controls = map(lambda a: np.asarray(a, dtype=float), (x, y, controls))
    if x.ndim != 1 or x.shape != y.shape or controls.ndim != 2:
        raise ValueError("Expected paired vectors and a control matrix")
    if len(controls) != len(x) or not all(np.isfinite(a).all() for a in (x, y, controls)):
        raise ValueError("Mismatched or nonfinite observations")
    return rank_correlation(residualize(x, controls), residualize(y, controls))
