"""Moment-index masks."""

from __future__ import annotations

import numpy as np


def moment_mask(order: int, region: str = "triangular", include_trivial: bool = False) -> np.ndarray:
    """Return a boolean mask selecting moment coefficients.

    Parameters
    ----------
    order:
        Maximum moment order. The returned mask has shape ``(order + 1, order + 1)``.
    region:
        ``"triangular"`` selects indices with ``p + q <= order``.
        ``"square"`` selects all indices ``0 <= p,q <= order``.
    include_trivial:
        If ``False``, removes the entries ``(0,0)``, ``(1,0)``, and ``(0,1)``.
        For normalized central moments these entries are either constant or
        numerically close to zero and usually uninformative for comparison.

    Returns
    -------
    numpy.ndarray
        Boolean mask of selected entries.
    """
    if order < 0:
        raise ValueError("order must be non-negative")

    region = region.lower()
    p, q = np.indices((order + 1, order + 1))

    if region in {"triangular", "triangle", "p_plus_q_le_n"}:
        mask = (p + q) <= order
    elif region in {"square", "full", "pq_le_n"}:
        mask = np.ones((order + 1, order + 1), dtype=bool)
    else:
        raise ValueError("region must be 'triangular' or 'square'")

    if not include_trivial:
        for i, j in [(0, 0), (1, 0), (0, 1)]:
            if i <= order and j <= order:
                mask[i, j] = False

    return mask


def selected_moment_values(matrix: np.ndarray, region: str = "triangular", include_trivial: bool = False) -> np.ndarray:
    """Return selected moment values from a moment matrix as a 1D array."""
    if matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
    order = matrix.shape[0] - 1
    return matrix[moment_mask(order, region=region, include_trivial=include_trivial)]
