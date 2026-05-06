"""Distances between moment matrices."""

from __future__ import annotations

import numpy as np

from .masks import moment_mask


def _weights(order: int, scheme: str = "uniform", region: str = "triangular", include_trivial: bool = False) -> np.ndarray:
    p, q = np.indices((order + 1, order + 1))
    scheme = scheme.lower()
    if scheme == "uniform":
        w = np.ones((order + 1, order + 1), dtype=float)
    elif scheme in {"inverse_order", "inverse"}:
        w = 1.0 / (1.0 + p + q)
    elif scheme in {"order", "linear_order"}:
        w = 1.0 + p + q
    elif scheme in {"squared_order", "quadratic_order"}:
        w = (1.0 + p + q) ** 2
    else:
        raise ValueError("Unknown weighting scheme")
    w = w * moment_mask(order, region=region, include_trivial=include_trivial)
    total = np.sum(w)
    if total <= 0:
        raise ValueError("empty moment mask")
    return w / total


def moment_distance(
    a: np.ndarray,
    b: np.ndarray,
    *,
    distance: str = "rmse",
    region: str = "triangular",
    include_trivial: bool = False,
    weighting: str = "uniform",
) -> float:
    """Compute a distance between two moment matrices.

    Parameters
    ----------
    a, b:
        Moment matrices of equal shape.
    distance:
        One of ``"rmse"``, ``"frobenius"``, ``"mae"``, ``"max"``, or
        ``"weighted"``.
    region:
        Moment-index mask: ``"triangular"`` or ``"square"``.
    include_trivial:
        Whether to include ``nu00``, ``nu10``, and ``nu01``.
    weighting:
        Weighting scheme for ``distance="weighted"``. Also accepted for
        ``distance="weighted_rmse"``.

    Returns
    -------
    float
        Non-negative distance.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if a.shape != b.shape:
        raise ValueError("moment matrices must have equal shape")
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("moment matrices must be square 2D arrays")

    order = a.shape[0] - 1
    mask = moment_mask(order, region=region, include_trivial=include_trivial)
    diff = a - b
    values = diff[mask]

    if values.size == 0:
        raise ValueError("empty moment mask")

    distance = distance.lower()
    if distance in {"rmse", "rms"}:
        return float(np.sqrt(np.mean(values ** 2)))
    if distance in {"frobenius", "f", "l2"}:
        return float(np.sqrt(np.sum(values ** 2)))
    if distance in {"mae", "l1_mean"}:
        return float(np.mean(np.abs(values)))
    if distance in {"max", "linf", "chebyshev"}:
        return float(np.max(np.abs(values)))
    if distance in {"weighted", "weighted_rmse", "w"}:
        w = _weights(order, scheme=weighting, region=region, include_trivial=include_trivial)
        return float(np.sqrt(np.sum(w * diff ** 2)))

    raise ValueError("Unknown distance. Use 'rmse', 'frobenius', 'mae', 'max', or 'weighted'.")
