"""Geometric moment matrices."""

from __future__ import annotations

import numpy as np
from skimage.measure import moments, moments_central, moments_normalized

from .color import as_float01


def _prepare_gray(image: np.ndarray, *, clip: bool = True) -> np.ndarray:
    arr = as_float01(image, clip=clip)
    if arr.ndim != 2:
        raise ValueError("moment functions expect a single 2D channel")
    if not np.all(np.isfinite(arr)):
        raise ValueError("image contains NaN or infinite values")
    return arr.astype(np.float64, copy=False)


def raw_moment_matrix(image: np.ndarray, order: int = 4, *, clip: bool = True, spacing=None) -> np.ndarray:
    """Return raw geometric moments up to ``order`` for a 2D image channel."""
    if order < 0:
        raise ValueError("order must be non-negative")
    img = _prepare_gray(image, clip=clip)
    return moments(img, order=order, spacing=spacing)


def central_moment_matrix(
    image: np.ndarray,
    order: int = 4,
    *,
    clip: bool = True,
    center=None,
    spacing=None,
) -> np.ndarray:
    """Return central geometric moments up to ``order`` for a 2D image channel."""
    if order < 0:
        raise ValueError("order must be non-negative")
    img = _prepare_gray(image, clip=clip)
    m = moments(img, order=order, spacing=spacing)

    if center is None:
        if abs(m[0, 0]) < 1e-30:
            # Degenerate all-zero image. Use image center to avoid division by zero.
            center = ((img.shape[0] - 1) / 2.0, (img.shape[1] - 1) / 2.0)
        else:
            center = (m[1, 0] / m[0, 0], m[0, 1] / m[0, 0])

    return moments_central(img, center=center, order=order, spacing=spacing)


def normalized_moment_matrix(
    image: np.ndarray,
    order: int = 4,
    *,
    clip: bool = True,
    spacing=None,
) -> np.ndarray:
    """Return normalized central geometric moments up to ``order``.

    The returned matrix is ``(order+1, order+1)`` and contains the usual
    normalized central moments ``nu[p, q]``. In the continuous setting these
    quantities are invariant to uniform scaling. In discrete computations they
    should be interpreted as a numerical approximation whose robustness depends
    on the moment computation scheme and interpolation protocol.
    """
    mu = central_moment_matrix(image, order=order, clip=clip, spacing=spacing)
    nu = moments_normalized(mu, order=order)
    # skimage intentionally leaves some low-order normalized moments undefined
    # (NaN), e.g. nu00, nu10, nu01. For MSIQ we keep the standard
    # convention nu00=1 and nu10=nu01=0 so that the full moment matrix is
    # finite. These entries are excluded by default from comparisons.
    nu = np.nan_to_num(nu, nan=0.0, posinf=0.0, neginf=0.0)
    if order >= 0:
        nu[0, 0] = 1.0
    if order >= 1:
        nu[1, 0] = 0.0
        nu[0, 1] = 0.0
    return nu
