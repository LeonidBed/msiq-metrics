"""Diagnostic routines for MSIQ."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import numpy as np
from skimage.transform import resize

from .metrics import msiq


def resize_image(image: np.ndarray, scale: float, *, order: int = 1, preserve_range: bool = True) -> np.ndarray:
    """Resize an image by a scale factor using scikit-image."""
    if scale <= 0:
        raise ValueError("scale must be positive")
    arr = np.asarray(image)
    new_shape = tuple(max(1, int(round(s * scale))) for s in arr.shape[:2])
    if arr.ndim == 3:
        output_shape = new_shape + (arr.shape[2],)
    else:
        output_shape = new_shape
    out = resize(
        arr,
        output_shape,
        order=order,
        mode="reflect",
        anti_aliasing=(scale < 1.0),
        preserve_range=preserve_range,
    )
    return np.asarray(out, dtype=np.float64)


def scale_robustness_test(
    image: np.ndarray,
    *,
    scales: Sequence[float] = (0.5, 0.75, 1.5, 2.0),
    order: int = 4,
    channel_mode: str = "gray",
    distance: str = "rmse",
    region: str = "triangular",
    resize_order: int = 1,
) -> List[Dict[str, float]]:
    """Measure MSIQ between an image and uniformly rescaled versions.

    This diagnostic helps evaluate the discrete scale-sensitivity of the
    chosen moment computation protocol. Continuous normalized moments are
    scale-invariant; discrete computations are approximate.
    """
    rows = []
    for scale in scales:
        scaled = resize_image(image, scale, order=resize_order)
        score = msiq(
            image,
            scaled,
            order=order,
            channel_mode=channel_mode,
            distance=distance,
            region=region,
        )
        rows.append(
            {
                "scale": float(scale),
                "order": int(order),
                "channel_mode": channel_mode,
                "distance": distance,
                "region": region,
                "resize_order": int(resize_order),
                "msiq": float(score),
            }
        )
    return rows
