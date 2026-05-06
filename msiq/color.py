"""Color-space conversion utilities for MSIQ."""

from __future__ import annotations

import math
from typing import Dict, Iterable, Tuple

import numpy as np
from skimage import color


def as_float01(image: np.ndarray, *, clip: bool = True) -> np.ndarray:
    """Convert an image to float64, usually in the range [0, 1]."""
    arr = np.asarray(image)
    if arr.dtype == np.uint8:
        out = arr.astype(np.float64) / 255.0
    elif arr.dtype == np.uint16:
        out = arr.astype(np.float64) / 65535.0
    else:
        out = arr.astype(np.float64)
        if out.size > 0 and np.nanmax(out) > 1.0:
            out = out / 255.0
    if clip:
        out = np.clip(out, 0.0, 1.0)
    return out


def _ensure_rgb_float(image: np.ndarray) -> np.ndarray:
    arr = as_float01(image)
    if arr.ndim == 2:
        return np.repeat(arr[..., None], 3, axis=2)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        return arr[..., :3]
    raise ValueError("image must be grayscale or RGB-like")


def extract_channels(image: np.ndarray, channel_mode: str = "gray") -> Dict[str, np.ndarray]:
    """Extract comparison channels from an image.

    Supported modes
    ---------------
    ``gray``:
        Convert RGB to luminance-like grayscale using scikit-image.
    ``rgb``:
        Return R, G, B channels.
    ``y``:
        Return Y channel of YCbCr.
    ``ycbcr``:
        Return Y, Cb, Cr channels, normalized approximately to [0,1].
    ``hsv``:
        Return H, S, V channels directly.
    ``hsv_circular``:
        Return hue as cos/sin channels, plus S and V. This avoids the
        discontinuity of hue at the 0/1 boundary.

    Returns
    -------
    dict[str, numpy.ndarray]
        Mapping from channel names to 2D float arrays.
    """
    mode = channel_mode.lower()
    arr = as_float01(image)

    if mode in {"gray", "grey", "grayscale"}:
        if arr.ndim == 2:
            return {"gray": arr}
        rgb = _ensure_rgb_float(arr)
        return {"gray": color.rgb2gray(rgb)}

    rgb = _ensure_rgb_float(arr)

    if mode == "rgb":
        return {"R": rgb[..., 0], "G": rgb[..., 1], "B": rgb[..., 2]}

    if mode in {"y", "luma", "luminance"}:
        ycbcr = color.rgb2ycbcr(rgb)
        # skimage Y is in [16, 235] for valid range; normalize conservatively.
        y = np.clip((ycbcr[..., 0] - 16.0) / 219.0, 0.0, 1.0)
        return {"Y": y}

    if mode == "ycbcr":
        ycbcr = color.rgb2ycbcr(rgb)
        y = np.clip((ycbcr[..., 0] - 16.0) / 219.0, 0.0, 1.0)
        cb = np.clip((ycbcr[..., 1] - 16.0) / 224.0, 0.0, 1.0)
        cr = np.clip((ycbcr[..., 2] - 16.0) / 224.0, 0.0, 1.0)
        return {"Y": y, "Cb": cb, "Cr": cr}

    if mode == "hsv":
        hsv = color.rgb2hsv(rgb)
        return {"H": hsv[..., 0], "S": hsv[..., 1], "V": hsv[..., 2]}

    if mode in {"hsv_circular", "hsv-circular", "circular_hsv"}:
        hsv = color.rgb2hsv(rgb)
        h = hsv[..., 0]
        return {
            "H_cos": 0.5 + 0.5 * np.cos(2.0 * math.pi * h),
            "H_sin": 0.5 + 0.5 * np.sin(2.0 * math.pi * h),
            "S": hsv[..., 1],
            "V": hsv[..., 2],
        }

    raise ValueError(
        "Unknown channel_mode. Use 'gray', 'rgb', 'y', 'ycbcr', 'hsv', or 'hsv_circular'."
    )


def default_channel_weights(channel_names: Iterable[str], channel_mode: str = "gray") -> Dict[str, float]:
    """Return default aggregation weights for extracted channels."""
    names = list(channel_names)
    mode = channel_mode.lower()
    if mode in {"gray", "grey", "grayscale", "y", "luma", "luminance"}:
        return {name: 1.0 for name in names}
    if mode == "rgb":
        return {"R": 1.0, "G": 1.0, "B": 1.0}
    if mode == "ycbcr":
        return {"Y": 1.0, "Cb": 0.25, "Cr": 0.25}
    if mode == "hsv":
        return {"H": 0.5, "S": 0.5, "V": 1.0}
    if mode in {"hsv_circular", "hsv-circular", "circular_hsv"}:
        return {"H_cos": 0.25, "H_sin": 0.25, "S": 0.5, "V": 1.0}
    return {name: 1.0 for name in names}


def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    total = float(sum(max(0.0, v) for v in weights.values()))
    if total <= 0:
        raise ValueError("channel weights must contain at least one positive value")
    return {k: max(0.0, float(v)) / total for k, v in weights.items()}
