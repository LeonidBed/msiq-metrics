"""Main MSIQ metric API."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from .color import extract_channels, default_channel_weights, normalize_weights
from .distances import moment_distance
from .moments import normalized_moment_matrix


def _aggregate_scores(scores: Dict[str, float], weights: Optional[Dict[str, float]] = None, channel_mode: str = "gray") -> float:
    if weights is None:
        weights = default_channel_weights(scores.keys(), channel_mode=channel_mode)
    weights = normalize_weights({k: weights.get(k, 0.0) for k in scores.keys()})
    return float(sum(weights[k] * scores[k] for k in scores.keys()))


def msiq(
    reference: np.ndarray,
    test: np.ndarray,
    *,
    order: int = 4,
    channel_mode: str = "gray",
    distance: str = "rmse",
    region: str = "triangular",
    include_trivial: bool = False,
    weighting: str = "uniform",
    channel_weights: Optional[Dict[str, float]] = None,
    clip: bool = True,
) -> float:
    """Compute MSIQ between a reference image and a test image.

    Parameters
    ----------
    reference, test:
        Input images. They may be grayscale or RGB-like arrays. If the shapes
        differ, no automatic resizing is performed; MSIQ itself is scale-free
        through normalized moments, and therefore direct same-size resizing is
        not required. The images must still represent comparable content.
    order:
        Maximum moment order.
    channel_mode:
        ``gray``, ``rgb``, ``y``, ``ycbcr``, ``hsv``, or ``hsv_circular``.
    distance:
        ``rmse``, ``frobenius``, ``mae``, ``max``, or ``weighted``.
    region:
        ``triangular`` for ``p+q<=N`` or ``square`` for all ``p,q<=N``.
    include_trivial:
        Whether to include ``nu00``, ``nu10`` and ``nu01`` in the distance.
    weighting:
        Weighting scheme for weighted distance.
    channel_weights:
        Optional channel aggregation weights.
    clip:
        Clip input values to [0,1].

    Returns
    -------
    float
        Aggregated MSIQ score. Lower values indicate better preservation of
        global normalized moment structure.
    """
    report = msiq_report(
        reference,
        test,
        order=order,
        channel_mode=channel_mode,
        distance=distance,
        region=region,
        include_trivial=include_trivial,
        weighting=weighting,
        channel_weights=channel_weights,
        clip=clip,
        return_matrices=False,
    )
    return float(report["score"])


def msiq_report(
    reference: np.ndarray,
    test: np.ndarray,
    *,
    order: int = 4,
    channel_mode: str = "gray",
    distance: str = "rmse",
    region: str = "triangular",
    include_trivial: bool = False,
    weighting: str = "uniform",
    channel_weights: Optional[Dict[str, float]] = None,
    clip: bool = True,
    return_matrices: bool = True,
) -> Dict[str, object]:
    """Compute MSIQ and return a detailed report."""
    ref_channels = extract_channels(reference, channel_mode=channel_mode)
    test_channels = extract_channels(test, channel_mode=channel_mode)

    if set(ref_channels.keys()) != set(test_channels.keys()):
        raise RuntimeError("internal channel extraction mismatch")

    channel_scores: Dict[str, float] = {}
    ref_matrices: Dict[str, np.ndarray] = {}
    test_matrices: Dict[str, np.ndarray] = {}
    diff_matrices: Dict[str, np.ndarray] = {}

    for name in ref_channels.keys():
        ref_m = normalized_moment_matrix(ref_channels[name], order=order, clip=clip)
        test_m = normalized_moment_matrix(test_channels[name], order=order, clip=clip)
        score = moment_distance(
            ref_m,
            test_m,
            distance=distance,
            region=region,
            include_trivial=include_trivial,
            weighting=weighting,
        )
        channel_scores[name] = float(score)

        if return_matrices:
            ref_matrices[name] = ref_m
            test_matrices[name] = test_m
            diff_matrices[name] = test_m - ref_m

    if channel_weights is None:
        weights = default_channel_weights(channel_scores.keys(), channel_mode=channel_mode)
    else:
        weights = channel_weights
    weights = normalize_weights({k: weights.get(k, 0.0) for k in channel_scores.keys()})

    aggregate = _aggregate_scores(channel_scores, weights, channel_mode=channel_mode)

    result: Dict[str, object] = {
        "score": aggregate,
        "order": order,
        "channel_mode": channel_mode,
        "distance": distance,
        "region": region,
        "include_trivial": include_trivial,
        "weighting": weighting,
        "channel_scores": channel_scores,
        "channel_weights": weights,
    }

    if return_matrices:
        result.update(
            {
                "reference_moment_matrices": ref_matrices,
                "test_moment_matrices": test_matrices,
                "difference_matrices": diff_matrices,
            }
        )

    return result


def compare_protocol(
    reference: np.ndarray,
    test: np.ndarray,
    *,
    order: int = 4,
    channel_modes: Sequence[str] = ("gray", "y", "rgb", "hsv_circular"),
    distances: Sequence[str] = ("rmse", "weighted"),
    region: str = "triangular",
    include_trivial: bool = False,
    weighting: str = "uniform",
    clip: bool = True,
):
    """Evaluate MSIQ under several color and distance protocols.

    Returns a list of dictionaries so that pandas can be used optionally by
    the caller.
    """
    rows = []
    for mode in channel_modes:
        for dist in distances:
            score = msiq(
                reference,
                test,
                order=order,
                channel_mode=mode,
                distance=dist,
                region=region,
                include_trivial=include_trivial,
                weighting=weighting,
                clip=clip,
            )
            rows.append(
                {
                    "order": order,
                    "channel_mode": mode,
                    "distance": dist,
                    "region": region,
                    "include_trivial": include_trivial,
                    "weighting": weighting,
                    "score": score,
                }
            )
    return rows
