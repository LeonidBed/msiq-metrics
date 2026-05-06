"""Visualization helpers for moment matrices."""

from __future__ import annotations

import numpy as np

from .metrics import msiq_report


def moment_difference_matrix(reference, test, *, order=4, channel_mode="gray", channel=None):
    """Return the moment-difference matrix for one channel."""
    report = msiq_report(reference, test, order=order, channel_mode=channel_mode, return_matrices=True)
    diffs = report["difference_matrices"]
    if channel is None:
        channel = next(iter(diffs.keys()))
    return diffs[channel]


def plot_moment_difference(reference, test, *, order=4, channel_mode="gray", channel=None, ax=None):
    """Plot a heatmap of the moment-difference matrix.

    Requires matplotlib. The function returns ``(fig, ax)``.
    """
    import matplotlib.pyplot as plt

    diff = moment_difference_matrix(reference, test, order=order, channel_mode=channel_mode, channel=channel)
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    else:
        fig = ax.figure
    im = ax.imshow(diff)
    ax.set_title(f"Moment difference, order={order}, mode={channel_mode}")
    ax.set_xlabel("q")
    ax.set_ylabel("p")
    fig.colorbar(im, ax=ax)
    return fig, ax
