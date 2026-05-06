"""MSIQ: Moment-based Scale-Invariant Quality metrics.

The package provides diagnostic, moment-based metrics for assessing whether
super-resolved or rescaled images preserve global scale-invariant image geometry.

MSIQ is not intended as a universal perceptual image-quality metric. It is a
complementary diagnostic measure for global moment-geometry preservation.
"""

from .moments import normalized_moment_matrix, raw_moment_matrix, central_moment_matrix
from .distances import moment_distance
from .metrics import msiq, msiq_report, compare_protocol

__all__ = [
    "normalized_moment_matrix",
    "raw_moment_matrix",
    "central_moment_matrix",
    "moment_distance",
    "msiq",
    "msiq_report",
    "compare_protocol",
]

__version__ = "0.1.0"
