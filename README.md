# MSIQ Metrics

`msiq-metrics` is a compact Python package for **Moment-based Scale-Invariant Quality** (MSIQ) diagnostics.

MSIQ is **not** intended as a universal perceptual image-quality metric. It is a complementary, model-free and scale-free diagnostic measure for checking whether an image-processing or super-resolution procedure preserves the global scale-invariant moment geometry of the reference image.

## Installation

For local development:

```bash
pip install -e .
```

The package can be installed via:

```bash
pip install msiq-metrics
```

## Basic usage

```python
from skimage import data
from skimage.transform import resize
from msiq import msiq, msiq_report

reference = data.camera() / 255.0
test = resize(reference, (768, 768), preserve_range=True)

score = msiq(reference, test, order=4, channel_mode="gray", distance="rmse")
print(score)

report = msiq_report(reference, test, order=4, channel_mode="gray")
print(report["channel_scores"])
```

## Color protocols

MSIQ can be computed not only on grayscale images but also in several color representations:

```python
msiq(gt, sr, channel_mode="gray")
msiq(gt, sr, channel_mode="rgb")
msiq(gt, sr, channel_mode="y")
msiq(gt, sr, channel_mode="ycbcr")
msiq(gt, sr, channel_mode="hsv")
msiq(gt, sr, channel_mode="hsv_circular")
```

The `hsv_circular` mode represents hue using two channels,

\[
\cos(2\pi H),\qquad \sin(2\pi H),
\]

which avoids the artificial discontinuity of hue at the 0/1 boundary.

## Moment matrix

The normalized central geometric moments are represented as a matrix

\[
\mathcal{N}_N(I)=\bigl(\nu_{pq}(I)\bigr)_{0\leq p,q\leq N}.
\]

Distances can be computed on the full square region or on the triangular region \(p+q\leq N\):

```python
msiq(gt, sr, order=6, region="triangular")
msiq(gt, sr, order=6, region="square")
```

## Distances

Supported distances:

```python
msiq(gt, sr, distance="rmse")
msiq(gt, sr, distance="frobenius")
msiq(gt, sr, distance="mae")
msiq(gt, sr, distance="max")
msiq(gt, sr, distance="weighted", weighting="inverse_order")
```

## Protocol comparison

```python
from msiq import compare_protocol

rows = compare_protocol(
    gt,
    sr,
    order=4,
    channel_modes=["gray", "y", "rgb", "hsv_circular"],
    distances=["rmse", "weighted"],
)
```

## Scale robustness diagnostic

```python
from msiq.diagnostics import scale_robustness_test

rows = scale_robustness_test(
    image,
    scales=[0.5, 0.75, 1.5, 2.0, 3.0],
    order=4,
    channel_mode="gray",
)
```

This diagnostic is useful because normalized moments are exactly scale-invariant in the continuous setting, while every discrete implementation is only an approximation.

## Citation

A formal citation will be added after the corresponding paper is published.

## License

MIT.
