import numpy as np
from skimage import data
from skimage.transform import resize

from msiq import msiq, normalized_moment_matrix, msiq_report, compare_protocol


def test_identical_image_zero():
    img = data.camera() / 255.0
    assert msiq(img, img, order=4) == 0.0


def test_normalized_moment_matrix_shape():
    img = data.camera() / 255.0
    mat = normalized_moment_matrix(img, order=5)
    assert mat.shape == (6, 6)
    assert np.isfinite(mat).all()


def test_resized_image_small_score():
    img = data.camera() / 255.0
    scaled = resize(img, (768, 768), preserve_range=True)
    score = msiq(img, scaled, order=4)
    assert np.isfinite(score)
    assert score >= 0.0


def test_color_report():
    img = data.astronaut() / 255.0
    report = msiq_report(img, img, order=3, channel_mode="hsv_circular")
    assert report["score"] == 0.0
    assert set(report["channel_scores"].keys()) == {"H_cos", "H_sin", "S", "V"}


def test_compare_protocol():
    img = data.camera() / 255.0
    rows = compare_protocol(img, img, order=3, channel_modes=["gray"], distances=["rmse", "weighted"])
    assert len(rows) == 2
    assert all(row["score"] == 0.0 for row in rows)
