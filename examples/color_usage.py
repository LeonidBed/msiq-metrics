from skimage import data
from skimage.transform import resize

from msiq import msiq, msiq_report

reference = data.astronaut() / 255.0
test = resize(reference, (600, 600), preserve_range=True)

for mode in ["gray", "y", "rgb", "ycbcr", "hsv", "hsv_circular"]:
    score = msiq(reference, test, order=4, channel_mode=mode, distance="rmse")
    print(mode, score)

report = msiq_report(reference, test, order=4, channel_mode="hsv_circular")
print(report["channel_scores"])
