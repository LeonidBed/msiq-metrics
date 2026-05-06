from skimage import data
from skimage.transform import resize

from msiq import msiq, msiq_report, compare_protocol

reference = data.camera() / 255.0
test = resize(reference, (768, 768), preserve_range=True)

score = msiq(reference, test, order=4, channel_mode="gray", distance="rmse")
print("MSIQ_RMSE:", score)

report = msiq_report(reference, test, order=4, channel_mode="gray")
print(report["channel_scores"])

rows = compare_protocol(reference, test, order=4)
for row in rows:
    print(row)
