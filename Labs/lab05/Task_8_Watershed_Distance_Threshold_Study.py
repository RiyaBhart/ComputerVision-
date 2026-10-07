import cv2
import numpy as np
import matplotlib.pyplot as plt

# Lab 05 - Task 8
# Tuning Watershed for Difficult Object Separation

IMAGE_PATH = "coins.jpg"
OUTPUT_PATH = "task8_watershed_threshold_study.png"

img = cv2.imread(IMAGE_PATH)
if img is None:
    raise FileNotFoundError("Put the touching-coins image at: " + IMAGE_PATH)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)

_, binary = cv2.threshold(
    blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)

kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
sure_bg = cv2.dilate(opening, kernel, iterations=3)
dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

# Four experiments as required.
ratios = [0.30, 0.40, 0.50, 0.60]

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
records = []

for ax, ratio in zip(axes, ratios):
    threshold = ratio * dist.max()
    _, sure_fg = cv2.threshold(dist, threshold, 255, 0)
    sure_fg = np.uint8(sure_fg)

    unknown = cv2.subtract(sure_bg, sure_fg)

    marker_count, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    result = img.copy()
    watershed_markers = cv2.watershed(result, markers)
    result[watershed_markers == -1] = [0, 0, 255]

    separated_regions = len(np.unique(watershed_markers)) - 2
    # -1 is boundary and 1 is background, so subtracting 2 leaves object labels.

    records.append((ratio, marker_count - 1, separated_regions))

    ax.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    ax.set_title(
        f"Threshold={ratio:.2f}\n"
        f"Markers={marker_count-1}, Regions={separated_regions}"
    )
    ax.axis("off")

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
plt.show()

print("\nExperiment | Distance Threshold | Foreground Markers | Separated Regions")
for i, (ratio, markers, regions) in enumerate(records, 1):
    print(f"{i:9d} | {ratio:.2f} x max(dist) | {markers:18d} | {regions}")

print("\nAnalysis:")
print("A low distance threshold produces larger sure-foreground regions and may merge")
print("touching objects. A high threshold produces smaller, more conservative foreground")
print("regions and may create fewer markers, causing objects to remain merged.")
print("Choose the range that produces approximately one meaningful marker per object.")
