import cv2
import numpy as np
import matplotlib.pyplot as plt

# Lab 05 - Task 7
# Separating Touching Coins using Marker-Based Watershed

IMAGE_PATH = "coins.jpg"
OUTPUT_PATH = "task7_watershed_pipeline.png"

img = cv2.imread(IMAGE_PATH)
if img is None:
    raise FileNotFoundError("Put the touching-coins image at: " + IMAGE_PATH)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# 1-2. Preprocessing and thresholding
_, binary = cv2.threshold(
    blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)

# 3. Noise removal
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

# 4. Sure background
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# 5. Distance transform
dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

# 6. Sure foreground
distance_threshold = 0.45 * dist.max()
_, sure_fg = cv2.threshold(dist, distance_threshold, 255, 0)
sure_fg = np.uint8(sure_fg)

# 7. Unknown region
unknown = cv2.subtract(sure_bg, sure_fg)

# 8. Marker labeling
num_markers, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown == 255] = 0

# 9. Watershed
watershed_img = img.copy()
markers = cv2.watershed(watershed_img, markers)

# 10. Boundary visualization
watershed_img[markers == -1] = [0, 0, 255]

fig, axes = plt.subplots(2, 4, figsize=(18, 9))
axes = axes.ravel()

axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0].set_title("Original")

axes[1].imshow(binary, cmap="gray")
axes[1].set_title("Threshold")

axes[2].imshow(sure_bg, cmap="gray")
axes[2].set_title("Sure Background")

axes[3].imshow(dist, cmap="gray")
axes[3].set_title("Distance Transform")

axes[4].imshow(sure_fg, cmap="gray")
axes[4].set_title("Sure Foreground")

axes[5].imshow(unknown, cmap="gray")
axes[5].set_title("Unknown Region")

axes[6].imshow(markers, cmap="nipy_spectral")
axes[6].set_title("Markers")

axes[7].imshow(cv2.cvtColor(watershed_img, cv2.COLOR_BGR2RGB))
axes[7].set_title("Final Watershed")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
plt.show()

print("Distance-transform foreground threshold:", distance_threshold)
print("Number of foreground markers:", num_markers - 1)
print("\nPipeline:")
print("Preprocessing -> Thresholding -> Noise removal -> Sure background ->")
print("Distance transform -> Sure foreground -> Unknown -> Markers -> Watershed")
