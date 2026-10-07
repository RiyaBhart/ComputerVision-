import cv2
import numpy as np
import matplotlib.pyplot as plt

# Lab 05 - Task 10
# Design a Segmentation System for an Unknown Image
#
# Chosen image: unevenly illuminated document.
# Chosen methods:
# 1. Global thresholding
# 2. Adaptive thresholding
# 3. Otsu's thresholding
#
# These three methods are deliberately compared because uneven illumination
# is the main challenge in this selected image.

IMAGE_PATH = "document.jpg"
OUTPUT_PATH = "task10_final_comparison.png"

img = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("Put the chosen unknown image at: " + IMAGE_PATH)

# Method 1: Global threshold
global_threshold = 140
_, global_result = cv2.threshold(
    img, global_threshold, 255, cv2.THRESH_BINARY
)

# Method 2: Adaptive threshold
adaptive_block = 31
adaptive_C = 10
adaptive_result = cv2.adaptiveThreshold(
    img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY, adaptive_block, adaptive_C
)

# Method 3: Otsu
otsu_value, otsu_result = cv2.threshold(
    img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
axes[0].imshow(img, cmap="gray")
axes[0].set_title("Original")

axes[1].imshow(global_result, cmap="gray")
axes[1].set_title(f"Global T={global_threshold}")

axes[2].imshow(adaptive_result, cmap="gray")
axes[2].set_title(
    f"Adaptive block={adaptive_block}, C={adaptive_C}"
)

axes[3].imshow(otsu_result, cmap="gray")
axes[3].set_title(f"Otsu T={otsu_value:.1f}")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
plt.show()

print("\nTechnical comparison")
print("=" * 70)

print("\n1. Global thresholding")
print("Assumption: one intensity threshold can separate foreground from background.")
print("Successfully identifies: text where its intensity is consistently separated.")
print("Incorrect segmentation: text/background in strongly differently illuminated areas.")
print("Most important parameter:", global_threshold)

print("\n2. Adaptive thresholding")
print("Assumption: a local neighbourhood provides a better threshold than one global value.")
print("Successfully identifies: text across regions with changing illumination.")
print("Incorrect segmentation: may create local noise or break characters if parameters are poor.")
print("Most important parameters:", adaptive_block, "and C")

print("\n3. Otsu thresholding")
print("Assumption: the histogram contains reasonably separable intensity classes.")
print("Successfully identifies: broad foreground/background separation when the histogram is bimodal.")
print("Incorrect segmentation: can struggle when illumination changes strongly across the page.")
print("Automatically selected threshold:", otsu_value)

print("\nConclusion:")
print("For an unevenly illuminated document, adaptive thresholding is expected to produce")
print("the most meaningful text regions because it calculates a local threshold rather than")
print("forcing the entire image to use one intensity cutoff. The decisive image property is")
print("the spatially varying illumination.")
