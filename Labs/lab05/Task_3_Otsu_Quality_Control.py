import cv2
import numpy as np
import matplotlib.pyplot as plt

# Lab 05 - Task 3
# Quality-Control System Using Otsu's Thresholding

IMAGE_PATH = "coins.jpg"
OUTPUT_PATH = "task3_otsu_comparison.png"

img = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("Put the grayscale/object image at: " + IMAGE_PATH)

# Original Otsu
threshold1, otsu1 = cv2.threshold(
    img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

# Slight preprocessing/contrast change
blurred = cv2.GaussianBlur(img, (5, 5), 0)
threshold2, otsu2 = cv2.threshold(
    blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

print(f"Otsu threshold on original image: {threshold1:.2f}")
print(f"Otsu threshold after Gaussian blur: {threshold2:.2f}")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(img, cmap="gray")
axes[0].set_title("Original")
axes[0].axis("off")

axes[1].hist(img.ravel(), bins=256, range=(0, 256))
axes[1].axvline(threshold1, linestyle="--", linewidth=2,
                label=f"Otsu={threshold1:.1f}")
axes[1].set_title("Grayscale Histogram")
axes[1].set_xlabel("Intensity")
axes[1].set_ylabel("Pixels")
axes[1].legend()

axes[2].imshow(otsu1, cmap="gray")
axes[2].set_title(f"Otsu Binary Mask (T={threshold1:.1f})")
axes[2].axis("off")

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
plt.show()

print("\nWhy Otsu is useful:")
print("Otsu automatically selects a threshold from the image histogram by maximizing")
print("between-class variance. Therefore the programmer does not have to manually")
print("guess a threshold when the image contains reasonably separable object/background classes.")
