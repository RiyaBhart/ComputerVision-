import cv2
import numpy as np
import matplotlib.pyplot as plt

# Lab 05 - Task 1
# Automated Inspection of a Document Under Uneven Lighting

IMAGE_PATH = "document.jpg"
OUTPUT_PATH = "task1_threshold_comparison.png"

img = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("Put the document image at: " + IMAGE_PATH)

# Three global thresholds
T1, T2, T3 = 80, 140, 200
_, global1 = cv2.threshold(img, T1, 255, cv2.THRESH_BINARY)
_, global2 = cv2.threshold(img, T2, 255, cv2.THRESH_BINARY)
_, global3 = cv2.threshold(img, T3, 255, cv2.THRESH_BINARY)

# Adaptive threshold
block_size = 31
C = 10
adaptive = cv2.adaptiveThreshold(
    img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY, block_size, C
)

fig, axes = plt.subplots(1, 5, figsize=(20, 5))
results = [
    (img, "Original"),
    (global1, f"Global T={T1}"),
    (global2, f"Global T={T2}"),
    (global3, f"Global T={T3}"),
    (adaptive, f"Adaptive\nblock={block_size}, C={C}")
]

for ax, (image, title) in zip(axes, results):
    ax.imshow(image, cmap="gray")
    ax.set_title(title)
    ax.axis("off")

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
plt.show()

print("Selected parameters:")
print(f"Global thresholds: {T1}, {T2}, {T3}")
print(f"Adaptive: Gaussian, blockSize={block_size}, C={C}")
print("\nAnalysis:")
print("A single global threshold assumes that one intensity cutoff works everywhere.")
print("With uneven illumination, dark parts of the page can lose text while bright")
print("parts can contain background pixels classified as foreground.")
