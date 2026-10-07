import cv2
import numpy as np
import matplotlib.pyplot as plt

# Lab 05 - Task 2
# Choosing the Correct Adaptive Threshold Strategy

IMAGE_PATH = "document.jpg"
OUTPUT_PATH = "task2_adaptive_comparison.png"

img = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("Put the same document image from Task 1 at: " + IMAGE_PATH)

# At least three block sizes and three C values.
# Nine combinations are tested for Gaussian adaptive thresholding.
block_sizes = [11, 31, 61]
c_values = [2, 10, 20]

results = []
for block in block_sizes:
    for c in c_values:
        result = cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, block, c
        )
        results.append((result, f"Gaussian\nblock={block}, C={c}"))

# Compare Mean and Gaussian for a reasonable final parameter set.
mean_result = cv2.adaptiveThreshold(
    img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
    cv2.THRESH_BINARY, 31, 10
)
gaussian_result = cv2.adaptiveThreshold(
    img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY, 31, 10
)

fig, axes = plt.subplots(3, 4, figsize=(16, 12))
for ax, (result, title) in zip(axes.flat, results):
    ax.imshow(result, cmap="gray")
    ax.set_title(title)
    ax.axis("off")

axes[2, 1].imshow(mean_result, cmap="gray")
axes[2, 1].set_title("Mean\nblock=31, C=10")
axes[2, 1].axis("off")

axes[2, 2].imshow(gaussian_result, cmap="gray")
axes[2, 2].set_title("Gaussian\nblock=31, C=10")
axes[2, 2].axis("off")

axes[2, 3].imshow(img, cmap="gray")
axes[2, 3].set_title("Original")
axes[2, 3].axis("off")

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
plt.show()

print("Parameter study:")
print("blockSize values:", block_sizes)
print("C values:", c_values)
print("\nAnalysis:")
print("1. A neighbourhood that is too small reacts strongly to tiny local intensity changes")
print("   and can create broken characters or black noise.")
print("2. A neighbourhood that is too large becomes less local and may behave more like")
print("   a global threshold, reducing its ability to handle uneven illumination.")
print("3. Increasing C changes the local threshold and generally changes how much dark")
print("   detail is classified as foreground. Large C can remove weak/noisy foreground.")
print("4. The cleanest combination should be selected by visual inspection of character")
print("   continuity and background noise.")
print("5. Mean and Gaussian should be compared on the actual image; Gaussian usually")
print("   gives smoother local weighting because nearby pixels receive more influence.")
