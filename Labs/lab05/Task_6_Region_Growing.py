import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# Lab 05 - Task 6
# Segmenting a Region Inside a Medical Image using Region Growing

IMAGE_PATH = "brain_mri.jpg"
OUTPUT_PATH = "task6_region_growing.png"

img = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("Put the brain MRI image at: " + IMAGE_PATH)

def region_grow(image, seed, difference_threshold):
    h, w = image.shape
    sr, sc = seed

    if not (0 <= sr < h and 0 <= sc < w):
        raise ValueError("Seed point is outside the image.")

    visited = np.zeros((h, w), dtype=np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)

    seed_value = int(image[sr, sc])
    queue = deque([(sr, sc)])
    visited[sr, sc] = 1

    # 8-connected neighbourhood
    neighbours = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]

    while queue:
        r, c = queue.popleft()
        current_value = int(image[r, c])

        if abs(current_value - seed_value) <= difference_threshold:
            mask[r, c] = 255

            for dr, dc in neighbours:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc]:
                    visited[nr, nc] = 1
                    queue.append((nr, nc))

    return mask

# Two seed locations. Adjust these to meaningful points on the MRI.
h, w = img.shape
seeds = [
    (h // 2, w // 2),
    (h // 2, w // 3)
]

thresholds = [5, 15, 30]

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes[0, 0].imshow(img, cmap="gray")
axes[0, 0].set_title("Original")
axes[0, 0].axis("off")

for i, threshold in enumerate(thresholds):
    mask = region_grow(img, seeds[0], threshold)
    axes[0, i + 1].imshow(mask, cmap="gray")
    axes[0, i + 1].set_title(f"Seed 1, T={threshold}")
    axes[0, i + 1].axis("off")

axes[1, 0].imshow(img, cmap="gray")
axes[1, 0].set_title("Original")
axes[1, 0].axis("off")

for i, threshold in enumerate(thresholds):
    mask = region_grow(img, seeds[1], threshold)
    axes[1, i + 1].imshow(mask, cmap="gray")
    axes[1, i + 1].set_title(f"Seed 2, T={threshold}")
    axes[1, i + 1].axis("off")

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
plt.show()

print("Thresholds tested:", thresholds)
print("Seeds tested:", seeds)
print("\nAnalysis:")
print("Region growing depends on the seed because growth starts from that pixel.")
print("A different seed can lie in a different tissue/intensity region, causing the")
print("algorithm to grow into a different set of neighbouring pixels.")
print("A larger intensity-difference threshold permits more pixels to join the region.")
