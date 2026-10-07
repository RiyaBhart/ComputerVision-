import cv2
import numpy as np
import matplotlib.pyplot as plt

# Lab 05 - Task 9
# K-Means Image Segmentation

IMAGE_PATH = "natural_scene.jpg"
OUTPUT_PATH = "task9_kmeans_segmentation.png"

img = cv2.imread(IMAGE_PATH)
if img is None:
    raise FileNotFoundError("Put the color image at: " + IMAGE_PATH)

# K-Means works on pixel feature vectors.
pixels = img.reshape((-1, 3))
pixels = np.float32(pixels)

criteria = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    100,
    0.2
)

k_values = [2, 4, 6]
segmented_images = []

for k in k_values:
    compactness, labels, centers = cv2.kmeans(
        pixels,
        k,
        None,
        criteria,
        10,
        cv2.KMEANS_PP_CENTERS
    )

    centers = np.uint8(centers)
    segmented = centers[labels.flatten()]
    segmented = segmented.reshape(img.shape)

    segmented_images.append(segmented)

    print(f"K={k}")
    print(f"  Compactness: {compactness:.2f}")
    print(f"  Cluster centers:\n{centers}\n")

fig, axes = plt.subplots(1, 4, figsize=(20, 5))

axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0].set_title("Original")
axes[0].axis("off")

for ax, k, segmented in zip(axes[1:], k_values, segmented_images):
    ax.imshow(cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB))
    ax.set_title(f"K = {k}")
    ax.axis("off")

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
plt.show()

print("Analysis:")
print("K=2 gives the strongest color simplification and represents only broad regions.")
print("K=4 preserves more visual structure while still simplifying the image.")
print("K=6 preserves more distinct colors and therefore represents more detailed regions.")
print("Higher K generally reduces color simplification but can better preserve meaningful")
print("color differences in complex natural scenes.")
