import cv2
import numpy as np
import matplotlib.pyplot as plt

# Lab 05 - Task 5
# Detecting the Boundary of a Manufactured Part

IMAGE_PATH = "object.jpg"
OUTPUT_PATH = "task5_canny_comparison.png"

img = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("Put the object image at: " + IMAGE_PATH)

blur = cv2.GaussianBlur(img, (5, 5), 0)

# Three low/high threshold pairs.
pairs = [
    (30, 100),
    (50, 150),
    (80, 200)
]

edges = []
for low, high in pairs:
    edges.append(cv2.Canny(blur, low, high))

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
axes[0].imshow(img, cmap="gray")
axes[0].set_title("Original")
axes[0].axis("off")

for ax, edge, (low, high) in zip(axes[1:], edges, pairs):
    ax.imshow(edge, cmap="gray")
    ax.set_title(f"Canny {low}/{high}")
    ax.axis("off")

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
plt.show()

print("Canny threshold pairs:", pairs)
print("\nAnalysis:")
print("- Low thresholds detect more weak edges but can also introduce unwanted noise.")
print("- Higher thresholds are more selective and emphasize strong boundaries.")
print("- If the high threshold is increased while the low threshold stays fixed,")
print("  weak edges are more likely to disappear while strong edges remain.")
print("- Canny uses two-threshold hysteresis: strong edges are accepted and weak")
print("  edges are retained when connected to strong edges.")
