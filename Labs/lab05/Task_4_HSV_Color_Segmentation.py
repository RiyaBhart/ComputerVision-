import cv2
import numpy as np
import matplotlib.pyplot as plt

# Lab 05 - Task 4
# Sorting Objects by Color using HSV segmentation

IMAGE_PATH = "yellow_car.jpg"
OUTPUT_PATH = "task4_hsv_segmentation.png"

img = cv2.imread(IMAGE_PATH)
if img is None:
    raise FileNotFoundError("Put the color image at: " + IMAGE_PATH)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Example for a yellow object.
# Tune these values for the actual image.
restrictive_lower = np.array([25, 180, 150])
restrictive_upper = np.array([32, 255, 255])

final_lower = np.array([18, 80, 80])
final_upper = np.array([40, 255, 255])

restrictive_mask = cv2.inRange(hsv, restrictive_lower, restrictive_upper)
final_mask = cv2.inRange(hsv, final_lower, final_upper)

selected = cv2.bitwise_and(img, img, mask=final_mask)

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0].set_title("Original")
axes[0].axis("off")

axes[1].imshow(restrictive_mask, cmap="gray")
axes[1].set_title("Restrictive Mask")
axes[1].axis("off")

axes[2].imshow(final_mask, cmap="gray")
axes[2].set_title("Final HSV Mask")
axes[2].axis("off")

axes[3].imshow(cv2.cvtColor(selected, cv2.COLOR_BGR2RGB))
axes[3].set_title("Selected Color")
axes[3].axis("off")

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
plt.show()

print("Restrictive HSV range:", restrictive_lower, restrictive_upper)
print("Final HSV range:", final_lower, final_upper)
print("\nAnalysis:")
print("The restrictive mask fails when the selected hue/saturation/value range is too narrow.")
print("Different parts of the same object can have different brightness and saturation,")
print("so a narrow range can remove valid pixels. The final range is wider to capture")
print("the object more completely while still excluding most background pixels.")
