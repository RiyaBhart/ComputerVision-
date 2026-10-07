"""
Lab 06 - Task 7: Coin Detection and Counting
Technique: Hough Circle Transform
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

IMAGE_PATH = "coins.jpg"

image = cv2.imread(IMAGE_PATH)

if image is None:
    print("Image not found:", IMAGE_PATH)
    raise SystemExit

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (9, 9), 2)

circles = cv2.HoughCircles(
    gray,
    cv2.HOUGH_GRADIENT,
    dp=1.2,
    minDist=30,
    param1=100,
    param2=30,
    minRadius=10,
    maxRadius=100
)

result = image.copy()

count = 0

if circles is not None:
    circles = np.round(circles[0, :]).astype(int)

    for x, y, r in circles:
        cv2.circle(result, (x, y), r, (0, 255, 0), 2)
        cv2.circle(result, (x, y), 2, (0, 0, 255), 3)
        count += 1

print("Number of coins detected:", count)

plt.figure(figsize=(10, 7))
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.title("Detected Coins")
plt.axis("off")
plt.show()
