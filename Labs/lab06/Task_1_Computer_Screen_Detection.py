"""
Lab 06 - Task 1: Computer Screen Detection
Technique: Hough Line Transform
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

IMAGE_PATH = "screen.jpg"

image = cv2.imread(IMAGE_PATH)

if image is None:
    print("Image not found. Put your screen image at:", IMAGE_PATH)
    raise SystemExit

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blur, 50, 150)

lines = cv2.HoughLinesP(
    edges,
    rho=1,
    theta=np.pi / 180,
    threshold=100,
    minLineLength=100,
    maxLineGap=20
)

result = image.copy()

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 2)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.title("Detected Screen Lines")
plt.axis("off")

plt.show()
