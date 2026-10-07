"""
Lab 06 - Task 5: Panoramic Image Stitching
Technique: SIFT + Homography
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

IMAGE1_PATH = "image1.jpg"
IMAGE2_PATH = "image2.jpg"

img1 = cv2.imread(IMAGE1_PATH)
img2 = cv2.imread(IMAGE2_PATH)

if img1 is None or img2 is None:
    print("Put image1.jpg and image2.jpg in the same folder.")
    raise SystemExit

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()

kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

bf = cv2.BFMatcher()

matches = bf.knnMatch(des2, des1, k=2)

good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append(m)

print("Good matches:", len(good))

if len(good) < 4:
    raise SystemExit("Not enough matches for homography.")

src_pts = np.float32(
    [kp2[m.queryIdx].pt for m in good]
).reshape(-1, 1, 2)

dst_pts = np.float32(
    [kp1[m.trainIdx].pt for m in good]
).reshape(-1, 1, 2)

H, mask = cv2.findHomography(
    src_pts,
    dst_pts,
    cv2.RANSAC,
    5.0
)

if H is None:
    raise SystemExit("Homography could not be calculated.")

h1, w1 = img1.shape[:2]
h2, w2 = img2.shape[:2]

corners_img2 = np.float32([
    [0, 0],
    [w2, 0],
    [w2, h2],
    [0, h2]
]).reshape(-1, 1, 2)

corners_transformed = cv2.perspectiveTransform(
    corners_img2,
    H
)

corners_img1 = np.float32([
    [0, 0],
    [w1, 0],
    [w1, h1],
    [0, h1]
]).reshape(-1, 1, 2)

all_corners = np.concatenate(
    (corners_img1, corners_transformed),
    axis=0
)

[x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
[x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

translation = np.array([
    [1, 0, -x_min],
    [0, 1, -y_min],
    [0, 0, 1]
])

output_width = x_max - x_min
output_height = y_max - y_min

panorama = cv2.warpPerspective(
    img2,
    translation @ H,
    (output_width, output_height)
)

panorama[
    -y_min:h1 - y_min,
    -x_min:w1 - x_min
] = img1

plt.figure(figsize=(16, 7))
plt.imshow(cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB))
plt.title("Panorama")
plt.axis("off")
plt.show()
