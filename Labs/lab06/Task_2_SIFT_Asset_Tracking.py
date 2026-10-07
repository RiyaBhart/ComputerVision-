"""
Lab 06 - Task 2: Asset Tracking
Technique: SIFT feature matching
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

REFERENCE_PATH = "reference.jpg"
SCENE_PATH = "scene.jpg"

reference = cv2.imread(REFERENCE_PATH)
scene = cv2.imread(SCENE_PATH)

if reference is None or scene is None:
    print("Put reference.jpg and scene.jpg in the same folder.")
    raise SystemExit

gray_ref = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
gray_scene = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()

keypoints_ref, descriptors_ref = sift.detectAndCompute(gray_ref, None)
keypoints_scene, descriptors_scene = sift.detectAndCompute(gray_scene, None)

if descriptors_ref is None or descriptors_scene is None:
    print("Could not find enough SIFT features.")
    raise SystemExit

bf = cv2.BFMatcher()
matches = bf.knnMatch(descriptors_ref, descriptors_scene, k=2)

good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

print("Good matches:", len(good_matches))

matched_image = cv2.drawMatches(
    reference,
    keypoints_ref,
    scene,
    keypoints_scene,
    good_matches,
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

plt.figure(figsize=(14, 7))
plt.imshow(cv2.cvtColor(matched_image, cv2.COLOR_BGR2RGB))
plt.title("SIFT Asset Tracking")
plt.axis("off")
plt.show()
