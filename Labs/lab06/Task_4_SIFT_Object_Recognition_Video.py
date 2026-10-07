"""
Lab 06 - Task 4: Object Recognition in Video
Technique: SIFT + Homography

Requires a reference object image and a video.
"""

import cv2
import numpy as np

REFERENCE_PATH = "object.jpg"
VIDEO_PATH = "video.mp4"

reference = cv2.imread(REFERENCE_PATH)

if reference is None:
    print("Reference image not found:", REFERENCE_PATH)
    raise SystemExit

gray_reference = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray_reference, None)

if des1 is None:
    print("No SIFT features found in reference image.")
    raise SystemExit

bf = cv2.BFMatcher()

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Could not open:", VIDEO_PATH)
    raise SystemExit

while True:
    ret, frame = cap.read()

    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp2, des2 = sift.detectAndCompute(gray_frame, None)

    if des2 is not None:
        matches = bf.knnMatch(des1, des2, k=2)

        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)

        if len(good) >= 4:
            src_pts = np.float32(
                [kp1[m.queryIdx].pt for m in good]
            ).reshape(-1, 1, 2)

            dst_pts = np.float32(
                [kp2[m.trainIdx].pt for m in good]
            ).reshape(-1, 1, 2)

            H, mask = cv2.findHomography(
                src_pts,
                dst_pts,
                cv2.RANSAC,
                5.0
            )

            if H is not None:
                h, w = gray_reference.shape

                corners = np.float32([
                    [0, 0],
                    [w, 0],
                    [w, h],
                    [0, h]
                ]).reshape(-1, 1, 2)

                projected = cv2.perspectiveTransform(corners, H)

                cv2.polylines(
                    frame,
                    [np.int32(projected)],
                    True,
                    (0, 255, 0),
                    3
                )

                cv2.putText(
                    frame,
                    "Object Detected",
                    (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

    cv2.imshow("SIFT Object Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
