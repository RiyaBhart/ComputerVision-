"""
Lab 06 - Task 6: Lane Detection
Technique: Canny Edge Detection + Hough Transform
"""

import cv2
import numpy as np

VIDEO_PATH = "road.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Could not open:", VIDEO_PATH)
    raise SystemExit

while True:
    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    height, width = edges.shape

    mask = np.zeros_like(edges)

    roi = np.array([[
        (int(0.1 * width), height),
        (int(0.45 * width), int(0.6 * height)),
        (int(0.55 * width), int(0.6 * height)),
        (int(0.9 * width), height)
    ]], dtype=np.int32)

    cv2.fillPoly(mask, roi, 255)
    cropped_edges = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(
        cropped_edges,
        1,
        np.pi / 180,
        threshold=50,
        minLineLength=40,
        maxLineGap=100
    )

    line_image = np.zeros_like(frame)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            if x2 == x1:
                continue

            slope = (y2 - y1) / (x2 - x1)

            # Keep mostly diagonal lane lines
            if abs(slope) > 0.4:
                cv2.line(
                    line_image,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    5
                )

    result = cv2.addWeighted(frame, 0.8, line_image, 1.0, 0)

    cv2.imshow("Lane Detection", result)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
