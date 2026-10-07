"""
Lab 06 - Task 8: Smart Security System
Technique: Background subtraction + contour detection

The system detects movement and displays an alarm when
significant foreground activity is detected.
"""

import cv2

VIDEO_PATH = "security.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Could not open:", VIDEO_PATH)
    raise SystemExit

background_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=50,
    detectShadows=True
)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    fg_mask = background_subtractor.apply(frame)

    _, thresh = cv2.threshold(
        fg_mask,
        200,
        255,
        cv2.THRESH_BINARY
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    thresh = cv2.morphologyEx(
        thresh,
        cv2.MORPH_OPEN,
        kernel
    )

    thresh = cv2.dilate(
        thresh,
        kernel,
        iterations=2
    )

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    alarm = False

    for contour in contours:
        area = cv2.contourArea(contour)

        if area > 1000:
            alarm = True

            x, y, w, h = cv2.boundingRect(contour)

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

    if alarm:
        cv2.putText(
            frame,
            "ALARM: MOTION DETECTED",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )
    else:
        cv2.putText(
            frame,
            "STATUS: SAFE",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    cv2.imshow("Smart Security System", frame)

    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
