# %% [markdown]
# # Task 3: Real-Time Echocardiogram Video Analysis
# 
# This notebook builds a real-time OpenCV pipeline that captures each frame
# of an echocardiogram .mp4 video, applies mathematical enhancements, and
# displays the corrected feed live alongside the raw feed.

# %% Imports
import cv2
import numpy as np
import os

# Create output directory
os.makedirs('output', exist_ok=True)

# %% [markdown]
# ## 1. Video Capture Setup
# Initialize `cv2.VideoCapture()` to read from the provided ultrasound
# `.mp4` file rather than a webcam.

# %% Video Capture Setup
cap = cv2.VideoCapture('data/sample_echo.mp4')

if not cap.isOpened():
    raise FileNotFoundError("Could not open data/sample_echo.mp4")

fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Video Properties:")
print(f"  Resolution : {width} x {height}")
print(f"  FPS        : {fps}")
print(f"  Frames     : {frame_count}")

# %% [markdown]
# ## 2. Enhancement Functions
# Define helper functions for each processing step applied to every frame.

# %% Enhancement Functions
def apply_histogram_equalization(frame_gray):
    """Apply histogram equalization to combat murky ultrasound contrast."""
    return cv2.equalizeHist(frame_gray)


def apply_colormap(frame_gray):
    """Convert to a colored heatmap (COLORMAP_JET) to highlight blood flow."""
    return cv2.applyColorMap(frame_gray, cv2.COLORMAP_JET)


def apply_color_balance(frame_bgr):
    """Apply Gray World color balance adjustment."""
    frame_float = frame_bgr.astype(np.float64)
    channel_means = frame_float.mean(axis=(0, 1))
    overall_mean = channel_means.mean()
    balanced = np.zeros_like(frame_float)
    for i in range(3):
        if channel_means[i] > 0:
            balanced[:, :, i] = frame_float[:, :, i] * (overall_mean / channel_means[i])
    return np.clip(balanced, 0, 255).astype(np.uint8)


def apply_log_transform(frame_gray):
    """Apply logarithmic transformation to reveal dark heart chamber regions."""
    img_float = frame_gray.astype(np.float64)
    max_val = img_float.max()
    if max_val == 0:
        return frame_gray
    c = 255.0 / np.log(1 + max_val)
    log_img = c * np.log(1 + img_float)
    return np.clip(log_img, 0, 255).astype(np.uint8)


def apply_gamma_transform(frame_gray, gamma=0.6):
    """Apply power-law transformation to suppress backscatter noise."""
    normalized = frame_gray.astype(np.float64) / 255.0
    corrected = np.power(normalized, gamma)
    return (corrected * 255).astype(np.uint8)


def process_frame(frame):
    """Apply the full enhancement pipeline to a single frame."""
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Step 1: Histogram Equalization
    equalized = apply_histogram_equalization(gray)

    # Step 2: Logarithmic Transformation (on equalized)
    log_img = apply_log_transform(equalized)

    # Step 3: Power-Law / Gamma Transformation
    gamma_img = apply_gamma_transform(log_img, gamma=0.6)

    # Step 4: Color Heatmap (COLORMAP_JET)
    heatmap = apply_colormap(gamma_img)

    # Step 5: Color Balance
    balanced = apply_color_balance(heatmap)

    return balanced

# %% [markdown]
# ## 3. Real-Time Video Processing Loop
# Start a `while` loop to extract frames continuously. For every individual
# frame, apply the full enhancement pipeline and concatenate the raw and
# enhanced frames side-by-side.
# 
# Press **q** to exit the video loop.

# %% Real-Time Video Processing Loop
sample_saved = False
frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Apply the full enhancement pipeline
    enhanced = process_frame(frame)

    # Resize enhanced to match original frame dimensions (if needed)
    enhanced_resized = cv2.resize(enhanced, (frame.shape[1], frame.shape[0]))

    # Concatenate raw and enhanced side-by-side (Monitoring Array)
    side_by_side = np.hstack([frame, enhanced_resized])

    # Add labels
    cv2.putText(side_by_side, 'RAW', (5, 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    cv2.putText(side_by_side, 'ENHANCED', (frame.shape[1] + 5, 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    # Display
    cv2.imshow('Echocardiogram — Raw vs Enhanced', side_by_side)

    # Save a sample output frame (first frame)
    if not sample_saved:
        cv2.imwrite('output/sample_side_by_side.png', side_by_side)
        sample_saved = True
        print("Saved sample frame to output/sample_side_by_side.png")

    # Save additional sample at middle of video
    if frame_idx == frame_count // 2:
        cv2.imwrite('output/sample_mid_frame.png', side_by_side)
        print("Saved mid-video frame to output/sample_mid_frame.png")

    frame_idx += 1

    # Press 'q' to exit
    if cv2.waitKey(int(1000 / max(fps, 1))) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"\nProcessed {frame_idx} frames total.")
print("Output frames saved in output/ directory.")
