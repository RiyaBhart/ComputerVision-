# Task 3: Real-Time Echocardiogram Video Analysis

## Overview

This task builds a real-time OpenCV pipeline that processes an echocardiogram (ultrasound) `.mp4` video frame-by-frame, applies mathematical enhancements, and displays the corrected feed alongside the raw feed.

## Processing Pipeline (Per Frame)

| Step | Operation | Purpose |
|------|-----------|---------|
| 1 | **Histogram Equalization** | Combat murky ultrasound contrast |
| 2 | **Logarithmic Transform** | Reveal dark heart chamber regions |
| 3 | **Gamma Correction (γ=0.6)** | Suppress backscatter noise |
| 4 | **COLORMAP_JET Heatmap** | Highlight blood flow intensities |
| 5 | **Color Balance** | Gray World correction on colored output |

## How to Run the Video Loop

```bash
cd task3
python realtime_echo.py
```

Or open the notebook:
```bash
jupyter notebook realtime_echo.ipynb
```

A window will appear showing the **raw** ultrasound stream on the left and the **fully enhanced** stream on the right.

### Controls
- Press **q** to exit the video loop

## Output

- `output/sample_side_by_side.png` — First frame comparison
- `output/sample_mid_frame.png` — Mid-video frame comparison
