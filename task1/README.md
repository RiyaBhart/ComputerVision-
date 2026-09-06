# Task 1: Diagnostic Enhancement of Chest X-Rays

## Overview

This task processes a raw chest X-ray image through a multi-stage enhancement pipeline to maximize structural visibility for diagnostic purposes.

## Processing Pipeline

| Step | Operation | Purpose |
|------|-----------|---------|
| 1 | **Load & Display** | Load the grayscale X-ray and inspect pixel range |
| 2 | **Histogram Equalization** | Redistribute pixel intensities to reveal hidden structures |
| 3 | **Color Mapping (JET)** | Map intensity to a color spectrum for fluid boundary detection |
| 4 | **Color Balance** | Gray World white-balance to neutralize artificial color casts |
| 5 | **Thresholding** | Isolate the densest tissues (bone, fluid) via binary threshold |
| 6 | **Log Transform** | Expand dark background regions to reveal faint ribcage edges |
| 7 | **Gamma Correction** | γ=0.5 power-law to brighten midtones and soften bone contrast |

## How to Run

```bash
cd task1
jupyter notebook xray_enhancement.ipynb
```

Or run as a Python script:
```bash
python xray_enhancement.py
```

## Output

All enhanced images are saved to the `output/` directory.
