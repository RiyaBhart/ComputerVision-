# Task 2: Multi-Modal Cardiac Image Fusion

## Overview

This task fuses a CT slice and its corresponding MRI slice of the heart into a single, unified diagnostic view using weighted blending and post-processing.

## Fusion Weighting Logic

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| CT Weight (α) | **0.7** | Preserves sharp anatomical edges and structural boundaries |
| MRI Weight (β) | **0.3** | Highlights internal soft tissue variations |
| Brightness (γ) | **0** | No artificial brightness offset |

The heavier CT weight ensures that bone boundaries and chamber walls remain crisp, while the MRI contribution adds soft-tissue contrast that CT alone cannot capture.

## Processing Pipeline

| Step | Operation | Purpose |
|------|-----------|---------|
| 1 | **Load Modalities** | Load matched CT and MRI slices |
| 2 | **Histogram Equalization** | Independently maximize dynamic range of both |
| 3 | **Color Mapping & Overlay** | CT→HOT, MRI→BONE colormaps; overlay |
| 4 | **Weighted Fusion** | `cv2.addWeighted(ct, 0.7, mri, 0.3, 0)` |
| 5 | **Log & Gamma Post-Processing** | Prevent black-crush / white-blowout |
| 6 | **Comparative Analysis** | Side-by-side CT vs MRI vs Fused |

## How to Run

```bash
cd task2
jupyter notebook modal_fusion.ipynb
```

Or run as a Python script:
```bash
python modal_fusion.py
```

## Output

All fused heatmaps and comparison charts are saved to the `output/` directory.
