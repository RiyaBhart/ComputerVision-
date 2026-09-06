# Medical Imaging Portfolio — Lab 02

A collection of three medical image processing tasks demonstrating diagnostic enhancement, multi-modal fusion, and real-time video analysis using OpenCV and NumPy.

## Repository Structure

```
├── task1/                         # Chest X-Ray Enhancement
│   ├── data/sample_xray.png
│   ├── output/
│   ├── xray_enhancement.ipynb
│   └── README.md
│
├── task2/                         # CT & MRI Cardiac Fusion
│   ├── data/sample_ct.png
│   ├── data/sample_mri.png
│   ├── output/
│   ├── modal_fusion.ipynb
│   └── README.md
│
├── task3/                         # Echocardiogram Video Analysis
│   ├── data/sample_echo.mp4
│   ├── output/
│   ├── realtime_echo.ipynb
│   └── README.md
│
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Running

Open the `.ipynb` notebook in each task folder with Jupyter Notebook or VS Code and run all cells. For Task 3, a window will appear showing the raw and enhanced video side-by-side — press **q** to exit.
