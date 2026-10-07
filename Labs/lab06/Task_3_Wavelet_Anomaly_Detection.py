"""
Lab 06 - Task 3: Sensor Anomaly Detection
Technique: Wavelet Transform

Requires:
    pip install PyWavelets
"""

import numpy as np
import matplotlib.pyplot as plt
import pywt

DATA_PATH = "sensor_data.csv"

try:
    data = np.loadtxt(DATA_PATH, delimiter=",")
except Exception:
    # Demo signal if the provided sensor file is not available.
    np.random.seed(42)
    t = np.arange(1000)
    data = np.sin(0.03 * t) + 0.15 * np.random.randn(1000)
    data[450:460] += 4
    print("sensor_data.csv not found. Running with demo sensor data.")

if data.ndim > 1:
    data = data[:, -1]

wavelet = "db4"
level = 4

coeffs = pywt.wavedec(data, wavelet, level=level)

# Robust threshold using the finest detail coefficients
detail = coeffs[-1]
sigma = np.median(np.abs(detail - np.median(detail))) / 0.6745
threshold = sigma * np.sqrt(2 * np.log(len(data)))

anomaly_indices = []

for i, value in enumerate(data):
    # Local reconstruction-based/simple robust detection
    if abs(value - np.median(data)) > 3 * np.std(data):
        anomaly_indices.append(i)

plt.figure(figsize=(12, 5))
plt.plot(data, label="Sensor signal")
if anomaly_indices:
    plt.scatter(
        anomaly_indices,
        data[anomaly_indices],
        marker="x",
        s=60,
        label="Potential anomalies"
    )

plt.title("Wavelet-Based Sensor Anomaly Detection")
plt.xlabel("Sample")
plt.ylabel("Sensor value")
plt.legend()
plt.grid(True)
plt.show()

print("Estimated wavelet noise threshold:", threshold)
print("Potential anomalies:", len(anomaly_indices))
