import cv2
import numpy as np
img=cv2.imread("image.jpg",cv2.IMREAD_GRAYSCALE); roi=img[100:400,100:500]
print("ROI mean:",np.mean(roi)); print("ROI std:",np.std(roi)); print("ROI min:",np.min(roi)); print("ROI max:",np.max(roi))
