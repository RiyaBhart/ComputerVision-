import cv2
import numpy as np
img=cv2.imread("image.jpg",cv2.IMREAD_GRAYSCALE)
print("Mean:",np.mean(img)); print("Std:",np.std(img)); print("Min:",np.min(img)); print("Max:",np.max(img))
