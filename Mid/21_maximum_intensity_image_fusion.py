import cv2
import numpy as np
import matplotlib.pyplot as plt
a=cv2.imread("image1.jpg",cv2.IMREAD_GRAYSCALE); b=cv2.imread("image2.jpg",cv2.IMREAD_GRAYSCALE); b=cv2.resize(b,(a.shape[1],a.shape[0])); fused=np.maximum(a,b)
plt.imshow(fused,cmap="gray"); plt.title("Maximum Intensity Fusion"); plt.axis("off"); plt.show()
