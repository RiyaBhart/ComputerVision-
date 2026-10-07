import cv2
import numpy as np
import pywt
import matplotlib.pyplot as plt
img=cv2.imread("image.jpg",cv2.IMREAD_GRAYSCALE); LL,(LH,HL,HH)=pywt.dwt2(img,"haar"); detail=np.abs(LH)+np.abs(HL)+np.abs(HH); threshold=np.mean(detail)+2*np.std(detail); anomaly=detail>threshold
plt.figure(figsize=(10,4)); plt.subplot(1,2,1); plt.imshow(img,cmap="gray"); plt.title("Original"); plt.axis("off"); plt.subplot(1,2,2); plt.imshow(anomaly,cmap="gray"); plt.title("Anomaly"); plt.axis("off"); plt.show()
# LL=low frequency; LH,HL,HH=detail components
