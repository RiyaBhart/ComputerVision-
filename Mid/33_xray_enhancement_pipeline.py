import cv2
import numpy as np
import matplotlib.pyplot as plt
img=cv2.imread("xray.jpg"); gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY); hist=cv2.calcHist([gray],[0],None,[256],[0,256]); equalized=cv2.equalizeHist(gray); g=gray.astype(np.float32); c=255/np.log(1+np.max(g)); log_img=np.uint8(c*np.log(1+g)); adaptive=cv2.adaptiveThreshold(equalized,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
plt.figure(figsize=(12,8));
for i,(x,t) in enumerate([(gray,"X-Ray"),(hist,"Histogram"),(equalized,"Equalized"),(log_img,"Log"),(adaptive,"Adaptive")],1):
    plt.subplot(2,3,i); plt.plot(x) if i==2 else plt.imshow(x,cmap="gray"); plt.title(t); plt.axis("off") if i!=2 else None
plt.tight_layout(); plt.show()
