import cv2
import numpy as np
import matplotlib.pyplot as plt
img=cv2.imread("binary_image.png",cv2.IMREAD_GRAYSCALE); _,binary=cv2.threshold(img,127,255,cv2.THRESH_BINARY); k=np.ones((5,5),np.uint8)
opening=cv2.morphologyEx(binary,cv2.MORPH_OPEN,k); closing=cv2.morphologyEx(binary,cv2.MORPH_CLOSE,k)
plt.figure(figsize=(12,4));
for i,(x,t) in enumerate([(binary,"Binary"),(opening,"Opening"),(closing,"Closing")],1): plt.subplot(1,3,i); plt.imshow(x,cmap="gray"); plt.title(t); plt.axis("off")
plt.tight_layout(); plt.show()
