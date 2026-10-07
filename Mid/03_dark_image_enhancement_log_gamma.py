import cv2
import numpy as np
import matplotlib.pyplot as plt
img=cv2.imread("dark_image.jpg")
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
equalized=cv2.equalizeHist(gray)
g=gray.astype(np.float32); c=255/np.log(1+np.max(g)); log_img=np.uint8(c*np.log(1+g))
gamma=0.5; gamma_img=np.uint8(255*((gray/255.0)**gamma))
plt.figure(figsize=(10,7))
for i,(x,t) in enumerate([(gray,"Original"),(equalized,"Equalized"),(log_img,"Log"),(gamma_img,"Gamma")],1):
    plt.subplot(2,2,i); plt.imshow(x,cmap="gray"); plt.title(t); plt.axis("off")
plt.tight_layout(); plt.show()
