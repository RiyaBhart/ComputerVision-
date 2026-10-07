import cv2
import matplotlib.pyplot as plt
img=cv2.imread("image.jpg"); gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
_,global_thresh=cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
_,otsu=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
adaptive=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
plt.figure(figsize=(10,7))
for i,(x,t) in enumerate([(gray,"Original"),(global_thresh,"Global"),(otsu,"Otsu"),(adaptive,"Adaptive")],1):
    plt.subplot(2,2,i); plt.imshow(x,cmap="gray"); plt.title(t); plt.axis("off")
plt.tight_layout(); plt.show()
