import cv2
import matplotlib.pyplot as plt
img=cv2.imread("image.jpg"); gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
blur=cv2.GaussianBlur(gray,(5,5),0); edges=cv2.Canny(blur,50,150)
plt.figure(figsize=(10,4))
for i,(x,t) in enumerate([(gray,"Original"),(blur,"Gaussian Blur"),(edges,"Canny")],1):
    plt.subplot(1,3,i); plt.imshow(x,cmap="gray"); plt.title(t); plt.axis("off")
plt.tight_layout(); plt.show()
# For salt-and-pepper noise: median=cv2.medianBlur(gray,5)
