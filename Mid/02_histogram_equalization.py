import cv2
import matplotlib.pyplot as plt
img=cv2.imread("image.jpg")
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
hist1=cv2.calcHist([gray],[0],None,[256],[0,256])
equalized=cv2.equalizeHist(gray)
hist2=cv2.calcHist([equalized],[0],None,[256],[0,256])
plt.figure(figsize=(10,6))
plt.subplot(2,2,1); plt.imshow(gray,cmap="gray"); plt.title("Original"); plt.axis("off")
plt.subplot(2,2,2); plt.plot(hist1); plt.title("Original Histogram")
plt.subplot(2,2,3); plt.imshow(equalized,cmap="gray"); plt.title("Equalized"); plt.axis("off")
plt.subplot(2,2,4); plt.plot(hist2); plt.title("Equalized Histogram")
plt.tight_layout(); plt.show()
