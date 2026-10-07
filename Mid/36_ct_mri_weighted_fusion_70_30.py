import cv2
import matplotlib.pyplot as plt
ct=cv2.imread("ct.jpg",cv2.IMREAD_GRAYSCALE); mri=cv2.imread("mri.jpg",cv2.IMREAD_GRAYSCALE); mri=cv2.resize(mri,(ct.shape[1],ct.shape[0])); fused=cv2.addWeighted(ct,.7,mri,.3,0)
plt.imshow(fused,cmap="gray"); plt.title("70% CT + 30% MRI"); plt.axis("off"); plt.show()
