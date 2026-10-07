import cv2
import numpy as np
import matplotlib.pyplot as plt
img=cv2.imread("image.jpg"); gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
mask=np.zeros(gray.shape,dtype=np.uint8)
points=np.array([[100,100],[500,100],[500,400],[100,400]],np.int32)
cv2.fillPoly(mask,[points],255)
result=cv2.bitwise_and(gray,gray,mask=mask)
plt.figure(figsize=(10,4))
plt.subplot(1,3,1); plt.imshow(gray,cmap="gray"); plt.title("Original"); plt.axis("off")
plt.subplot(1,3,2); plt.imshow(mask,cmap="gray"); plt.title("Mask"); plt.axis("off")
plt.subplot(1,3,3); plt.imshow(result,cmap="gray"); plt.title("ROI"); plt.axis("off")
plt.tight_layout(); plt.show()
