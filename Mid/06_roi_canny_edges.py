import cv2
import numpy as np
import matplotlib.pyplot as plt
img=cv2.imread("image.jpg"); gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
mask=np.zeros(gray.shape,dtype=np.uint8); pts=np.array([[100,100],[500,100],[500,400],[100,400]],np.int32)
cv2.fillPoly(mask,[pts],255); roi=cv2.bitwise_and(gray,gray,mask=mask); edges=cv2.Canny(roi,50,150)
plt.imshow(edges,cmap="gray"); plt.title("Canny inside ROI"); plt.axis("off"); plt.show()
