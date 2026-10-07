import cv2
import numpy as np
import matplotlib.pyplot as plt
img=cv2.imread("image.jpg"); h,w=img.shape[:2]
src=np.float32([[100,100],[500,100],[500,400],[100,400]]); dst=np.float32([[80,120],[520,90],[550,430],[70,450]])
H,mask=cv2.findHomography(src,dst,cv2.RANSAC,5.0); warped=cv2.warpPerspective(img,H,(w,h))
plt.imshow(cv2.cvtColor(warped,cv2.COLOR_BGR2RGB)); plt.title("Homography / Perspective Warp"); plt.axis("off"); plt.show()
