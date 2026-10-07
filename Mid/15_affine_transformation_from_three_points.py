import cv2
import numpy as np
import matplotlib.pyplot as plt
img=cv2.imread("image.jpg"); h,w=img.shape[:2]
src=np.float32([[100,100],[400,100],[100,400]]); dst=np.float32([[150,120],[450,100],[120,450]])
M=cv2.getAffineTransform(src,dst); result=cv2.warpAffine(img,M,(w,h))
plt.imshow(cv2.cvtColor(result,cv2.COLOR_BGR2RGB)); plt.title("3-Point Affine Transform"); plt.axis("off"); plt.show()
