import cv2
import numpy as np
import matplotlib.pyplot as plt
img=cv2.imread("objects.jpg"); gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY); _,th=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
k=np.ones((3,3),np.uint8); opening=cv2.morphologyEx(th,cv2.MORPH_OPEN,k,iterations=2); sure_bg=cv2.dilate(opening,k,iterations=3); dist=cv2.distanceTransform(opening,cv2.DIST_L2,5); _,fg=cv2.threshold(dist,.5*dist.max(),255,0); fg=np.uint8(fg); unknown=cv2.subtract(sure_bg,fg); _,markers=cv2.connectedComponents(fg); markers=markers+1; markers[unknown==255]=0; markers=cv2.watershed(img,markers); result=img.copy(); result[markers==-1]=[0,0,255]
plt.imshow(cv2.cvtColor(result,cv2.COLOR_BGR2RGB)); plt.title("Watershed"); plt.axis("off"); plt.show()
