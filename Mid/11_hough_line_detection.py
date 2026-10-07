import cv2
import numpy as np
import matplotlib.pyplot as plt
img=cv2.imread("image.jpg"); gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY); edges=cv2.Canny(gray,50,150)
lines=cv2.HoughLinesP(edges,1,np.pi/180,threshold=50,minLineLength=50,maxLineGap=20); result=img.copy()
if lines is not None:
    for line in lines:
        x1,y1,x2,y2=line[0]; cv2.line(result,(x1,y1),(x2,y2),(0,255,0),2)
plt.imshow(cv2.cvtColor(result,cv2.COLOR_BGR2RGB)); plt.title("Hough Lines"); plt.axis("off"); plt.show()
