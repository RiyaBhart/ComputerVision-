import cv2
import numpy as np
import matplotlib.pyplot as plt
img=cv2.imread("image.jpg"); gray=cv2.GaussianBlur(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),(5,5),0)
circles=cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,dp=1,minDist=30,param1=100,param2=30,minRadius=10,maxRadius=100); result=img.copy()
if circles is not None:
    for x,y,r in np.uint16(np.around(circles))[0]:
        cv2.circle(result,(x,y),r,(0,255,0),2); cv2.circle(result,(x,y),2,(0,0,255),3)
plt.imshow(cv2.cvtColor(result,cv2.COLOR_BGR2RGB)); plt.title("Hough Circles"); plt.axis("off"); plt.show()
