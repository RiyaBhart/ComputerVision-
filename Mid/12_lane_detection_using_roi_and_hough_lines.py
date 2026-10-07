import cv2
import numpy as np
import matplotlib.pyplot as plt
img=cv2.imread("road.jpg"); gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY); blur=cv2.GaussianBlur(gray,(5,5),0); edges=cv2.Canny(blur,50,150)
h,w=edges.shape; mask=np.zeros_like(edges); pts=np.array([[0,h],[w,h],[int(w*.55),int(h*.60)],[int(w*.45),int(h*.60)]],np.int32); cv2.fillPoly(mask,[pts],255)
roi=cv2.bitwise_and(edges,mask); lines=cv2.HoughLinesP(roi,1,np.pi/180,50,minLineLength=50,maxLineGap=20); result=img.copy()
if lines is not None:
    for line in lines:
        x1,y1,x2,y2=line[0]; cv2.line(result,(x1,y1),(x2,y2),(0,255,0),3)
plt.imshow(cv2.cvtColor(result,cv2.COLOR_BGR2RGB)); plt.title("Lane Detection"); plt.axis("off"); plt.show()
