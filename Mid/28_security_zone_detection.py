import cv2
import numpy as np
import matplotlib.pyplot as plt
img=cv2.imread("image.jpg"); pts=np.array([[100,100],[500,100],[500,400],[100,400]],np.int32); cv2.polylines(img,[pts],True,(0,0,255),2)
x,y=300,250; inside=cv2.pointPolygonTest(pts,(x,y),False); print("Inside" if inside>=0 else "Outside")
cv2.circle(img,(x,y),5,(255,0,0),-1); plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)); plt.title("Security Zone"); plt.axis("off"); plt.show()
