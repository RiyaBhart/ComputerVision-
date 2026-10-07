import cv2
import numpy as np
import matplotlib.pyplot as plt
img1=cv2.imread("left.jpg"); img2=cv2.imread("right.jpg"); sift=cv2.SIFT_create(); kp1,d1=sift.detectAndCompute(cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY),None); kp2,d2=sift.detectAndCompute(cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY),None)
matches=cv2.BFMatcher().knnMatch(d2,d1,k=2); good=[m for m,n in matches if m.distance<.75*n.distance]
if len(good)>=4:
    src=np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1,1,2); dst=np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1,1,2); H,_=cv2.findHomography(src,dst,cv2.RANSAC,5.0); h,w=img1.shape[:2]; h2,w2=img2.shape[:2]; pano=cv2.warpPerspective(img2,H,(w+w2,max(h,h2))); pano[0:h,0:w]=img1; plt.imshow(cv2.cvtColor(pano,cv2.COLOR_BGR2RGB)); plt.title("Panorama"); plt.axis("off"); plt.show()
else: print("Not enough good matches")
