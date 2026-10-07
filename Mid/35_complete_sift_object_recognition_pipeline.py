import cv2
import numpy as np
import matplotlib.pyplot as plt
template=cv2.imread("template.jpg"); scene=cv2.imread("scene.jpg"); sift=cv2.SIFT_create(); kp1,d1=sift.detectAndCompute(cv2.cvtColor(template,cv2.COLOR_BGR2GRAY),None); kp2,d2=sift.detectAndCompute(cv2.cvtColor(scene,cv2.COLOR_BGR2GRAY),None); matches=cv2.BFMatcher().knnMatch(d1,d2,k=2); good=[m for m,n in matches if m.distance<.75*n.distance]
if len(good)>=4:
    src=np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2); dst=np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2); H,_=cv2.findHomography(src,dst,cv2.RANSAC,5.0); h,w=template.shape[:2]; corners=np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2); tc=cv2.perspectiveTransform(corners,H); result=scene.copy(); cv2.polylines(result,[np.int32(tc)],True,(0,255,0),3); plt.imshow(cv2.cvtColor(result,cv2.COLOR_BGR2RGB)); plt.title("SIFT + Homography"); plt.axis("off"); plt.show()
else: print("Not enough good matches")
