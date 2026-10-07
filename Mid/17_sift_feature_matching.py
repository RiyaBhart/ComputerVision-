import cv2
import matplotlib.pyplot as plt
img1=cv2.imread("image1.jpg"); img2=cv2.imread("image2.jpg")
g1=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY); g2=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY); sift=cv2.SIFT_create()
kp1,d1=sift.detectAndCompute(g1,None); kp2,d2=sift.detectAndCompute(g2,None); matches=cv2.BFMatcher().knnMatch(d1,d2,k=2)
good=[m for m,n in matches if m.distance<.75*n.distance]
result=cv2.drawMatches(img1,kp1,img2,kp2,good,None,flags=2); plt.imshow(cv2.cvtColor(result,cv2.COLOR_BGR2RGB)); plt.title("SIFT Matching"); plt.axis("off"); plt.show(); print("Good matches:",len(good))
