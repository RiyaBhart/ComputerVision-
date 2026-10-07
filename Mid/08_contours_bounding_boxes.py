import cv2
import matplotlib.pyplot as plt
img=cv2.imread("image.jpg"); gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
_,thresh=cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
contours,_=cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
result=img.copy()
for c in contours:
    if cv2.contourArea(c)>500:
        cv2.drawContours(result,[c],-1,(0,255,0),2); x,y,w,h=cv2.boundingRect(c); cv2.rectangle(result,(x,y),(x+w,y+h),(255,0,0),2)
plt.imshow(cv2.cvtColor(result,cv2.COLOR_BGR2RGB)); plt.title("Contours + Boxes"); plt.axis("off"); plt.show()
