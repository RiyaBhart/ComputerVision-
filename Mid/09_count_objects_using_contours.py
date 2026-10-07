import cv2
img=cv2.imread("image.jpg"); gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
_,thresh=cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
contours,_=cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
count=sum(1 for c in contours if cv2.contourArea(c)>500)
print("Number of objects:",count)
