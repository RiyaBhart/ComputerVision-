import cv2
cap=cv2.VideoCapture("video.mp4"); ret,prev=cap.read()
if not ret: raise SystemExit("Could not read first frame")
prev=cv2.cvtColor(prev,cv2.COLOR_BGR2GRAY)
while True:
    ret,frame=cap.read()
    if not ret: break
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY); diff=cv2.absdiff(prev,gray); _,th=cv2.threshold(diff,30,255,cv2.THRESH_BINARY); cv2.imshow("Motion",th); prev=gray
    if cv2.waitKey(30)&0xFF==ord("q"): break
cap.release(); cv2.destroyAllWindows()
