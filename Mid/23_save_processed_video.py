import cv2
cap=cv2.VideoCapture("video.mp4")
w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); fps=cap.get(cv2.CAP_PROP_FPS)
out=cv2.VideoWriter("processed_video.mp4",cv2.VideoWriter_fourcc(*"mp4v"),fps,(w,h),False)
while True:
    ret,frame=cap.read()
    if not ret: break
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY); edges=cv2.Canny(gray,50,150); out.write(edges); cv2.imshow("Processed",edges)
    if cv2.waitKey(30)&0xFF==ord("q"): break
cap.release(); out.release(); cv2.destroyAllWindows()
