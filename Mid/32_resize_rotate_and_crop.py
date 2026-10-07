import cv2
import matplotlib.pyplot as plt
img=cv2.imread("image.jpg"); resized=cv2.resize(img,(500,400)); h,w=resized.shape[:2]; M=cv2.getRotationMatrix2D((w//2,h//2),45,1); rotated=cv2.warpAffine(resized,M,(w,h)); crop=rotated[100:300,100:400]
plt.figure(figsize=(12,4));
for i,(x,t) in enumerate([(resized,"Resized"),(rotated,"Rotated"),(crop,"Cropped")],1): plt.subplot(1,3,i); plt.imshow(cv2.cvtColor(x,cv2.COLOR_BGR2RGB)); plt.title(t); plt.axis("off")
plt.tight_layout(); plt.show()
