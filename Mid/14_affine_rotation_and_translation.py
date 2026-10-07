import cv2
import matplotlib.pyplot as plt
img=cv2.imread("image.jpg"); h,w=img.shape[:2]; M=cv2.getRotationMatrix2D((w//2,h//2),30,1.2); M[0,2]+=50; M[1,2]+=30
result=cv2.warpAffine(img,M,(w,h)); plt.imshow(cv2.cvtColor(result,cv2.COLOR_BGR2RGB)); plt.title("Affine Rotation + Translation"); plt.axis("off"); plt.show()
