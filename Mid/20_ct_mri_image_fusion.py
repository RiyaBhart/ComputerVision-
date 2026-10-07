import cv2
import matplotlib.pyplot as plt
ct=cv2.imread("ct.jpg",cv2.IMREAD_GRAYSCALE); mri=cv2.imread("mri.jpg",cv2.IMREAD_GRAYSCALE)
mri=cv2.resize(mri,(ct.shape[1],ct.shape[0])); fused=cv2.addWeighted(ct,.5,mri,.5,0)
plt.figure(figsize=(12,4));
for i,(x,t) in enumerate([(ct,"CT"),(mri,"MRI"),(fused,"Fused")],1): plt.subplot(1,3,i); plt.imshow(x,cmap="gray"); plt.title(t); plt.axis("off")
plt.tight_layout(); plt.show()
# 70/30: fused=cv2.addWeighted(ct,.7,mri,.3,0)
