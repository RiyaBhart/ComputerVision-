import cv2
import numpy as np
import matplotlib.pyplot as plt
img=cv2.imread("image.jpg"); rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB); pixels=np.float32(rgb.reshape((-1,3)))
criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,100,0.2); K=3
_,labels,centers=cv2.kmeans(pixels,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
centers=np.uint8(centers); segmented=centers[labels.flatten()].reshape(rgb.shape)
plt.figure(figsize=(8,4)); plt.subplot(1,2,1); plt.imshow(rgb); plt.title("Original"); plt.axis("off"); plt.subplot(1,2,2); plt.imshow(segmented); plt.title("K-Means"); plt.axis("off"); plt.show()
