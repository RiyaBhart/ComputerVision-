import cv2
import numpy as np
points=np.array([[100,100],[500,100],[500,400],[100,400]],np.int32); x,y=300,250
inside=cv2.pointPolygonTest(points,(x,y),False)
print("Object is inside restricted zone" if inside>=0 else "Object is outside restricted zone")
