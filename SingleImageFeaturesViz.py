import cv2 
import matplotlib.pyplot as plt
import numpy as np

path = 'frame-000377.png'
img = cv2.imread(path, 0)

# Define detectors
ORB = cv2.ORB_create()
SIFT = cv2.SURF_create()

# Get keypoints 
kp = ORB.detect(img, None)

# Get descriptors
kp, descriptors = ORB.compute(img, kp)


img2 = cv2.drawKeypoints(img, kp, None, color = (255,0,0), flags = 0)


# Get keypoints 
kp = SIFT.detect(img, None)

# Get descriptors
kp, descriptors = SIFT.compute(img, kp)


img2 = cv2.drawKeypoints(img, kp, None, color = (255,0,0), flags = 0)

plt.imshow(img2)
plt.show()

