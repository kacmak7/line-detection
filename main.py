import os
import cv2
import numpy

image_path = 'input/Untitled-design.jpg';

img = cv2.imread(image_path, 0)
#print type(img.shape)

if len(img.shape) > 2:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#show
cv2.imshow('image', img)
cv2.waitKey(0)
cd.destroyAllWindows()


