import os
import cv2
import numpy

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


image_path = 'input/Untitled-design.jpg';

img = cv2.imread(image_path, 0)
#print type(img.shape)

if len(img.shape) > 2:
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
else:
    img_grey = img

img_gaus = gaussian_blur(img, 5)

img_edge = cv2.Canny(img_gaus, 50, 150)

#show
cv2.imshow('virgin image', img)
cv2.imshow('gaussian blur', img_gaus)
cv2.imshow('edges', img_edge)

k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
elif k == ord('s'): # wait for 's' key to save and exit
    cv2.imwrite('output/image.jpg',img)
    cv2.destroyAllWindows()

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
