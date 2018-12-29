import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def region_of_interest(img, vertices):
   
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    #plt.figure()
    #plt.imshow(mask)
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image





image_path = 'input/yello-white-road-line_resize_md.jpg';

img = cv2.imread(image_path)
#print type(img.shape)

if len(img.shape) > 2:
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
else:
    img_grey = img

kernel_size = 5
img_gaus = cv2.GaussianBlur(img_grey, (kernel_size, kernel_size), 0)
img_edge = cv2.Canny(img_gaus, 50, 150)

mask = np.zeros_like(img_edge)

#defining a 3 channel or 1 channel color to fill the mask with depending on the input image
if len(img.shape) > 2:
    channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
    ignore_mask_color = (255,) * channel_count
else:
    ignore_mask_color = 255

imshape = img.shape
vertices = np.array([[(0,imshape[0]),(450,290),(490,290),(imshape[1],imshape[0])]],dtype=np.int32)
cv2.fillPoly(mask, vertices, ignore_mask_color)
masked_edges = cv2.bitwise_and(img_edge, mask)



#show
cv2.imshow('virgin image', img)
cv2.imshow('gaussian blur', img_gaus)
cv2.imshow('edges', img_edge)
cv2.imshow('mask', mask)
plt.imshow(masked_edges)
plt.show()

#exit
k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
elif k == ord('s'): # wait for 's' key to save and exit
    cv2.imwrite('output/image.jpg', img_edge)
    cv2.destroyAllWindows()
