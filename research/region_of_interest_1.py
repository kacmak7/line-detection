import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#input = "solidWhiteRight.jpg"
input = "20190312183128_1.jpg"

img = mpimg.imread(input) # RGB

if len(img.shape) > 2:
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
else:
    img_grey = img

# defining a 3 channel or 1 channel color to fill the mask with depending on the input image
if len(img.shape) > 2:
    channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
    ignore_mask_color = (255,) * channel_count
else:
    ignore_mask_color = 255

mask = np.zeros_like(img)
imshape = img.shape
vertices = np.array([[(0,imshape[0]),(900,300),(1120,300),(imshape[1],imshape[0])]],dtype=np.int32)
cv2.fillPoly(mask, vertices, ignore_mask_color)

# write and show
cv2.imwrite('OUTPUT.jpg', mask)
print('saved output')
cv2.imshow('mask', mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
