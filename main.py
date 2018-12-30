import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image_path = 'input/solidWhiteRight.jpg';
img = mpimg.imread(image_path)
#print type(img.shape)
#img_grey = img
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

rho = 2             #resolution of 'r' in pixels
theta = np.pi/180   #degrees    
threshold = 15      #minimum number of intersections to detect a line
min_line_length = 40
max_line_gap = 20
line_image = np.copy(img)*0

lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)

#draw
for line in lines:
    for x1,y1,x2,y2 in line:
        cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)

color_edges = np.dstack((img_edge,img_edge,img_edge))
lines_edges = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0)
plt.imshow(lines_edges)
plt.show()

#show
#cv2.imshow('virgin image', img)
#cv2.imshow('gray image', img_grey)
#cv2.imshow('gaussian blur', img_gaus)
#cv2.imshow('edges', img_edge)
#cv2.imshow('mask', mask)
plt.imshow(masked_edges)
plt.show()
