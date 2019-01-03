import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image_path = 'input/solidWhiteRight.jpg';

def process(image):
    img = mpimg.imread(image_path)
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
    
    rho = 2                         #resolution of 'r' in pixels
    theta = np.pi/180               #degrees    
    threshold = 150                 #minimum number of intersections to detect a line
    min_line_length = 110           #minimum length of line to detect it
    max_line_gap = 80               #merge broken white lines
    line_image = np.copy(img)*0     
    
    lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap) 
    
    #draw
    #if lines is None:
    for line in lines:
        for x1,y1,x2,y2 in line:
            thickness = 6
            color = (250,0,0)
            cv2.line(line_image,(x1,y1),(x2,y2),color,thickness)
    
    color_edges = np.dstack((img_edge,img_edge,img_edge))
    lines_edges = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0)
    
    #show
    #cv2.imshow('virgin image', img)
    #cv2.imshow('gray image', img_grey)
    #cv2.imshow('gaussian blur', img_gaus)
    #cv2.imshow('edges', img_edge)
    #cv2.imshow('mask', mask)
    #plt.imshow(masked_edges)
    plt.imshow(lines_edges)
    plt.show()






process(image_path)
