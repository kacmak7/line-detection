import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse
import imghdr
#from moviepy.editor import VideoFileClip
#from IPython.display import HTML

#TODO: video processing
def process(input='', output=''):

    def pipeline(img): #img has to be 960x540
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
     
        rho = 2                         #resolution of 'r'
        theta = np.pi/180               
        threshold = 150                 #minimum number of intersections to detect a line
        min_line_length = 110           #minimum length of line to detect it
        max_line_gap = 80               #merge broken white lines
        line_image = np.copy(img)*0     
    
        lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap) 

        #draw
        if lines is not None:
            for line in lines:
                for x1,y1,x2,y2 in line:
                    thickness = 6
                    color = (255,0,0) #blue
                    cv2.line(line_image,(x1,y1),(x2,y2),color,thickness)
    
        color_edges = np.dstack((img_edge,img_edge,img_edge))
        lines_edges = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0)

        #save
        if (output != ''):
            cv2.imwrite(output + '/' + 'image' + '.jpg', lines_edges)

        #show
        #cv2.imshow('virgin image', img)
        #cv2.imshow('gray image', img_grey)
        #cv2.imshow('gaussian blur', img_gaus)
        cv2.imshow('edges', img_edge)
        #cv2.imshow('mask', mask) #region of interest
        #cv2.imshow('masked edges', masked_edges)
        cv2.imshow('lin', lines_edges)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        #TODO: return original image with marked lines
    

   
    # START
    #check if input is an image or video
    video_formats = ('.avi', '.mp4')
    if (imghdr.what(input)):
        img = mpimg.imread(input) # RGB
        pipeline(img)
    
    elif (input.lower().endswith(video_formats)):
        video = cv2.VideoCapture(input)

        while True:
            ret, frame = video.read()
            cv2.imshow(frame)
            
            if not ret:
                video = cv2.VideoCapture(input)

    else:
        print('Bad input resource')
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detect lines on your image')
    parser.add_argument('-i', '--i', help='Specify path of input file', type=str, required=False, dest = 'input')
    parser.add_argument('-o', '--o', help='Specify path of output directory', type=str, required=False, dest = 'output')
    args = parser.parse_args()

    if (args.input):
        input_path = args.input
    else:
        input_path = 'input/solidWhiteRight.jpg';

    if (args.output):
        output_path = args.output
    else:
        output_path = ''

    process(input_path, output_path)
