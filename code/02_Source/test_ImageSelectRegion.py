# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 13:26:24 2020

@author: Anonym


Selecting a Region in an Image (OpenCV)

"""
# import modules
import os
import matplotlib.pyplot as plt #2.1.2
import numpy as np
import cv2

# to specify
PATH = r'D:\Color\Aishwarya Rai'

# set directory 
os.getcwd()
os.chdir(PATH)


image = cv2.imread('BasicPic1.jpeg') # BGR with numpy.uint8, 0-255 val 
print(image.shape) 
# (382, 235, 3)
# plt.imshow(image)
# plt.show()

# it looks like the blue and red channels have been mixed up. 
# In fact, OpenCV by default reads images in BGR format.
# OpenCV stores RGB values inverting R and B channels, i.e. BGR, thus BGR to RGB: 

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image) #now it is in RGB 
plt.show()



drawing = False # true if mouse is pressed
pt1_x , pt1_y = None , None

# mouse callback function
def line_drawing(event,x,y,flags,param):
    global pt1_x,pt1_y,drawing

    if event==cv2.EVENT_LBUTTONDOWN:
        drawing=True
        pt1_x,pt1_y=x,y

    elif event==cv2.EVENT_MOUSEMOVE:
        if drawing==True:
            cv2.line(image,(pt1_x,pt1_y),(x,y),color=(255,255,255),thickness=2)
            pt1_x,pt1_y=x,y
    elif event==cv2.EVENT_LBUTTONUP:
        drawing=False
        cv2.line(image,(pt1_x,pt1_y),(x,y),color=(255,255,255),thickness=2)        

# cv2.fillConvexPoly(img, points, color[, lineType[, shift]]) 


cv2.namedWindow('test draw')
mask = cv2.setMouseCallback('test draw',line_drawing)

while(1):
    cv2.imshow('test draw',image)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cv2.destroyAllWindows()

#%%


cv2.imshow(mask)