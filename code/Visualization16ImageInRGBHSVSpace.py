# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 12:37:57 2020

@author: lsamsi
"""

# import modules
import os
import matplotlib.pyplot as plt #2.1.2
import numpy as np
import cv2


# set directory 
os.getcwd()
os.chdir(r'D:\thesis\images')


image = cv2.imread('nemo.png') # BGR with numpy.uint8, 0-255 val 

print(image.shape) 
# (382, 235, 3)
# plt.imshow(image)
# plt.show()

# plot in RGB 
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image) #now it is in RGB 
plt.show()

#get a color from image
color = image[381,234]
print(color)
#[203 151 103] # correct color


#%%
# Visualizing image in RGB Color Space

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors

r, g, b = cv2.split(image) 
r.shape
g.shape
#  (298, 198)
fig = plt.figure()
axis = fig.add_subplot(1, 1, 1, projection="3d")

pixel_colors = image.reshape((np.shape(image)[0]*np.shape(image)[1], 3))
norm = colors.Normalize(vmin=-1.,vmax=1.) #quench it into -1, 1 interval
norm.autoscale(pixel_colors)
pixel_colors = norm(pixel_colors).tolist()


axis.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors, marker=".")
axis.set_xlabel("Red") 
axis.set_ylabel("Green") 
axis.set_zlabel("Blue")  

plt.title("All pixel colors in the image in RGB Space")
plt.show()


#%%
# Visualizing image in HSV Color Space

# TODO: plot cylinder - real HSV space 
# TODO: pick one color dot in the plot and see the HSV values 
# TODO: rotate 3D interactively 
# TODO: plot each in 2D 

def floatify(img_uint8): 
    return img_uint8.astype(np.float32) / 255 


image = floatify(image)
# convert numpy.uint8 to numpy.float32 after imread for color models (HSV) in range 0-1 requiring floats

# convert RGB to HSV
hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV) #last rendering of image is in RGB 

#plot a color in image
color = hsv_image[381,234]
print(color)
# [28.799994    0.49261075  0.79607844] correct 

h, s, v = cv2.split(hsv_image)
fig = plt.figure()
axis = fig.add_subplot(1, 1, 1, projection="3d")

h_max = h.flatten().max()
# 359.63416
# hue max 360°
s_max = s.flatten().max()
# 0.9999997
# saturation max 100 - opencv uses 1 instead of 100: see https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html
v_max = v.flatten().max()
# 1.0
 # saturation max 100 

axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
axis.set_xlabel("Hue") 
axis.set_ylabel("Saturation") 
axis.set_zlabel("Value")
axis.view_init(30, 30) # rotate 3D to desired angle 
plt.title("All pixel colors in the image in HSV Space")
plt.show()
# visually separable? based on hue, brightness, or saturation? 

    
    


