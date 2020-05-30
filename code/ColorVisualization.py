# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 19:42:03 2020

@author: lsamsi

Color Patch Visualization

!!! WARNING: Visuazlizations only happen in RGB !!!

"""

# import modules
import os
import matplotlib.pyplot as plt #2.1.2
import numpy as np
import pandas as pd
import cv2

# color picker: https://www.ginifab.com/feeds/pms/color_picker_from_image.php

# to specify
PATH = r'D:\thesis\code'

# change directory 
os.chdir(PATH)

#%%


# convert color 
def convert_color(col, conversion=cv2.COLOR_BGR2Lab): #default 
    color = cv2.cvtColor(np.array([[col]], dtype=np.float32), conversion)[0, 0]
    return color

# display color 
def display_color(color, origin=None):
    """helper function: convert_color """
    # convert color to RGB
    if origin == 'BGR': 
        rgb_color = convert_color(color, cv2.COLOR_BGR2RGB)
    elif origin == 'LAB': 
        rgb_color = convert_color(color, cv2.COLOR_LAB2RGB)*255
    else: 
        rgb_color = color
    square = np.full((10, 10, 3), rgb_color, dtype=np.uint8) / 255.0
     #display RGB colors patch 
    plt.figure(figsize = (5,2))
    plt.imshow(square) 
    plt.axis('off')
    plt.show() 

# convert numpy array of colors
def convert_array(nparray, origin, target='RGB'): 
    """helper function: convert_color """
    # convert to RGB
    rgb_colors = []
    for col in nparray: 
        if origin == 'BGR':        
            rgb_color = convert_color(col, cv2.COLOR_BGR2RGB)*255
        if origin == 'LAB':     
            rgb_color = convert_color(col, cv2.COLOR_LAB2RGB)*255
        if origin == 'HSV':     
            rgb_color = convert_color(col, cv2.COLOR_HSV2RGB)*255
        rgb_colors.append(rgb_color)
    return rgb_colors

# display color palette as bar 
def display_color_grid(palette, origin='RGB', colorbar_count=10):
    """helper function: convert_array, convert_color """
    if origin == 'BGR':
        rgbcolors = convert_array(palette, 'BGR')
    if origin == 'LAB': 
        rgbcolors = convert_array(palette, 'LAB')
    if origin == 'HSV': 
        rgbcolors = convert_array(palette, 'HSV')
    x= 0
    for r in rgbcolors: 
        if len(rgbcolors[x:x+colorbar_count]) == colorbar_count:
            palette = np.array(rgbcolors[x:x+colorbar_count])[np.newaxis, :, :]
            plt.figure(figsize=(colorbar_count*2,5))
            plt.imshow(palette.astype('uint8'))
            #plt.imshow(palette)
            plt.axis('off')
            plt.show()
            x += colorbar_count
        else: 
            if x == len(palette): 
                break
            else: 
                palette = np.array(rgbcolors[x:])[np.newaxis, :, :]
                plt.figure(figsize=(colorbar_count*2,2))
                plt.imshow(palette.astype('uint8'))
                plt.axis('off')
                plt.show()
                break




#%%
#######################
### SINGLE COLORS #####
#######################

import matplotlib.pyplot as plt #2.1.2
import numpy as np
import cv2 

# define BGR color 
bgr_color = [135, 173, 145]
# define a numpy shape 
a = np.full((190, 266, 3), bgr_color, dtype=np.uint8)
print(a.shape) 

# visualize color in cv2
cv2.imshow('A BGR-color', a)

# visualize color in plt
fig = plt.figure(figsize = (4,2))
plt.imshow(a) 
plt.axis('off')


#%%

### RGB ###
    
# define RGB colors 
red = (255,0,0) 
green = (0,255,0)
blue = (0, 0, 255)

# fill square 
square_1 = np.full((10, 10, 3), red, dtype=np.uint8) / 255.0
square_2 = np.full((10, 10, 3), green, dtype=np.uint8) / 255.0
square_3 = np.full((10, 10, 3), blue, dtype=np.uint8) / 255.0

#display RGB colors patches 
fig = plt.figure(figsize = (10,4))
plt.subplot(3, 1, 1)
plt.imshow(square_1) 
plt.axis('off')
plt.subplot(3, 1, 2)
plt.imshow(square_2)
plt.axis('off')
plt.subplot(3, 1, 3)
plt.imshow(square_3)
plt.axis('off')
txt_1 = 'Red: \nRGB: (255, 0, 0) \nHSV: (0°,s,v)'
txt_2 = 'Green: \nRGB: (0, 255, 0)  \nHSV: (120°,s,v)'
txt_3 = 'Blue: \nRGB: (0, 0, 255)  \nHSV: (240°,s,v)'
fig.text(.57, .76, txt_1)
fig.text(.57, .49, txt_2)
fig.text(.57, .23, txt_3)
plt.suptitle('Color Patches in RGB', ha='left')
plt.show()


#%%

# define RGB colors 
yellow = (255,255,0) 
cyan = (0,255,255)
magenta = (255, 0, 255)

# fill square 
square_1 = np.full((10, 10, 3), yellow, dtype=np.uint8) / 255.0
square_2 = np.full((10, 10, 3), cyan, dtype=np.uint8) / 255.0
square_3 = np.full((10, 10, 3), magenta, dtype=np.uint8) / 255.0

#display RGB colors patches 
fig = plt.figure(figsize = (10,4))
plt.subplot(3, 1, 1)
plt.imshow(square_1) 
plt.axis('off')
plt.subplot(3, 1, 2)
plt.imshow(square_2)
plt.axis('off')
plt.subplot(3, 1, 3)
plt.imshow(square_3)
plt.axis('off')
txt_1 = f'Yellow: \nRGB: {yellow} \nHSV: (60°,s,v)'
txt_2 = f'Cyan: \nRGB: {cyan}  \nHSV: (180°,s,v)'
txt_3 = f'Magenta: \nRGB: {magenta}  \nHSV: (300°,s,v)'
fig.text(.57, .76, txt_1)
fig.text(.57, .49, txt_2)
fig.text(.57, .23, txt_3)
plt.suptitle('Color Patches in RGB', ha='left')
plt.show()


#%%
# define RGB colors 
orange = (255,128,0) 
green_blue = (0,255,128)
purple = (128, 0, 255)

# fill square 
square_1 = np.full((10, 10, 3), orange, dtype=np.uint8) / 255.0
square_2 = np.full((10, 10, 3), green_blue, dtype=np.uint8) / 255.0
square_3 = np.full((10, 10, 3), purple, dtype=np.uint8) / 255.0

#display RGB colors patches 
fig = plt.figure(figsize = (10,4))
plt.subplot(3, 1, 1)
plt.imshow(square_1) 
plt.axis('off')
plt.subplot(3, 1, 2)
plt.imshow(square_2)
plt.axis('off')
plt.subplot(3, 1, 3)
plt.imshow(square_3)
plt.axis('off')
txt_1 = f'Orange: \nRGB: {orange} \nHSV: (30°,s,v)'
txt_2 = f'Green-blue: \nRGB: {green_blue}  \nHSV: (150°,s,v)'
txt_3 = f'Purple: \nRGB: {purple}  \nHSV: (270°,s,v)'
fig.text(.57, .76, txt_1)
fig.text(.57, .49, txt_2)
fig.text(.57, .23, txt_3)
plt.suptitle('Color Patches in RGB', ha='left')
plt.show()

#%%
# define RGB colors 
green_yellow = (128,255,0) 
blue_yellow = (0,128, 255)
red_yellow = (255, 0, 128)

# fill square 
square_1 = np.full((10, 10, 3), orange, dtype=np.uint8) / 255.0
square_2 = np.full((10, 10, 3), blue_yellow, dtype=np.uint8) / 255.0
square_3 = np.full((10, 10, 3), red_yellow, dtype=np.uint8) / 255.0

#display RGB colors patches 
fig = plt.figure(figsize = (10,4))
plt.subplot(3, 1, 1)
plt.imshow(square_1) 
plt.axis('off')
plt.subplot(3, 1, 2)
plt.imshow(square_2)
plt.axis('off')
plt.subplot(3, 1, 3)
plt.imshow(square_3)
plt.axis('off')
txt_1 = f'Orange: \nRGB: {orange} \nHSV: (90°,s,v)'
txt_2 = f'Blue-yellow: \nRGB: {blue_yellow}  \nHSV: (210°,s,v)'
txt_3 = f'Red-yellow: \nRGB: {red_yellow}  \nHSV: (330°,s,v)'
fig.text(.57, .76, txt_1)
fig.text(.57, .49, txt_2)
fig.text(.57, .23, txt_3)
plt.suptitle('Color Patches in RGB', ha='left')
plt.show()


#%%
### HSV ###
# display the colors in Python is to make small square images of the desired color and plot them in Matplotlib. 
# Matplotlib only interprets colors in RGB, but handy conversion functions are provided for the major color spaces so that we can plot images in other color spaces:
from matplotlib.colors import hsv_to_rgb

# define HSV colors 
light_orange = (18,255,255) 
dark_orange = (1,190,200)
light_white = (49.999916076660156
, 0.31578928232192993
, 0.2235294133424759)
dark_white = (51.724082946777344
, 0.4603172242641449
, 0.24705882370471954)

# normalize to 8-bit image range 0-1 for viewing 
upper_square = np.full((10, 10, 3), light_white, dtype=np.uint8) / 255.0
lower_square = np.full((10, 10, 3), dark_white, dtype=np.uint8) / 255.0

#display HSV color patches in RGB 
plt.subplot(1, 2, 1)
plt.imshow(hsv_to_rgb(upper_square)) 
plt.title('upper limit')
plt.subplot(1, 2, 2)
plt.imshow(hsv_to_rgb(lower_square))
plt.title('lower limit')
plt.suptitle('Color Patches in HSV')
plt.show()


#%%
#######################
### TWO COLORS #####
#######################

# specify two bgr colors 
bgr_1 = [135, 173, 145]
bgr_2 = [135, 173, 25]

# specify color patch area size for colors to display 
patch = (190, 266, 3)
a = np.full(patch, bgr_1, dtype=np.uint8)
b = np.full(patch, bgr_2, dtype=np.uint8)
c = np.vstack((a, b))
print(c.shape) 

cv2.imshow('Two Colors', c)


#%%

#####################
### COLOR WHEEL #####
#####################

#### 30° STEPS ######

# define RGB colors
os.chdir(r'D:\thesis\code\pd12hues')
rgbhsv = pd.read_csv('rgbhsv_12.csv')

# fill square 
squares = []
for i in range(len(rgbhsv)): 
    square = np.full((10, 10, 3), eval(rgbhsv['RGB'][i]), dtype=np.uint8) / 255.0
    squares.append(square)
    

#display HSV colors patches in RGB
os.chdir(r'D:\thesis\code\12hues')
fig = plt.figure(figsize = (30,5))
for i in range(1,13):
    plt.subplot(1, 12, i)
    plt.imshow(squares[i-1]) 
    plt.axis('off')
    color = rgbhsv['name'][i-1]
    rgb = rgbhsv['RGB'][i-1] 
    hsv = rgbhsv['HSV'][i-1] 
    plt.title(f'{color}', size=20)
    #plt.title(f'{color} \nRGB: {rgb} \nHSV: {hsv}', size=20)
plt.suptitle('Color Patches in LCH in 30°-hue steps around the Color Wheel', size=40)
fig.savefig('hsv_colorwheel_patches.png')
plt.show()


#%%

# https://www.nixsensor.com/free-color-converter/

### L*ab ###
import os
import pandas as pd


os.chdir(r'D:\thesis\code\pd12hues')
lablchrgb_12 = pd.read_csv('lablchrgb_12_handcorrected.csv')

squares= []
for i in range(len(lablchrgb_12['RGB'])):
    # get color and fill 
    color = eval(lablchrgb_12['RGB'][i])
    square = np.full((10, 10, 3), color, dtype=np.uint8)
    squares.append(square)

#display LCH color patches in RGB 
os.chdir(r'D:\thesis\code\12hues')
fig = plt.figure(figsize = (30,5))    
for i in range(1,13):  
    plt.subplot(1, 12, i)
    plt.imshow(squares[i-1]) 
    plt.axis('off')
    color = lablchrgb_12['name'][i-1]
    plt.title(f'{color}', size=20)
plt.suptitle('Color Patches in LCH in 30°-hue steps around the Color Wheel', size=40)
#fig.savefig('lch_colorwheel_patches.png')
plt.show()


#%%

###############################
### BASIC COLOR CATEGORIES ####
###############################

#### 6 colors ######
### red, orange, yellow, green, blue, purple



### RGB ###
import os
import pandas as pd


os.chdir(r'D:\thesis\code\pd6hues')
rgbhsvlch_6 = pd.read_csv('rgbhsvlch_6.csv')

squares= []
for i in range(len(rgbhsvlch_6['RGB'])):
    # get color and fill 
    color = eval(rgbhsvlch_6['RGB'][i])
    square = np.full((10, 10, 3), color, dtype=np.uint8)
    squares.append(square)

#display LCH color patches in RGB 
os.chdir(r'D:\thesis\code\6hues')
fig = plt.figure(figsize = (20,5))    
for i in range(1,7):  
    plt.subplot(1, 6, i)
    plt.imshow(squares[i-1]) 
    plt.axis('off')
    color = rgbhsvlch_6['name'][i-1]
    plt.title(f'{color}', size=30)
plt.suptitle('6 Basic Colors in RGB', size=40)
fig.savefig('lch_6basiccolors_patches.png')
plt.show()

