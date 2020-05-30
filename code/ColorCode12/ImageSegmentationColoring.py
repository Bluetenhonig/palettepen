# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 19:42:03 2020

@author: lsamsi

=====================
Image Segementation and Coloring 
=====================

Segment Image by building a mask from the image. 
Color mask with colors from a color palette. 
"""

# import modules
import os
import matplotlib.pyplot as plt #2.1.2
import numpy as np
import cv2

# color picker: https://www.ginifab.com/feeds/pms/color_picker_from_image.php
# https://htmlcolorcodes.com/

# set directory 
os.getcwd()
os.chdir(r'D:\thesis\images\stars')

# set parameters
PERSON_NAME = 'zode'
EXTENSION = '.jpg'
CP = 'bs1'


#%%

### IMAGE ###
# load image in BGR 
image = cv2.imread(f'{PERSON_NAME}.png') # BGR with numpy.uint8, 0-255 val 
#print(image.shape) 
# (382, 235, 3)
# plt.imshow(image)
# plt.show()

# OpenCV stores RGB values inverting R and B channels, i.e. BGR, thus 
# BGR to RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image) #now it is in RGB 
plt.axis('off')
plt.show()


#%%

# RGB to HSV
hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV) #last rendering of image is in RGB 
# plt.imshow(hsv_image) #now it is in HSV 
# plt.show()


# Image Segmentation
# Image Segmentation with mask based on boundaries of HSV values   
# upper and lower color bound in 8-bit image HSV 
# Hue Saturation Value - tweak the values by hand and see whether the mask gets better
# light_orange = (1, 190, 200)
# dark_orange = (18, 255, 255)

# TODO: Schieberegler with real-time viewing of mask changes in window

if PERSON_NAME == 'zode': 
    lowerb = (20, 10, 120) # all values smaller than hsv
    upperb = (255, 255, 255) # all values bigger, yields only positive difference
#  create threshold mask 
# inRange() takes three parameters: the image, the lower range, and the higher range. 
# It returns a binary mask (an ndarray of 1s and 0s) the size of the image where values of 1 indicate values within the range, and zero values indicate values outside:
# build the mask 
mask = cv2.inRange(hsv_image, lowerb, upperb) # inRange(first input array src, lowerb, upperb, [dst])  
# mask on top of the original image
result = cv2.bitwise_and(image, image, mask=mask)
# print(np.mean(result))
# plot mask and the original image with the mask on top:
plt.subplot(1, 2, 1)
plt.imshow(mask, cmap="gray")
plt.subplot(1, 2, 2)
plt.imshow(result)
plt.show()



# Image Segmentation with contour finding (see sckit-image.org/docs) edge-based 
# TODO  scikit-image.org/docs/dev/user_guide/tutorial_segmentation.html 

#%%
### COLOR ###
# Image Recolorization 
# Recolor image to another color where we found a specific color with a mask

# One Color-Image Only  
os.chdir(r'D:\thesis\code\pd4cpInrgb')
txt_cp = 'CP: TW'
txt_rgb = 'RGB: (1,2,3)'
txt_hsv = 'HSV: (1,2,3)'
rose = (235, 140, 182)
image[mask>0]=rose
fig = plt.figure(figsize=(10,4))
plt.imshow(image)
plt.axis('off')
plt.title('Zooey Deschanel')
fig.text(0.42, .09, txt_cp)
fig.text(0.42, .05, txt_rgb)
fig.text(0.42, .01, txt_hsv)
fig.savefig(f'{CP}/{PERSON_NAME}.png')
plt.show()

# type(rose)
# tuple

#%%
# Many Colors-Images 
import pandas as pd 
# load rgb colors 
os.chdir(r'D:\thesis\code\pd4cpInrgb')
rgbs = pd.read_csv(f"{CP}.csv") 

# load hsv colors
os.chdir(r'D:\thesis\code\pd4cpInhsv')
hsvs = pd.read_csv(f"{CP}.csv")

rgb = rgbs['rgb']
hsv = hsvs['hsv']

#%%

# get tuple data structure from rgb as str
tups = []
for i in range(len(rgb)): 
    test_str = rgb[i][1:-1]
    res = tuple(map(int, test_str.split(', ')))
    tups.append(res)

# get list data structure from hsv as str
lups = []   
for i in range(len(hsv)): 
    lups.append(hsv[i])
 
#%%
    
# save image 
os.chdir(r'D:\thesis\code\pd4cpInrgb')
# TODO: in 1 step 
try: 
    os.mkdir(f'{CP}')  #mkdir only one dir at a time 
    os.mkdir(f'{PERSON_NAME}')
    os.chdir(f'D:/thesis/code/pd4cpInrgb/{CP}/{PERSON_NAME}')
except: 
    pass

#%%
# iterate all rgb colors over image and save
for i in range(len(tups)): 
    image[mask>0]= tups[i]
    rgb = tups[i]
    hsv = lups[i]
    index = tuple(rgbs[['row', 'column']].iloc[i,])
    txt_cp = f'CP: TW {index}'
    txt_rgb = f'RGB: {rgb}'
    txt_hsv = f'HSV: {hsv}'
    fig = plt.figure(figsize=(10,4))
    plt.imshow(image)
    plt.axis('off')
    plt.title('Zooey Deschanel')
    fig.text(0.42, .09, txt_cp)
    fig.text(0.42, .05, txt_rgb)
    fig.text(0.42, .01, txt_hsv)
    #fig.savefig(f'{CP}/{PERSON_NAME}/{i}.png')
    plt.show()
    
#%%

# divide images into basic colors and name the color as legend in image

# to specify
COLOR_CATEGORY = 'violet'

# manually-adjusted color classification chart for 8 color categories
hue_range = {'red': {'pure': 355, 'range': [341,10]}
            , 'orange': {'pure': 25, 'range': [11,40]}
            , 'yellow': {'pure': 50, 'range': [41,60]}
            , 'green': {'pure': 115, 'range': [61,170]}
            , 'cyan': {'pure': 185, 'range': [171,200]}
            , 'blue': {'pure': 230, 'range': [201,250]}
            , 'violet': {'pure': 275, 'range': [251,290]}
            , 'magenta': {'pure': 315, 'range': [291,340]}
            }

# restructure 
dico = {'rgb': tups, 'hsv': [eval(i) for i in hsvs['hsv']]} 
data = pd.DataFrame(dico)


boundaries = []
for key, value in hue_range.items():
    if key != 'red': 
        boundaries.append(hue_range[key]['range'])


#check value in which boundary
color_cat = []
for i in range(len(data['hsv'])):
    hue = data['hsv'][i][0]
    if (hue >= hue_range['red']['range'][0]) or (hue <= hue_range['red']['range'][1]):
        color_cat.append('red')
    else: 
        bracket = [[low, high] for [low, high] in boundaries  if low <= hue <= high]
        for key, value in hue_range.items():  # for name, age in dictionary.iteritems():  (for Python 2.x)
            if value['range'] == bracket[0]:
                color_cat.append(key)
# new columns                
data['colorname'] = color_cat

# sort df 
data = data.sort_values(by=['colorname']).reset_index(drop=True)
# filter df
#data = data[data['colorname']==COLOR_CATEGORY]

#%%

# iterate all rgb colors over image and save
for i in range(len(data)): 
    image[mask>0]= data['rgb'].tolist()[i]
    rgb = data['rgb'].tolist()[i]
    hsv = data['hsv'].tolist()[i]
    colorname = data['colorname'].tolist()[i]
    index = tuple(rgbs[['row', 'column']].iloc[i,])
    txt_cp = f'CP: {CP[0:2].upper()} {index}'
    txt_rgb = f'RGB: {rgb}'
    txt_hsv = f'HSV: {hsv}'
    txt_col = f'Color: {colorname}'
    fig = plt.figure(figsize=(10,6))
    plt.imshow(image)
    plt.axis('off')
    plt.title('Zooey Deschanel')
    fig.text(0.38, .10, txt_cp)
    fig.text(0.38, .07, txt_rgb)
    fig.text(0.38, .04, txt_hsv)
    fig.text(0.38, .01, txt_col)
    #fig.savefig(f'{CP}/{PERSON_NAME}/{i}.png')
    plt.show()