# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 19:42:03 2020

@author: lsamsi
"""

# websites 
# converter/match: http://www.easyrgb.com
# converter: convertingcolors.com/
# converter: colormine.org/convert/lab-to-lch 
# brain-teaser: sensing.konicaminolta.us/blog/identfiying-color-differences-using-l-a-b-or-l-c-h-coordinates
# brain-teaser: zschuessler.github.io/DeltaE/learn 


# import modules
import os
import matplotlib.pyplot as plt #2.1.2
import numpy as np
import pandas as pd
import cv2

# color picker: https://www.ginifab.com/feeds/pms/color_picker_from_image.php

# load all availbe color spaces in opencv
#flags = [i for i in dir(cv2) if i.startswith('COLOR_')]
#flag_len = len(flags)
#print(f"Color spaces available: {flag_len}")
        

#%%
#############
### COLOR ###   
#############
# Color Value (Color Space) Conversion 

bgr2rgb = cv2.COLOR_BGR2RGB
rgb2bgr = cv2.COLOR_RGB2BGR
bgr2lab = cv2.COLOR_BGR2Lab
lab2bgr = cv2.COLOR_Lab2BGR
rgb2lab = cv2.COLOR_RGB2Lab
lab2rgb = cv2.COLOR_Lab2RGB
rgb2hsv = cv2.COLOR_RGB2HSV
hsv2rgb = cv2.COLOR_HSV2RGB

def lab2lch(lab, h_as_degree = True):
    """
    :param lab: np.ndarray (dtype:float32) l : {0, ..., 100}, a : {-128, ..., 128}, l : {-128, ..., 128}
    :return: lch: np.ndarray (dtype:float32), l : {0, ..., 100}, c : {0, ..., 128}, h : {0, ..., 2*PI}
    """
    if not isinstance(lab, np.ndarray):
        lab = np.array(lab, dtype=np.float32)
    lch = np.zeros_like(lab, dtype=np.float32)
    lch[..., 0] = lab[..., 0]
    lch[..., 1] = np.linalg.norm(lab[..., 1:3], axis=len(lab.shape) - 1)
    lch[..., 2] = np.arctan2(lab[..., 2], lab[..., 1])
    lch[..., 2] += np.where(lch[..., 2] < 0., 2 * np.pi, 0)
    if h_as_degree:
        lch[..., 2] = (lch[..., 2] / (2*np.pi)) * 360
    return lch

def lch2lab(lch, h_as_degree = True):
    """
    :param lch: np.ndarray (dtype:float32), l : {0, ..., 100}, c : {0, ..., 128}, h : {0, ..., 2*PI}  
    :return: lab: np.ndarray (dtype:float32) l : {0, ..., 100}, a : {-128, ..., 128}, l : {-128, ..., 128}
    """
    if not isinstance(lch, np.ndarray):
        lch = np.array(lch, dtype=np.float32)
    lab = np.zeros_like(lch, dtype=np.float32)
    lab[..., 0] = lch[..., 0]
    if h_as_degree:
        lch[..., 2] = lch[..., 2] *np.pi / 180
    lab[..., 1] = lch[..., 1]*np.cos(lch[..., 2])
    lab[..., 2] = lch[..., 1]*np.sin(lch[..., 2])
    return lab

def rgb_to_hex(red, green, blue):
    """Return color as #rrggbb for the given color values."""
    return '#%02x%02x%02x' % (red, green, blue)

def hex_to_rgb(value):
    """Return (red, green, blue) for the color given as #rrggbb."""
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

def convert_color(color, origin, target, conversion=bgr2lab): 
    """converts color from one color space to another 
    parameter: tuple/list of color (3-valued int/float) 
    returns: tuple/list of color (3-valued int)"""
    # pre-processing
    if origin == "LAB": 
        assert color[0] <= 100 and color[0]>=0, "Color not in LAB scale."
        assert color[1] <= 128 and color[1]>=-128, "Color not in LAB scale."
        assert color[2] <= 128 and color[2]>=-128, "Color not in LAB scale."  
    if origin == "RGB" and target == "HEX": 
        if type(color[0]) == float: 
            color = int(color[0]*100), int(color[1]*100), int(color[2]*100)
        r,g,b = color
        color = rgb_to_hex(r,g,b)
        return color 
    if origin == "HEX" and target == "RGB": 
        color = hex_to_rgb(color)
        return color 
    if (origin=="RGB" or origin=="BGR") and type(color[0]) == int:
        assert color[0] <=255 and color[1] <= 255 and color[2] <= 255, "Color not in 0-255 RGB scale."
        # from 0-255 scale to 0-1 scale 
        a,b,c = color
        color = a/255, b/255, c/255
    elif (origin=="RGB" or origin=="BGR") and type(color[0]) == float: 
        assert color[0] <=1 and color[1] <= 1 and color[2] <= 1, "Color not 0-1 RGB scale."
    if origin == "HSV" and color[1] >= 1: 
        color = color[0], color[1]/100, color[2]/100
    if origin == "LAB" or origin == "LCH": 
        assert color[0] <= 100, 'Luminance channel of color is not in the scale.' 
    if origin == "LAB" and target == "LCH": 
        color = lab2lch(color)
        color = round(color[0],1), round(color[1],1), int(round(color[2]))
        return color
    if origin == "LCH" and target == "LAB": 
        color = lch2lab(color)
        color = int(round(color[0],0)), int(round(color[1],0)), int(round(color[2]))
        return color
    # convert color    
    color = cv2.cvtColor(np.array([[color]], dtype=np.float32), conversion)[0, 0]
    # post-processing
    if target == "RGB" or target == "BGR": 
        color = color *255
    if target == "HSV": 
        color = int(round(color[0],0)), round(color[1]*100,1), round(color[2]*100,1)
        return color 
    a,b,c = color 
    color = int(round(a,0)), int(round(b,0)), int(round(c,0)) 
#    color = round(a,2), round(b,2), round(c,2) 
    return color        


# HSV: 8-bit to 32-bit *** 32-bit to 8-bit  
def convert_hsv8_32bit(color, source, dest):
    """  Convert one specific HSV color 
    - from 8-bit image HSV to 32-bit image HSV 
    - from 32-bit image HSV to 8-bit image RGB
    - 8-bit image: 0-180°, 0-255, 0-255 scales
    - 32-bit image: 0-360°,0-100,0-100 scales
    Arguments: 
        color -- list, original HSV color 
        source -- str, original HSV bit image space
        dest -- str, target HSV bit image space
    Returns:
        color -- list, target HSV color """
   
    if source == "8-bit" and dest == "32-bit": 
        h,s,v  = color
        assert h <= 180 and s <= 255 and v <= 255 
        h = int(h*2)
        s = int(round( s/255*100, 0))
        v = int(round( v/255*100, 0))
        color = (h, s, v) # to 0-360, 0-100, 0-100
    elif source == "32-bit" and dest == "8-bit": 
        h,s,v  = color
        assert h <= 360 and s <= 100 and v <= 100
        h = int(h/2)
        s = int(round( s/100*255, 0))
        v = int(round( v/100*255, 0))
        color = (h, s, v)  # to 0-180, 0-255, 0-255
    return color 

# # 8-bit image in HSV space
# dark_orange = (1,190,200)
# convert_hsv8_32bit(dark_orange, '8-bit', '32-bit')
# # (2, 75, 78)

# dark_white = (28,25,82) 
# convert_hsv8_32bit(dark_orange, '32-bit', '8-bit')
# # (14, 64, 209)


# mapping from radians to cartesian (based on function lch to lab) 
def hsvdeg2hsvcart(hsv, h_as_degree = True):
    """
    convert hsv in polar view to cartesian view
    """
    if not isinstance(hsv, np.ndarray):
        hsv = np.array(hsv, dtype=np.float32)
    hsv_cart = np.zeros_like(hsv, dtype=np.float32)
    hsv_cart[..., 2] = hsv[..., 2]
    if h_as_degree:
        hsv[..., 0] = hsv[..., 0] *np.pi / 180
    hsv_cart[..., 0] = hsv[..., 1]*np.cos(hsv[..., 0])
    hsv_cart[..., 1] = hsv[..., 1]*np.sin(hsv[..., 0])
    return hsv_cart  




#%%
color = (0.5,0.2,0.3)
type(color[0])
color[0]
value = [120,78,50]


print(convert_color((0.5,0.2,0.3), "RGB", "BGR", rgb2bgr))
print(convert_color((0.5,0.2,0.3), "BGR", "RGB", bgr2rgb))

print(convert_color((0.5,0.2,0.3), "RGB", "HEX"))
print(convert_color('#32141e', "HEX", "RGB"))

print(convert_color((0.5,0.2,0.3), "RGB", "HSV", rgb2hsv)) 
print(convert_color((340,60.2,50.2), "HSV", "RGB", hsv2rgb))

print(convert_color((128,51,77), "RGB", "LAB", rgb2lab))
print(convert_color((100,51,77), "LAB", "RGB", lab2rgb))

print(convert_color((100,51,77), "LAB", "LCH")) 
print(convert_color((90,60.2,50), "LCH", "LAB"))

#%%
import cv2
import numpy as np

def convert_color(col, cvt_code, ret_image = False):
    if isinstance(col, np.ndarray):
        if col.dtype == np.uint8:
            col = col.astype(np.float32) / 255
    elif isinstance(col[0], int):
        col = np.array(col, dtype = np.float32) / 255
    col = np.array(col, dtype=np.float32)
    if len(col.shape) == 1:
        col = np.array([[col]],dtype=np.float32)
    if ret_image:
        return cv2.cvtColor(col, cvt_code)
    else:
        return cv2.cvtColor(col, cvt_code)[0, 0]
def check_datatype(col):
    if col.dtype != np.float32:
        col = col.astype(np.float32) / 255
    return col

def myfunc(v):
    v = check_datatype(v)
    assert v.dtype == np.float32, ValueError("bad Datatype")
c1 = [128,128,128]
print(convert_color(c1, cv2.COLOR_BGR2LAB))
c1 = [0.5,0.5,0.5]
print(convert_color(c1, cv2.COLOR_BGR2LAB))

#%%
#############
### IMAGE ###
#############

# set directory 
os.getcwd()
os.chdir(r'D:\thesis\input_images')
# Image Channel (Color Space) Conversion 
# load BGR image 
image = cv2.imread('nemo.png') # BGR with numpy.uint8, 0-255 val 
print(image.shape) 
# (382, 235, 3)
# plt.imshow(image)
# plt.show()

# it looks like the blue and red channels have been mixed up. 
# In fact, OpenCV by default reads images in BGR format.
# OpenCV stores RGB values inverting R and B channels, i.e. BGR, thus BGR to RGB: 

# BGR to RGB 
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image) #now it is in RGB 
plt.show()

#plot a color in image
# color = image[381,234]
# print(color)
#[203 151 103] # correct color

# RGB to HSV 
def floatify(img_uint8): 
    img_floats = img_uint8.astype(np.float32) / 255 
    return img_floats

image = floatify(image)
# convert numpy.uint8 to numpy.float32 after imread for color models (HSV) in range 0-1 requiring floats

# convert RGB to HSV
hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV) #last rendering of image is in RGB 
plt.imshow(hsv_image) #now it is in HSV 
plt.show()




#%%


#############################################
### Builds Dataframes: Color Wheel Colors ###
#############################################

# Requirement: Color Conversion (see section above)
# Color Wheel 30°-steps in different color spaces 
import pandas as pd 

# HSV-Color Wheel in 30°-steps 
lst1 = [(255,0,0) ,(255,128,0) ,(255,255,0),(128,255,0),(0,255,0),(0,255,128),(0,255,255),(0,128, 255),(0, 0, 255),(128, 0, 255),(255, 0, 255),(255, 0, 128)]
lst2 = []
lst3 = ['red', 'orange', 'yellow', 'green-yellow', 'green', 'green-blue', 'cyan', 'blue-yellow','blue','purple','magenta','red-yellow']

for i in range(len(lst1)): 
    lst2.append(rgb2hsv(lst1[i]))
    
df = pd.DataFrame()
df['RGB'] = lst1
df['HSV'] = lst2
df['name'] = lst3

os.chdir(r'D:\thesis\code\pd12hues')
df.to_csv('rgbhsv_12.csv')

#%%

# LCH-Color Wheel in 30°-steps 
# get 12 hues of 30°-steps for L*CH's H-channel 
lablch_twelve = dict()
for i in np.linspace(0,360,num=12, endpoint=False): 
    lch_colors = (50,100,i)
    lab_colors = lab2lch(lch_colors)
    lablch_twelve[lch_colors] = lab_colors

# build pd frame with lab lch for 12 30°-step hues


lst = []
lst2 = []
for i in lablch_twelve.items(): 
    # {'LCH': i[0]}
    lst.append(i[0]) 
    lst2.append(i[1])
 
lst3 = []
for i in range(len(lst2)): 
    lst3.append(rgb2lab(lst2[i]))

lst4 = ['fuchsia', 'red', 'terracotta','olive', 'kelly','leaf', 'teal','atoll', 'azure','blue','purple','lilac']

import pandas as pd  
lablch_12 = pd.DataFrame()
lablch_12['LCH'] =lst
lablch_12['Lab'] =lst2
lablch_12['RGB'] = lst3
lablch_12['name'] = lst4

lablch_12= pd.read_csv('lablchrgb_12_handcorrected.csv')

# save dataframe with 12 lab-lch hues 
os.chdir(r'D:\thesis\code\pd12hues')
lablch_12.to_csv('lablchrgb_12.csv')
lablch_12.to_csv('lablchrgb_12_handcorrected.csv')

