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
os.chdir(r'D:\thesis\images')
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

#%%
###########################################
### Builds Dataframes: Color Classes ###
###########################################


# define 6 color categories as epicentres in color space

# load modules
import pandas as pd

# basic 6 colors (by Itten p. 22)
basic_colors = ['red', 'orange', 'yellow', 'green', 'blue', 'violet']
basic_colors_rgb  = [
                        [255,0,0],
                        [255,128,0],
                        [255,255,0],
                        [0,255,0],
                        [0,0,255],
                        [128,0,255]
                    ]

basic_colors_hsv  = []
basic_colors_lab  = []
basic_colors_lch  = []
basic_colors_hex  = []

for i in basic_colors_rgb:
    rgb = np.array(i)
    rgbi = np.array([[rgb/ 255]], dtype=np.float32)
    hsv = cv2.cvtColor(rgbi, cv2.COLOR_RGB2HSV)
    hsv = hsv[0, 0]
    basic_colors_hsv.append(hsv)

for i in basic_colors_rgb:
    rgb = np.array(i)
    rgbi = np.array([[rgb/ 255]], dtype=np.float32)
    lab = cv2.cvtColor(rgbi, cv2.COLOR_RGB2Lab)
    lab = lab[0, 0]
    lab = lab.tolist()
    lab = list(lab)
    basic_colors_lab.append(lab)
    
for i in basic_colors_rgb:
    rgb = np.array(i)
    rgbi = np.array([[rgb/ 255]], dtype=np.float32)
    lab = cv2.cvtColor(rgbi, cv2.COLOR_RGB2Lab)
    lch = lab2lch(lab, h_as_degree=True)[0, 0]
    lch = lch.tolist()
    basic_colors_lch.append(lch)

for i in basic_colors_rgb:
    r,g,b = np.array(i)
    hex_val = rgb_to_hex(r,g,b)
    basic_colors_hex.append(hex_val)
    
    
df = pd.DataFrame()
df['name'] = basic_colors
df['RGB'] = basic_colors_rgb
df['HSV'] = basic_colors_hsv
df['LAB'] = basic_colors_lab
df['LCH'] = basic_colors_lch
df['HEX'] = basic_colors_hex

os.chdir(r'D:\thesis\code\pd6hues')
df.to_csv('rgbhsvlablchhex_6.csv', index=True)

#%%

# define basic colors

# RGB space corners have the following 6 colors: 
#red, green, yellow, blue, violet, cyan 

#%%

# define basic colors

# LAB space corners have the following 9 colors: 
#blue, cyan, yellow, brown, pink, orange, red, green,  magenta


#%%


# define Boynton's 11 colors as epicenters 

# load modules
import pandas as pd

# 8 colors + 3 non-colors
basic_colors = ['orange', 'brown', 'gray', 'pink', 'magenta', 'yellow', 'green', 'red', 'blue', 'white', 'black']

# rgb of basic colors based on: colorhexa.com 
basic_colors_rgb  = [
                        [255,128,0], #orange
                        [165,42,42], #brown 
                        [128,128,128], #gray 
                        [255, 192, 203], #pink
                        [255, 0, 255], #magenta
                        [255,255,0], #yellow
                        [0,255,0], #green
                        [255,0,0], #red
                        [0,0,255], #blue
                        [255,255,255], #white
                        [0,0,0] #black
                    ]

basic_colors_hsv  = []
basic_colors_lab  = []
basic_colors_lch  = []
basic_colors_hex  = []

for i in basic_colors_rgb:
    rgb = np.array(i)
    rgbi = np.array([[rgb/ 255]], dtype=np.float32)
    hsv = cv2.cvtColor(rgbi, cv2.COLOR_RGB2HSV)
    hsv = hsv[0, 0]
    basic_colors_hsv.append(hsv)

for i in basic_colors_rgb:
    rgb = np.array(i)
    rgbi = np.array([[rgb/ 255]], dtype=np.float32)
    lab = cv2.cvtColor(rgbi, cv2.COLOR_RGB2Lab)
    lab = lab[0, 0]
    lab = lab.tolist()
    lab = list(lab)
    basic_colors_lab.append(lab)
    
for i in basic_colors_rgb:
    rgb = np.array(i)
    rgbi = np.array([[rgb/ 255]], dtype=np.float32)
    lab = cv2.cvtColor(rgbi, cv2.COLOR_RGB2Lab)
    lch = lab2lch(lab, h_as_degree=True)[0, 0]
    lch = lch.tolist()
    basic_colors_lch.append(lch)

for i in basic_colors_rgb:
    r,g,b = np.array(i)
    hex_val = rgb_to_hex(r,g,b)
    basic_colors_hex.append(hex_val)
    
    
df = pd.DataFrame()
df['name'] = basic_colors
df['RGB'] = basic_colors_rgb
df['HSV'] = basic_colors_hsv
df['LAB'] = basic_colors_lab
df['LCH'] = basic_colors_lch
df['HEX'] = basic_colors_hex

os.chdir(r'D:\thesis\code\pd11hues')
df.to_csv('rgbhsvlablchhex_11.csv', index=True)

#%%


# define Samsinger's 10 colors as epicenters (union of Itten + Boynton, RGB corners + LAB corners)

# load modules
import pandas as pd

basic_colors = [ 'pink',  'magenta', 'red', 'orange', 'yellow', 'brown', 'green', 'cyan', 'blue', 'violet']

# rgb of basic colors based on: colorhexa.com 
basic_colors_rgb  = [   
                      [255,192,203], #pink
                      [255,0,255], #magenta
                      [255,0,0], #red
                      [255,128,0], #orange              
                      [255,255,0], #yellow
                      [165,42,42], #brown
                      [0,255,0], #green
                      [0,255,255], #cyan
                      [0,0,255], #blue
                      [128,0,255] #violet
                    ]


basic_colors_hsv  = []
basic_colors_lab  = []
basic_colors_lch  = []
basic_colors_hex  = []

for i in basic_colors_rgb:
    rgb = np.array(i)
    rgbi = np.array([[rgb/ 255]], dtype=np.float32)
    hsv = cv2.cvtColor(rgbi, cv2.COLOR_RGB2HSV)
    hsv = hsv[0, 0]
    basic_colors_hsv.append(hsv)

for i in basic_colors_rgb:
    rgb = np.array(i)
    rgbi = np.array([[rgb/ 255]], dtype=np.float32)
    lab = cv2.cvtColor(rgbi, cv2.COLOR_RGB2Lab)
    lab = lab[0, 0]
    lab = lab.tolist()
    lab = list(lab)
    basic_colors_lab.append(lab)
    
for i in basic_colors_rgb:
    rgb = np.array(i)
    rgbi = np.array([[rgb/ 255]], dtype=np.float32)
    lab = cv2.cvtColor(rgbi, cv2.COLOR_RGB2Lab)
    lch = lab2lch(lab, h_as_degree=True)[0, 0]
    lch = lch.tolist()
    basic_colors_lch.append(lch)

for i in basic_colors_rgb:
    r,g,b = np.array(i)
    hex_val = rgb_to_hex(r,g,b)
    basic_colors_hex.append(hex_val)
    
    
df = pd.DataFrame()
df['name'] = basic_colors
df['RGB'] = basic_colors_rgb
df['HSV'] = basic_colors_hsv
df['LAB'] = basic_colors_lab
df['LCH'] = basic_colors_lch
df['HEX'] = basic_colors_hex

os.chdir(r'D:\thesis\code\pd11hues')
df.to_csv('rgbhsvlablchhex_10.csv', index=True)

# hand-adjusted values for HSV and LCH values in dictionary 
# basic_ten = {'red': {'rgb': [255,0,0], 'hsv': [0,1,1], 'lch': [50,100,40]}
#             , 'orange': {'rgb': [255,128,0], 'hsv': [30,1,1], 'lch': [70,100,60]}
#             , 'yellow': {'rgb': [255,255,0], 'hsv': [60,1,1], 'lch': [100,100,100]}
#             , 'green': {'rgb': [0,255,0], 'hsv': [120,1,1], 'lch': [90,100,130]}
#             , 'blue': {'rgb': [0,0,255], 'hsv': [240,1,1], 'lch': [30,100,300]}
#             , 'pink': {'rgb': [255,192,203], 'hsv': [350,.25,1], 'lch': [85,100,10]}
#             , 'magenta': {'rgb': [255,0,255], 'hsv': [300,1,1], 'lch': [60,100,330]}
#             , 'brown': {'rgb': [165,42,42], 'hsv': [0,1,.65], 'lch': [40,60,30]}
#             , 'cyan': {'rgb': [0,255,255], 'hsv': [180,1,1], 'lch': [90,50,190]}
#             , 'violet': {'rgb': [128,0,255], 'hsv': [270,1,1], 'lch': [40,100,315]}
#             }

#%%


# VIAN's 28 colors 
# convert srgb, lab into hsv, hsl and lch values 

# import modules
import os
import matplotlib.pyplot as plt #2.1.2
import numpy as np
import pandas as pd
import cv2

# load data
FOLDER_PATH = r'D:\thesis\images\google\ultramarine'
os.chdir(FOLDER_PATH)
col_decl = pd.read_csv('ultramarine.csv')

# load different data 
PATH = r'D:\thesis\code\pd28vianhues'
FILE = 'SRGBLABhsvhslLCHHEX_EngALL_VIANHuesColorThesaurus.xlsx'
os.chdir(PATH)
data = pd.read_excel(FILE, sep=" ", index_col=0)

# add new row to data set(for ultramarine)
new_row = col_decl[['VIAN_color_category', 'cielab', 'srgb', 'cielab', 'hsv', 'LCH', 'HEX']]
data = data.append(new_row.to_dict('records') , ignore_index=True)
data["language"].iloc[720] = "English"
data["english name"].iloc[720] = "ultramarine"

data['srgb_R'].iloc[-1] = int(data['srgb'].iloc[-1][1:3])
data['srgb_G'].iloc[-1] = int(data['srgb'].iloc[-1][7:9])
data['srgb_B'].iloc[-1] = int(data['srgb'].iloc[-1][12:16])

# fill up other cells with to fill up last row 
data['cielab_L'].iloc[-1] = eval(data['cielab'].iloc[-1])[0]
data['cielab_a'].iloc[-1] = eval(data['cielab'].iloc[-1])[1]
data['cielab_b'].iloc[-1] = eval(data['cielab'].iloc[-1])[2]
data['hsv_H'].iloc[-1] = eval(data['hsv'].iloc[-1])[0]
data['hsv_S'].iloc[-1] = eval(data['hsv'].iloc[-1])[1]
data['hsv_V'].iloc[-1] = eval(data['hsv'].iloc[-1])[2]
data['hsl_H'].iloc[-1] = None
data['hsl_S'].iloc[-1] = None
data['hsl_L'].iloc[-1] = None
data['LCH_L'].iloc[-1] = eval(data['LCH'].iloc[-1])[0]
data['LCH_C'].iloc[-1] = eval(data['LCH'].iloc[-1])[1]
data['LCH_H'].iloc[-1] = eval(data['LCH'].iloc[-1])[2]
data['HEX'].iloc[-1] = basic_colors_hex[0]

# convert rgb into all other cs 
data_small = data[['srgb_R', 'srgb_G', 'srgb_B']]
basic_colors_rgb = data_small.values.tolist()

basic_colors_hsv  = []
basic_colors_hsl  = []
basic_colors_lch  = []
basic_colors_hex  = []

for i in basic_colors_rgb:
    rgb = np.array(i)
    rgbi = np.array([[rgb/ 255]], dtype=np.float32)
    hsv = cv2.cvtColor(rgbi, cv2.COLOR_RGB2HSV)
    hsv = hsv[0, 0]
    hsv = hsv.tolist()
    basic_colors_hsv.append(hsv)
    
for i in basic_colors_rgb:
    rgb = np.array(i)
    rgbi = np.array([[rgb/ 255]], dtype=np.float32)
    hsl = cv2.cvtColor(rgbi, cv2.COLOR_RGB2HLS) 
    hsl = hsl[0, 0]
    hsl = hsl.tolist()
    basic_colors_hsl.append(hsl)
    
for i in basic_colors_rgb:
    rgb = np.array(i)
    rgbi = np.array([[rgb/ 255]], dtype=np.float32)
    lab = cv2.cvtColor(rgbi, cv2.COLOR_RGB2Lab)
    lch = lab2lch(lab, h_as_degree=True)[0, 0]
    lch = lch.tolist()
    basic_colors_lch.append(lch)
    
for i in basic_colors_rgb:
    r,g,b = np.array(i)
    hex_val = rgb_to_hex(int(r),int(g),int(b))
    basic_colors_hex.append(hex_val)
    
data['hsv'] = basic_colors_hsv
data['hsv_H'] = [i[0] for i in basic_colors_hsv]
data['hsv_S'] = [i[1] for i in basic_colors_hsv]
data['hsv_V'] = [i[2] for i in basic_colors_hsv]
data['hsl'] = basic_colors_hsl
data['hsl_H'] = [i[0] for i in basic_colors_hsl]
data['hsl_S'] = [i[1] for i in basic_colors_hsl]
data['hsl_L'] = [i[2] for i in basic_colors_hsl]
data['LCH'] = basic_colors_lch
data['LCH_L'] = [i[0] for i in basic_colors_lch]
data['LCH_C'] = [i[1] for i in basic_colors_lch]
data['LCH_H'] = [i[2] for i in basic_colors_lch]
data['HEX'] = basic_colors_hex

os.chdir(r'D:\thesis\code\pd28vianhues')
data.to_excel('SRGBLABhsvhslLCHHEX_EngAll_VIANHuesColorThesaurus.xlsx', index=True)

#%%
###########################################
### Builds Dataframes: Gridpoints   ###
###########################################


os.chdir(r'D:\thesis\code\pd4lab')
df = pd.read_csv('LAB_ABgridpoints.csv')

lst = []
for index, row in df.iterrows():
    lst.append([row[0], row[1], row[2]])
df['LAB'] =lst

hexs = []
for lab in df['LAB']:
    r,g,b = convert_color(lab, cv2.COLOR_Lab2RGB)
    r= int(round( r*255, 0))
    g= int(round( g*255, 0))
    b= int(round( b*255, 0))
    hex_val = rgb_to_hex(r,g,b)
    hexs.append(hex_val)

df['HEX'] = hexs

# save dataframe 
os.chdir(r'D:\thesis\code\pd4lab')
df.to_csv('LABHEX_ABgridpoints.csv')


#%%
##########################
### XKCD Color Survey  ###
##########################

# import modules
import os
import matplotlib.pyplot as plt #2.1.2
import numpy as np
import pandas as pd
import cv2

os.chdir(r'D:\thesis\color-survey')
df = pd.read_excel('xkcd.xlsx')


rgbs = []
labs = []
for hx in df['HEX']:
    rgb = convert_color(hx, "HEX", "RGB")
    rgbs.append(rgb)
    lab = convert_color(rgb, "RGB", "LAB", rgb2lab)
    labs.append(lab)

df['RGB'] = rgbs
df['LAB'] = labs

# save dataframe 
os.chdir(r'D:\thesis\code\pd4xkcd')
df.to_csv('HEXRGBLAB_xkcd.csv')

