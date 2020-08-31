# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 19:42:03 2020

@author: lsamsi

Panacea for easy conversion of color values to other color spaces. 

Step Before: have a color space value
Goal: convert a color space value to another color space value 
convert an image in a color space vale to another image in another color space value
Step After: visualization (typically) or other... 

Color Space Conversion: 
- For Colors
- For Images 

"""

# import modules
import os
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import cv2



#############
### COLOR ###   
#############

# declare variables 
bgr2rgb = cv2.COLOR_BGR2RGB
rgb2bgr = cv2.COLOR_RGB2BGR
bgr2lab = cv2.COLOR_BGR2Lab
lab2bgr = cv2.COLOR_Lab2BGR
rgb2lab = cv2.COLOR_RGB2Lab
lab2rgb = cv2.COLOR_Lab2RGB
rgb2hsv = cv2.COLOR_RGB2HSV
hsv2rgb = cv2.COLOR_HSV2RGB
rgb2hsl = cv2.COLOR_RGB2HLS
hsl2rgb = cv2.COLOR_HLS2RGB


# functions 

def get_all_available_color_spaces():  
    """loads all availbe color spaces in OpenCV"""
    flags = [i for i in dir(cv2) if i.startswith('COLOR_')]
    flag_len = len(flags)
    print(f"Color spaces available: {flag_len}")
    return flags
        


def convert_color(color, origin, target, conversion=bgr2lab): 
    """Main Color Converter: converts color from one color space to another 
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

# OpenCV color conversion extensions: 
    
### LCH 

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

### HEX
    
def rgb_to_hex(red, green, blue):
    """Return color as #rrggbb for the given color values."""
    return '#%02x%02x%02x' % (red, green, blue)

def hex_to_rgb(value):
    """Return (red, green, blue) for the color given as #rrggbb."""
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))



### HSV
    
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

 
def hsvdeg2hsvcart(hsv, h_as_degree = True):
    """
    convert hsv in polar view to cartesian view
    mapping from radians to cartesian (based on function lch to lab) 
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


#def convert_color(col, cvt_code, ret_image = False):
#    if isinstance(col, np.ndarray):
#        if col.dtype == np.uint8:
#            col = col.astype(np.float32) / 255
#    elif isinstance(col[0], int):
#        col = np.array(col, dtype = np.float32) / 255
#    col = np.array(col, dtype=np.float32)
#    if len(col.shape) == 1:
#        col = np.array([[col]],dtype=np.float32)
#    if ret_image:
#        return cv2.cvtColor(col, cvt_code)
#    else:
#        return cv2.cvtColor(col, cvt_code)[0, 0]
#def check_datatype(col):
#    if col.dtype != np.float32:
#        col = col.astype(np.float32) / 255
#    return col
#
#def myfunc(v):
#    v = check_datatype(v)
#    assert v.dtype == np.float32, ValueError("bad Datatype")
#c1 = [128,128,128]
#print(convert_color(c1, cv2.COLOR_BGR2LAB))
#c1 = [0.5,0.5,0.5]
#print(convert_color(c1, cv2.COLOR_BGR2LAB))

#%%
#############
### IMAGE ###
#############

# declare variables 
PATH = r'D:\thesis\input_images'
FILE =  'nemo.png'


def floatify(img_uint8): 
    """ RGB to HSV / BGR to LAB  """
    img_floats = img_uint8.astype(np.float32) / 255 
    return img_floats


def load_show_bgrimage(path, file, show=True): 
    """ load BGR image
    BGR with numpy.uint8, 0-255 val  """
    os.chdir(path)
    image = cv2.imread(file) 
    print(image.shape) 
    if show == True: 
        plt.imshow(image)
        plt.show()
    return image


def show_rgbimage(bgr_image, show=True):
    """ BGR to RGB image conversion"""
    rgb_image = cv2.cvtColor(bgr_image, bgr2rgb)
    if show == True: 
        plt.imshow(rgb_image) 
        plt.show()
    return rgb_image
    

def show_hsv_image(rgb_image, show=True): 
    """ RGB to HSV image conversion"""
    image = floatify(rgb_image)
    image = cv2.cvtColor(image, rgb2hsv)      
    if show == True: 
        plt.imshow(image) 
        plt.show()

def show_lab_image(bgr_image, show=True): 
    """ RGB to HSV image conversion"""
    image = floatify(bgr_image)
    image = cv2.cvtColor(image, bgr2lab)        
    if show == True: 
        plt.imshow(image) 
        plt.show()

def get_imagecolorpixel(image, coord1, coord2):   
     """ a color in an image is shown """
     color = image[coord1, coord2]
     return color
 
    

