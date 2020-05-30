# -*- coding: utf-8 -*-
"""
Created on Sat May  2 09:26:44 2020

@author: Linda Samsinger

Download Images form Google Image Search using an API 

"""



# to specify 
IMAGE = 'ultramarine'
FOLDER_PATH = r'D:\thesis\images\google\ultramarine'

#%%

import cv2 

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

#%%
### ALL IMAGES IN FOLDER ###

# openly 

## load modules
#from tkinter import *
#from tkinter.filedialog import askdirectory 
#from skimage.io import imread_collection
#
## pop up interface 
#root= Tk()    
#drcty = askdirectory(parent=root,title='Choose directory with image sequence stack files')
#print(drcty)
#path = str(drcty) + '/*.jpg'
#imgs = imread_collection(path) 
#root.destroy()
#
#print(f"Loaded {len(imgs)} images...")

#%%

# silently 

import os

files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(FOLDER_PATH):
    for file in f:
        if '.jpg' in file:
            files.append(os.path.join(r, file))

for f in files:
    print(f)
    
#%%
# find average color of an image
    
import numpy as np 
import cv2 
import pandas as pd
import matplotlib.pyplot as plt 
import statistics as s

# load and show image in BGR 
image = cv2.imread(files[15]) # BGR with numpy.uint8, 0-255 val 
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image) #now it is in RGB 
plt.axis('off')
plt.show()
image.shape

# crop image (remove white surroundings)
crop_img = image[80:100,80:100]

# calculate average RGB
average = image.mean(axis=0).mean(axis=0)
average = crop_img.mean(axis=0).mean(axis=0)

# show average color
a = np.full((100, 100, 3), average, dtype=np.uint8)
plt.title("Average Color")
plt.imshow(a) #now it is in RGB 
plt.axis('off')
plt.show()

#%%

# find average color of all images

avgs = []
for f in files: 
    image = cv2.imread(f) # BGR with numpy.uint8, 0-255 val 
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    try: 
        # crop image 
        MARGIN = 200
        image = image[MARGIN:image.shape[0]-MARGIN,MARGIN:image.shape[1]-MARGIN]
    except: 
        pass
    # calculate average RGB
    average = crop_img.mean(axis=0).mean(axis=0)
    avgs.append(list(average))

# average of averages RGB
avgs = np.array(avgs)
avgavg = avgs.mean(axis=0)
avgcolor = list(avgavg)
avgcolor = round(avgcolor[0]), round(avgcolor[1]), round(avgcolor[2])
print(f"Average RGB color across all images: {avgcolor}")

# show average of averages color
a = np.full((100, 100, 3), avgavg, dtype=np.uint8)
plt.imshow(a) #now it is in RGB 
plt.axis('off')
plt.show()

#%%
# find average color of all images 

COMPUTE_LAB = True 

avgs = []
for f in files: 
    # load image in BGR with numpy.uint8, 0-255 val 
    image = cv2.imread(f)  
    image = image.astype(np.float32) / 255
    # convert image to LAB 
    image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    try: 
        # crop image if possible 
        MARGIN = 50
        image = image[MARGIN:image.shape[0]-MARGIN,MARGIN:image.shape[1]-MARGIN]
    except: 
        pass
#     calculate average LAB
    average = image.mean(axis=0).mean(axis=0)
#    if np.isnan(np.array(average[0])) == 0: # if MARGIN too big 
    avgs.append(list(average))


# calculate average of averages LAB
avgs = np.array(avgs)
avgavg = avgs.mean(axis=0)
# transform to RGB for display
avgavg = convert_color(avgavg, "LAB", "RGB", lab2rgb)
avgcolor = list(avgavg)
avgcolor = round(avgcolor[0]), round(avgcolor[1]), round(avgcolor[2])
print(f"Average RGB color across all images: {avgcolor}")

# show average of averages color
a = np.full((100, 100, 3), avgavg, dtype=np.uint8)
plt.imshow(a) #now it is in RGB 
plt.axis('off')
plt.show()

#%%

# make dataframe for new color 

# transform color name to color values in all color spaces 

r,g,b = avgcolor
srgb = [r,g,b]

avgcolor2 = r/255,g/255,b/255
lab1 = convert_color(avgcolor2, "RGB", "LAB", rgb2lab)
l,a,b = lab1.tolist()
lab = np.round(l), np.round(a), np.round(b)
lab = list(lab)

hsv = convert_color(avgcolor2, "RGB", "HSV", rgb2hsv)
h,s,v = list(hsv)
hsv = np.round(h), np.round(s), np.round(v)
hsv = list(hsv)

lch = lab2lch(lab1)
l,c,h = list(lch)
lch = np.round(l), np.round(c), np.round(h)
lch = list(lch)

r,g,b = avgcolor
hx = rgb_to_hex(int(r), int(g), int(b))

col_decl = pd.DataFrame({'VIAN_color_category': 'ultramarine', 
                         'srgb': [srgb],
                         'cielab': [lab],
                         'hsv': [hsv],
                         'LCH': [lch],
                         'HEX': hx})

FOLDER_PATH = r'D:\thesis\images\google\ultramarine'
os.chdir(FOLDER_PATH)

col_decl.to_csv('ultramarine.csv')