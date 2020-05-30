# -*- coding: utf-8 -*-
"""
Created on Sat May 16 18:05:57 2020

@author: Anonym
"""

# import modules
import os
import matplotlib.pyplot as plt #2.1.2
import numpy as np
import cv2
import sys

# add path to environment var for importing own modules
print(sys.path)
sys.path.append('D:\thesis\code\ColorCode12')



#%%
#get all images in a folder
FOLDER_PATH = r'D:\thesis\code\ColorCode12\pinterest\vintage-style'
os.chdir(FOLDER_PATH)
BOARDNAME = os.path.split(FOLDER_PATH)[-1]
BOARDNAME = BOARDNAME[0].upper()+BOARDNAME[1:]

IMAGE = '0_#BD8CB5_4b2578e3ee58ba107806ee430ead95.jpg'
IMAGE_HEX = IMAGE.split('_')[1]



#%%
bgr2rgb = cv2.COLOR_BGR2RGB
rgb2bgr = cv2.COLOR_RGB2BGR
bgr2lab = cv2.COLOR_BGR2Lab
lab2bgr = cv2.COLOR_Lab2BGR
rgb2lab = cv2.COLOR_RGB2Lab
lab2rgb = cv2.COLOR_Lab2RGB
rgb2hsv = cv2.COLOR_RGB2HSV
hsv2rgb = cv2.COLOR_HSV2RGB

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
import cv2

image = cv2.imread(IMAGE) # BGR with numpy.uint8, 0-255 val 
print(image.shape)
 
# plot image in RGB 
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image) #now it is in RGB 
plt.axis('off')
plt.show()

#%%
# Dominant Color
 
# define RGB color 
dom_rgb_color = convert_color(IMAGE_HEX, "HEX", "RGB")
dom_lab_color = convert_color(dom_rgb_color, "RGB", "LAB", rgb2lab)

# define a numpy shape 
dom_square = np.full((10, 10, 3), dom_rgb_color, dtype=np.uint8) / 255.0

# visualize color in plt
plt.figure(figsize = (5,2))
plt.imshow(dom_square) 
plt.axis('off')
plt.show() 



#%%
# Color Palette 10 most dominant colors

import pandas as pd

# set directory 
import os 
os.chdir(r'D:\thesis\code\ColorCode12\pinterest\vintage-style')

# get bgr and ratio of colors 
palette = pd.read_csv('0_#BD8CB5_4b2578e3ee58ba107806ee430ead95_bgr_palette.csv')
ratio_widths = eval(palette['row 20'].iloc[3])
bgr_colors = eval(palette['row 20'].iloc[4])
b = [int(i[0]) for i in bgr_colors]
g = [int(i[1]) for i in bgr_colors]
r = [int(i[2]) for i in bgr_colors]
bgr_colors = list(zip(b,g,r))
bgr_colors = [list(i) for i in bgr_colors]
palette = pd.DataFrame({'bgr_colors':bgr_colors,
                        'ratio': ratio_widths})

palette = palette.sort_values(by=['ratio'], ascending=False).reset_index()

# get rgb of colors 
rgb_colors = []
lab_colors = []
for i in range(len(bgr_colors)):
    rgb = convert_color(palette['bgr_colors'].iloc[i], "BGR", "RGB", bgr2rgb)
    rgb_colors.append(rgb)
    lab = convert_color(palette['bgr_colors'].iloc[i], "BGR", "LAB", bgr2lab)
    lab_colors.append(lab)
    
palette['rgb_colors'] = rgb_colors
palette['lab_colors'] = lab_colors

# visualize color palette 
colorbar_count = 10
rgbcolors = palette['rgb_colors'].iloc[:colorbar_count]
labcolors = palette['lab_colors'].iloc[:colorbar_count]

#%%
# visualize color palette

# fill square 
squares = []
for i in range(len(rgbcolors)): 
    square = np.full((10, 10, 3), (rgbcolors[i]), dtype=np.uint8)
    squares.append(square)
    
#display colors patches in RGB
fig = plt.figure(figsize = (30,5))
for i in range(1,colorbar_count+1):
    plt.subplot(colorbar_count, 1, i)
    plt.imshow(squares[i-1]) 
    plt.axis('off')
#    plt.title(f'{i}', size=20)
    #plt.title(f'{color} \nRGB: {rgb} \nHSV: {hsv}', size=20)
plt.show()


    
#%%
# CP12 Classification of Color Palette + Dominant Color

# set directory 
import os 
os.chdir(r'D:\thesis\code\ColorCode12')

# import functions from other file
from ColorMLClassificationTo12CPs import display_color, knn_classify_color, knn_classifiedprob_color
# import functions and variables from other file
from ColorMLClassificationTo12CPs import *  

# top 10 colors + dominant color 
lab_cols = labcolors.tolist()
lab_cols.append(dom_lab_color)

# set directory 
os.getcwd()
os.chdir(r'D:\thesis\code\ColorCode12\pd4cpInlab')

# load dataframes 
data = pd.read_csv('12seasons.csv')
N_NEIGHBORS = 5

# return results 
pred_seasons = []
for LAB_COLOR in lab_cols: 
    # single color 
    print('---') 
    print('CLASSIFIER: KNN') 
    print(f'# neighbors: {N_NEIGHBORS}') 
    print('---') 
    print('Given Color: ')
    display_color(LAB_COLOR)
    season = knn_classify_color(LAB_COLOR, N_NEIGHBORS, data)
    print('Predicted Season: ', season) 
    pred_seasons.append(season)
    print('---') 
    probability_distribution = knn_classifiedprob_color(LAB_COLOR, N_NEIGHBORS, data)
    print('Top-3 highest probability: \n', probability_distribution.iloc[:3]) 

# many colors 
print('---') 
print("Top Seasons for ALL lab colors in Palette:")
print('---') 
from collections import Counter
counter = Counter(pred_seasons).most_common()
#print(counter)
topval = counter[0][1]
winseasons =  []
for i in counter: 
    if i[1] == topval: 
        winseasons.append(i[0])
print('Predicted Season for Palette:', ', '.join(winseasons))

#%%
# too many winseasons

# get season for all 101 colors 
if len(winseasons) >= 1: 
    lab_cols = palette['lab_colors']
# run pevious cell with all lab colors 

#%%

# image composition
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

print(image.shape)

# fill square 
squares = []
for i in range(len(rgbcolors)): 
    square = np.full((10, 10, 3), (rgbcolors[i]), dtype=np.uint8)
    squares.append(square)

os.chdir(r"D:\thesis\code\ColorCode12\pinterest\processed")
fig = plt.figure(figsize = (8,5))
gs = GridSpec(nrows=10, ncols=3,width_ratios=[10, 1, 6])
gs.update(wspace=0.0, hspace=0.05)
ax0 = fig.add_subplot(gs[:, 0]) # nrow, ncol
ax0.imshow(image) 
ax0.axis('off')
for i in range(0,10): 
    ax1 = fig.add_subplot(gs[i, 1])
    ax1.imshow(squares[i]) 
    ax1.axis('off')
ax2 = fig.add_subplot(gs[:4, 2]) # nrow, ncol
ax2.imshow(dom_square)
ax2.axis('off')
ax2 = fig.add_subplot(gs[4, 2]) # nrow, ncol
ax2.text(0.18,0,s=f"{' or '.join(winseasons)}", fontsize=18, fontweight='bold')
ax2.axis('off')
ax2 = fig.add_subplot(gs[5, 2]) # nrow, ncol
ax2.text(0.18,0,s=f"{BOARDNAME}", fontsize=18)
ax2.axis('off')
fig.tight_layout()
plt.savefig(f"{BOARDNAME}_{winseasons[0]}_{IMAGE}")
plt.show()

