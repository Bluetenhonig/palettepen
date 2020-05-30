# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 19:38:10 2020

@author: Linda Samsinger

Sort Color Palette by Hue 
Sort Color Palette by color ratio in image 
Sort Color Palette by Luminance 

"""

# import modules
import os
import matplotlib.pyplot as plt #2.1.2
import numpy as np
import pandas as pd
import cv2


#%%
########### Convert Color ###########

# convert color 
def convert_color(col, conversion=cv2.COLOR_BGR2Lab): #default 
    color = cv2.cvtColor(np.array([[col]], dtype=np.float32), conversion)[0, 0]
    return color


# convert numpy array of colors
def convert_array(nparray, origin, target): 
    # convert to RGB
    hsv_colors = []
    for col in nparray: 
        if origin == 'BGR' and target == 'HSV':        
            hsv_color = convert_color(col, cv2.COLOR_BGR2HSV)
        if origin == 'LAB' and target == 'HSV':    
            col = convert_color(col, cv2.COLOR_LAB2RGB)
            hsv_color = convert_color(col, cv2.COLOR_RGB2HSV)
        hsv_colors.append(hsv_color.tolist())
    return hsv_colors

#%%
    
########### Sort in HSV ###########
    
# to specify
PATH = r'D:\thesis\code\pd12seasons'
FILE = r'lab_cp1_knn5_steps51.csv'
NEW_FILE = r'lab_hsvsorted_cp1_knn5_steps51.csv'

# set directory 
os.chdir(PATH)
# load data
palette = pd.read_csv(FILE, index_col=0)

### processing
# eda data
palette.index.value_counts()
# restructure data
palett = [eval(l) for l in palette['lab']] # pd2list
# convert type 
palet = np.array(palett) # list2numpy array
# get hsv
hsvs = convert_array(palet, 'LAB', 'HSV')
# post hsv
palette['hsv'] = hsvs
# extract hue
palette['hue'] = palette.hsv.map(lambda x: int(round(x[0])))
# extract saturation
palette['sat'] = palette.hsv.map(lambda x: int(round(x[1])))
# extract value
palette['val'] = palette.hsv.map(lambda x: int(round(x[2])))
#sort by one column only: hue
#palette = palette.sort_values(by=['hue'])
# sort by multiple columns
palette = palette.sort_values(by=['hue', 'sat', 'val'])

#save palette
palette.to_csv(NEW_FILE)