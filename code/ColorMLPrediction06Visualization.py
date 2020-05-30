# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 16:25:48 2020

@author: Linda Samsinger

=====================
Classification Visualization
=====================

BOTTOM-UP APPROACH:
Use manually-classified test colors and an overlay of their respective 
color class center to visualize and determine 
the decision boundaries in a particular color space. 

Same goes for ML-classified test colors and their color class center. 
 
"""

#####################################
### Load Data 
 
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2


# to specify
DATA = 'VIAN'

# load data (requirements: LAB, (RGB,) HEX)
if DATA == 'SIX_COLORS': 
    os.chdir(r'D:\thesis\code\pd6hues')
    testcolors = pd.read_csv('rgbhsvlablch_6.csv', index_col=[0])
elif DATA == 'VIAN': 
    os.chdir(r'D:\thesis\code\pd28vianhues')
    testcolors = pd.read_excel('lab_vian_colors_testcolors216.xlsx')
    classcenter = pd.read_excel('labbgr_vian_colors_avg.xlsx', index_col=[0])
elif DATA == 'ELEVEN_COLORS': 
    os.chdir(r'D:\thesis\code\pd11hues')
    testcolors = pd.read_csv('rgbhsvlablchhex_11.csv', index_col=[0])    



# testcolors["cielab"] vs. classcenter["lab"]

LABEL = "cielab"# vs. cielab 
LABEL2 = "lab"
COLORS = "classcenter" # classcenter vs. testcolors 


data = testcolors.merge(classcenter, left_on='VIAN_color_category', right_on='vian_color', how='left')

#%%

###################################
### Plot Model as ScatterMatrix ###
###################################


# Plot LAB Space in a Matrix 

def convert_color(col, conversion=cv2.COLOR_BGR2Lab): #default 
    """" converts BGR to LAB by default supports all color spaces except lch 
    """
    color = cv2.cvtColor(np.array([[col]], dtype=np.float32), conversion)[0, 0]
    return color*255


# visualize a and b at constant luminance = 100 
    
PATCH = (100, 100, 3)
long = len(data[LABEL].tolist())
labels = data["VIAN_color_category"].tolist()
# 6*36 = 216

# original CS: lab 
if COLORS == "classcenter": 
    l = [eval(t)[0] for t in data[LABEL2].tolist()]
    a = [eval(t)[1] for t in data[LABEL2].tolist()] 
    b = [eval(t)[2] for t in data[LABEL2].tolist()]
    l2 = [eval(t)[0] for t in data[LABEL].tolist()]
elif COLORS == "testcolors": 
    l = [eval(t)[0] for t in data[LABEL].tolist()]
    a = [eval(t)[1] for t in data[LABEL].tolist()] 
    b = [eval(t)[2] for t in data[LABEL].tolist()]
# create lab colors (tuple-3)

ll = []
al = []
bl = []
lumlab = []

LUMINANCE = 0
for i in range(len(l)):
    if l2[i] == LUMINANCE: 
        ll.append(l[i])
        al.append(a[i])
        bl.append(b[i]) 
        lumlab.append(labels[i])

li = np.full(36, ll[0])
col = np.array(list(zip(li,al,bl)))

# len(cols[0]) 20 x 20 

matrix = []  

matrix.append(col.tolist()[:6])
matrix.append(col.tolist()[6:12]) 
matrix.append(col.tolist()[12:18]) 
matrix.append(col.tolist()[18:24]) 
matrix.append(col.tolist()[24:30]) 
matrix.append(col.tolist()[30:36]) 

          

# convert numpy of lab colors to bgr colors
bgr_cols = []
for j in matrix:
    bgr_col = []
    for i in j: 
        print(i)
        bgr = convert_color(i, cv2.COLOR_Lab2BGR)
        bgr_col.append(bgr)
    bgr_cols.append(bgr_col)
# len(bgr_cols[0])  20 x 20 

# put bgr colors into patches
result = []
for j in bgr_cols: 
    result_arr = []
    for i in j: 
        a = np.full(PATCH, i, dtype=np.uint8)
        result_arr.append(a)
    c = np.hstack(result_arr)
    result.append(c)
    
abcd = np.vstack(result)   
print(abcd.shape) #(2000, 2000, 3)

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
FONT_COLOR = BLACK 
abcd = cv2.putText(abcd, f'{lumlab[:6]}', (0, 50), cv2.FONT_HERSHEY_SIMPLEX, .8, FONT_COLOR, 1)
abcd = cv2.putText(abcd, f'{lumlab[6:12]}', (0, 150), cv2.FONT_HERSHEY_SIMPLEX, .8, FONT_COLOR, 1)
abcd = cv2.putText(abcd, f'{lumlab[12:18]}', (0, 250), cv2.FONT_HERSHEY_SIMPLEX, .9, FONT_COLOR, 1)
abcd = cv2.putText(abcd, f'{lumlab[18:24]}', (0, 350), cv2.FONT_HERSHEY_SIMPLEX, .95, FONT_COLOR, 1)
abcd = cv2.putText(abcd, f'{lumlab[24:30]}', (0, 450), cv2.FONT_HERSHEY_SIMPLEX, .85, FONT_COLOR, 1)
abcd = cv2.putText(abcd, f'{lumlab[30:36]}', (0, 550), cv2.FONT_HERSHEY_SIMPLEX, .85, FONT_COLOR, 1)

# cv2.imshow(f'LAB Matrix', abcd)
import os
os.chdir(r'D:\thesis\images')
cv2.imwrite(f'LAB_testcolors216_l{LUMINANCE}_label_{COLORS}.jpg', abcd)