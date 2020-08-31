# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 20:23:20 2020

@author: Anonym
"""

import numpy as np
import cv2 

#%%

# Basic Matrix Color Plotting
# Plot 4 Colors in a Matrix 

# specify four bgr colors 
bgr_1 = [135, 173, 145]
bgr_2 = [135, 173, 25]
bgr_3 = [135, 120, 145]
bgr_4 = [85, 173, 25]

# specify color patch area size for colors to display 
PATCH = (100, 100, 3)
a = np.full(PATCH, bgr_1, dtype=np.uint8)
b = np.full(PATCH, bgr_2, dtype=np.uint8)
c = np.full(PATCH, bgr_3, dtype=np.uint8)
d = np.full(PATCH, bgr_4, dtype=np.uint8)

# stack numpy array together - matrix 
ab = np.vstack((a, b))
cd = np.vstack((c, d))
abcd = np.hstack((ab, cd))
print(a.shape) 
print(ab.shape) 
print(cd.shape) 
print(abcd.shape) 

cv2.imshow('Four Colors in a Matrix', abcd)

#%%

# Plot LAB Space in a Matrix 

def convert_color(col, conversion=cv2.COLOR_BGR2Lab): #default 
    """" converts BGR to LAB by default supports all color spaces except lch 
    """
    color = cv2.cvtColor(np.array([[col]], dtype=np.float32), conversion)[0, 0]
    return color*255


# LAB Domain
# L = [0, 100]
# a = [-128, 128]
# b = [-128, 128]

PATCH = (100, 100, 3)
COLORS_WIDTH_COUNT = 20
LUMINANCE = 80 

# original CS: lab 
# l = np.full(COLORS_WIDTH_COUNT, LUMINANCE)
l = np.linspace(0, 100, COLORS_WIDTH_COUNT)
a = np.linspace(0, 0, COLORS_WIDTH_COUNT) 
b = np.linspace(-128, 128, COLORS_WIDTH_COUNT)

# create lab colors (tuple-3)
cols = []
for l_el in l:
    li = np.full(COLORS_WIDTH_COUNT, l_el)
    col = np.array(list(zip(li,a,b)))
    cols.append(col)
# len(cols[0]) 20 x 20 

# convert numpy of lab colors to bgr colors
bgr_cols = []
for j in cols:
    bgr_col = []
    for i in j: 
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

#abcd = cv2.putText(abcd, 'b*: 128 (blue-yellow)', (70, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
#abcd = cv2.putText(abcd, 'Luminance: 0 (dark)', (70, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
#abcd = cv2.putText(abcd, 'Luminance: 100 (light)', (70, 1770), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
#abcd = cv2.putText(abcd, 'a*: -128 (green)', (70, 1870), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
#abcd = cv2.putText(abcd, 'a*: 128 (red)', (1680, 1870), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

# cv2.imshow(f'LAB Matrix', abcd)
import os
os.chdir(r'D:\thesis\output_images')
cv2.imwrite(f'LAB_Matrix_a0_.jpg', abcd)


#%%

# Plot LCH Space in a Matrix 
def convert_color(col, conversion=cv2.COLOR_BGR2Lab): #default 
    """" converts BGR to LAB by default supports all color spaces except lch 
    """
    color = cv2.cvtColor(np.array([[col]], dtype=np.float32), conversion)[0, 0]
    return color*255

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


PATCH = (100, 100, 3)
COLORS_WIDTH_COUNT = 20
 

# original CS: lch 
l = np.linspace(0, 100, COLORS_WIDTH_COUNT) # luminance dark-to-light: 0-100 
c = np.linspace(100, 100, COLORS_WIDTH_COUNT) # constant chroma: 100
h = np.linspace(0, 360, COLORS_WIDTH_COUNT) # hue angles around h-channel: 0-360

# create lch colors (tuple-3)
cols = []
for l_el in l:
    li = np.full(COLORS_WIDTH_COUNT, l_el)
    col = np.array(list(zip(li,c,h)))
    cols.append(col)
# len(cols[0]) 20 x 20 



# convert numpy of lch colors to bgr colors
bgr_cols = []
for j in cols:
    bgr_col = []
    for i in j: 
        bgr = convert_color(lch2lab(i), cv2.COLOR_Lab2BGR) 
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

#abcd = cv2.putText(abcd, 'chroma: 0 (empty)', (70, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
#abcd = cv2.putText(abcd, 'luminance: 0 (dark)', (70, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255,255), 2)
#abcd = cv2.putText(abcd, 'luminance: 100 (light)', (70, 1770), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
#abcd = cv2.putText(abcd, 'hue: 0', (70, 1870), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
#abcd = cv2.putText(abcd, 'hue: 360', (1680, 1870), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

# cv2.imshow(f'LAB Matrix', abcd)
import os
os.chdir(r'D:\thesis\output_images')
cv2.imwrite(f'LCH_Matrix_c100_.jpg', abcd)


#%%

# Plot RGB Space in a Matrix 

def convert_color(col, conversion=cv2.COLOR_BGR2Lab): #default 
    """" converts BGR to LAB by default, supports all color spaces except lch 
    """
    color = cv2.cvtColor(np.array([[col]], dtype=np.float32), conversion)[0, 0]
    return color


PATCH = (100, 100, 3)
COLORS_WIDTH_COUNT = 20


# original CS: rgb 
r = np.linspace(0, 0, COLORS_WIDTH_COUNT) # luminance dark-to-light: 0-100 
g = np.linspace(0, 255, COLORS_WIDTH_COUNT) # constant chroma: 100
b = np.linspace(0, 255, COLORS_WIDTH_COUNT) # hue angles around h-channel: 0-360

# create rgb colors (tuple-3)
# cols = []
# for l_el in r:
#     ri = np.full(COLORS_WIDTH_COUNT, l_el)
#     col = np.array(list(zip(ri,g,b)))
#     cols.append(col)
# len(cols) 20 x 20 

# if y-axis cannot be held constant - not red, (or luminance), change code
cols = []
for l_el in g:
    gi = np.full(COLORS_WIDTH_COUNT, l_el)
    col = np.array(list(zip(r,gi,b)))
    cols.append(col)


# convert numpy of lch colors to bgr colors
bgr_cols = []
for j in cols:
    bgr_col = []
    for i in j: 
        bgr = convert_color(i, cv2.COLOR_RGB2BGR) 
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

# stack patches together 
abcd = np.vstack(result)   
print(abcd.shape) #(2000, 2000, 3)

abcd = cv2.putText(abcd, 'red: 0 ', (70, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
abcd = cv2.putText(abcd, 'no green: 0 ', (70, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255,255), 2)
abcd = cv2.putText(abcd, 'green: 255 ', (70, 1770), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
abcd = cv2.putText(abcd, 'no blue: 0', (70, 1870), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
abcd = cv2.putText(abcd, 'blue: 255', (1680, 1870), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255,255), 2)

# cv2.imshow(f'LAB Matrix', abcd)
import os
os.chdir(r'D:\thesis\output_images')
cv2.imwrite(f'RGB_Matrix_red0.jpg', abcd)


#%%

# Plot HSV Space in a Matrix 

def convert_color(col, conversion=cv2.COLOR_BGR2Lab): #default 
    """" converts BGR to LAB by default, supports all color spaces except lch 
    """
    color = cv2.cvtColor(np.array([[col]], dtype=np.float32), conversion)[0, 0]
    return color*255



PATCH = (100, 100, 3)
COLORS_WIDTH_COUNT = 20


# original CS: hsv 
h = np.linspace(0, 360, COLORS_WIDTH_COUNT) # hue: 0-360 
s = np.linspace(1, 1, COLORS_WIDTH_COUNT) # saturation: 100
v = np.linspace(0, 1, COLORS_WIDTH_COUNT) # value: 0-100

# create hsv colors (tuple-3)
cols = []
for l_el in v:
    vi = np.full(COLORS_WIDTH_COUNT, l_el)
    col = np.array(list(zip(h,s,vi)))
    cols.append(col)
# len(cols) 20 x 20 


# convert numpy of lch colors to bgr colors
bgr_cols = []
for j in cols:
    bgr_col = []
    for i in j: 
        bgr = convert_color(i, cv2.COLOR_HSV2BGR) 
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

# stack patches together 
abcd = np.vstack(result)   
print(abcd.shape) #(2000, 2000, 3)

#abcd = cv2.putText(abcd, 'saturation: 100 (full)', (70, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
#abcd = cv2.putText(abcd, 'value: 0 ', (70, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255,255), 2)
#abcd = cv2.putText(abcd, 'value: 100 ', (70, 1770), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
#abcd = cv2.putText(abcd, 'hue: 0', (70, 1870), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
#abcd = cv2.putText(abcd, 'hue: 360', (1680, 1870), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255,255), 2)

# cv2.imshow(f'LAB Matrix', abcd)
import os
os.chdir(r'D:\thesis\output_images')
cv2.imwrite(f'HSV_Matrix_s100_.jpg', abcd)


#%%

# Plot HSL Space in a Matrix 

def convert_color(col, conversion=cv2.COLOR_BGR2Lab): #default 
    """" converts BGR to LAB by default, supports all color spaces except lch 
    """
    color = cv2.cvtColor(np.array([[col]], dtype=np.float32), conversion)[0, 0]
    return color*255



PATCH = (100, 100, 3)
COLORS_WIDTH_COUNT = 20


# original CS: hls 
h = np.linspace(0, 360, COLORS_WIDTH_COUNT) # hue: 0-360 
l = np.linspace(0, 1, COLORS_WIDTH_COUNT) # luminance: 0-100
s = np.linspace(0, 0, COLORS_WIDTH_COUNT)  # saturation: 100

# create hsv colors (tuple-3)
cols = []
for l_el in l:
    li = np.full(COLORS_WIDTH_COUNT, l_el)
    col = np.array(list(zip(h,li,s)))
    cols.append(col)
# len(cols) 20 x 20 


# convert numpy of lch colors to bgr colors
bgr_cols = []
for j in cols:
    bgr_col = []
    for i in j: 
        bgr = convert_color(i, cv2.COLOR_HLS2BGR) 
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

# stack patches together 
abcd = np.vstack(result)   
print(abcd.shape) #(2000, 2000, 3)

abcd = cv2.putText(abcd, 'saturation: 0 (empty)', (70, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
abcd = cv2.putText(abcd, 'luminance: 0 ', (70, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255,255), 2)
abcd = cv2.putText(abcd, 'luminance: 100 ', (70, 1770), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
abcd = cv2.putText(abcd, 'hue: 0', (70, 1870), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
abcd = cv2.putText(abcd, 'hue: 360', (1680, 1870), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255,255), 2)

# cv2.imshow(f'LAB Matrix', abcd)
import os
os.chdir(r'D:\thesis\images')
cv2.imwrite(f'HLS_Matrix_s0.jpg', abcd)