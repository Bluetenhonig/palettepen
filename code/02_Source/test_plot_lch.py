# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 20:23:20 2020

@author: Anonym
"""

import numpy as np
import cv2 

#%%

# Plot 4 Colors in a Matrix 

# specify two bgr colors 
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

print(convert_color((40,114,101), cv2.COLOR_Lab2BGR))

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
a = np.linspace(-128, -128, COLORS_WIDTH_COUNT) 
b = np.linspace(-128, 128, COLORS_WIDTH_COUNT)

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
        a = np.full(patch, i, dtype=np.uint8)
        result_arr.append(a)
    c = np.hstack(result_arr)
    result.append(c)
    
abcd = np.vstack(result)   
print(abcd.shape) #(2000, 2000, 3)

abcd = cv2.putText(abcd, 'a*: -128 (green-red)', (70, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
abcd = cv2.putText(abcd, 'Luminance: 0 (dark)', (70, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
abcd = cv2.putText(abcd, 'Luminance: 100 (light)', (70, 1770), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
abcd = cv2.putText(abcd, 'b*: -128 (blue)', (70, 1870), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
abcd = cv2.putText(abcd, 'b*: 128 (yellow)', (1680, 1870), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

# cv2.imshow(f'LAB Matrix', abcd)
cv2.imwrite(f'LAB_Matrix_a-128.jpg', abcd)


#%%

# Plot LCH Space in a Matrix 

#%%

# Plot RGB Space in a Matrix 

#%%

# Plot HSV Space in a Matrix 
