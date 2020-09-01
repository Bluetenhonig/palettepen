# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 20:23:20 2020

@author: Anonym
"""

# import modules
import os
import numpy as np
import cv2 
import sys
sys.path.append(r'D:\thesis\code')
from ColorConversion00 import * 

# declare variables 
PATCH = (100, 100, 3)
COLORS_WIDTH_COUNT = 20


OUTPUT_FOLDER = r'D:\thesis\output_images'

#%%


def convert_color(col, conversion=cv2.COLOR_BGR2Lab):
    """" converts BGR to LAB by default supports all color spaces except lch """
    color = cv2.cvtColor(np.array([[col]], dtype=np.float32), conversion)[0, 0]
    return color*255

def four_color_matrix(bgrcolor1, bgrcolor2, bgrcolor3, bgrcolor4, patch): 
    a = np.full(patch, bgrcolor1, dtype=np.uint8)
    b = np.full(patch, bgrcolor2, dtype=np.uint8)
    c = np.full(patch, bgrcolor3, dtype=np.uint8)
    d = np.full(patch, bgrcolor4, dtype=np.uint8)
    ab = np.vstack((a, b))
    cd = np.vstack((c, d))
    abcd = np.hstack((ab, cd))
    print(a.shape) 
    print(ab.shape) 
    print(cd.shape) 
    print(abcd.shape)    
    cv2.imshow('Four Colors in a Matrix', abcd)


def space_color_matrix_lab(patch, colors_width_count, path, annotate=True, show=False, save=True): 
    l = np.linspace(0, 100, colors_width_count)
    a = np.linspace(0, 0, colors_width_count) 
    b = np.linspace(-128, 128, colors_width_count)
        
    cols = []
    for l_el in l:
        li = np.full(colors_width_count, l_el)
        col = np.array(list(zip(li,a,b)))
        cols.append(col)   
    bgr_cols = []
    for j in cols:
        bgr_col = []
        for i in j: 
            bgr = convert_color(i, cv2.COLOR_Lab2BGR)
            bgr_col.append(bgr)
        bgr_cols.append(bgr_col)   
    result = []
    for j in bgr_cols: 
        result_arr = []
        for i in j: 
            a = np.full(patch, i, dtype=np.uint8)
            result_arr.append(a)
        c = np.hstack(result_arr)
        result.append(c)
        
    abcd = np.vstack(result)   
    if annotate:     
        abcd = cv2.putText(abcd, 'b*: 128 (blue-yellow)', (70, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        abcd = cv2.putText(abcd, 'Luminance: 0 (dark)', (70, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        abcd = cv2.putText(abcd, 'Luminance: 100 (light)', (70, 1770), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        abcd = cv2.putText(abcd, 'a*: -128 (green)', (70, 1870), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        abcd = cv2.putText(abcd, 'a*: 128 (red)', (1680, 1870), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    if show: 
        cv2.imshow(f'LAB Matrix', abcd) 
    if save: 
        os.chdir(path)
        cv2.imwrite(f'LAB_Matrix_a0_.jpg', abcd)

def space_color_matrix_lch(patch, colors_width_count, path, annotate=True, show=False, save=True): 
    l = np.linspace(0, 100, colors_width_count) 
    c = np.linspace(100, 100, colors_width_count)
    h = np.linspace(0, 360, colors_width_count) 
    
    cols = []
    for l_el in l:
        li = np.full(colors_width_count, l_el)
        col = np.array(list(zip(li,c,h)))
        cols.append(col)
    
    bgr_cols = []
    for j in cols:
        bgr_col = []
        for i in j: 
            bgr = convert_color(lch2lab(i), cv2.COLOR_Lab2BGR) 
            bgr_col.append(bgr)
        bgr_cols.append(bgr_col)
    
    result = []
    for j in bgr_cols: 
        result_arr = []
        for i in j: 
            a = np.full(patch, i, dtype=np.uint8)
            result_arr.append(a)
        c = np.hstack(result_arr)
        result.append(c)        
    abcd = np.vstack(result)   
    if annotate: 
        abcd = cv2.putText(abcd, 'chroma: 0 (empty)', (70, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        abcd = cv2.putText(abcd, 'luminance: 0 (dark)', (70, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255,255), 2)
        abcd = cv2.putText(abcd, 'luminance: 100 (light)', (70, 1770), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        abcd = cv2.putText(abcd, 'hue: 0', (70, 1870), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        abcd = cv2.putText(abcd, 'hue: 360', (1680, 1870), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    if show: 
        cv2.imshow(f'LCH Matrix', abcd)
    if save:  
        os.chdir(path)
        cv2.imwrite(f'LCH_Matrix_c100_.jpg', abcd)


def space_color_matrix_rgb(patch, colors_width_count, path, annotate=True, show=False, save=True): 
    r = np.linspace(0, 0, colors_width_count)  
    g = np.linspace(0, 255, colors_width_count)
    b = np.linspace(0, 255, colors_width_count) 
    cols = []
    for l_el in g:
        gi = np.full(colors_width_count, l_el)
        col = np.array(list(zip(r,gi,b)))
        cols.append(col)

    bgr_cols = []
    for j in cols:
        bgr_col = []
        for i in j: 
            bgr = convert_color(i, cv2.COLOR_RGB2BGR) 
            bgr_col.append(bgr)
        bgr_cols.append(bgr_col)
    result = []
    for j in bgr_cols: 
        result_arr = []
        for i in j: 
            a = np.full(patch, i, dtype=np.uint8)
            result_arr.append(a)
        c = np.hstack(result_arr)
        result.append(c)

    abcd = np.vstack(result)   
    print(abcd.shape)
    if annotate: 
        abcd = cv2.putText(abcd, 'red: 0 ', (70, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        abcd = cv2.putText(abcd, 'no green: 0 ', (70, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255,255), 2)
        abcd = cv2.putText(abcd, 'green: 255 ', (70, 1770), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        abcd = cv2.putText(abcd, 'no blue: 0', (70, 1870), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        abcd = cv2.putText(abcd, 'blue: 255', (1680, 1870), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255,255), 2)
    if show:    
        cv2.imshow(f'RGB Matrix', abcd)
    if save: 
        os.chdir(path)
        cv2.imwrite(f'RGB_Matrix_red0.jpg', abcd)

def space_color_matrix_hsv(patch, colors_width_count, path, annotate=True, show=False, save=True):
    h = np.linspace(0, 360, colors_width_count) 
    s = np.linspace(1, 1, colors_width_count)
    v = np.linspace(0, 1, colors_width_count)
    
    cols = []
    for l_el in v:
        vi = np.full(colors_width_count, l_el)
        col = np.array(list(zip(h,s,vi)))
        cols.append(col)
    bgr_cols = []
    for j in cols:
        bgr_col = []
        for i in j: 
            bgr = convert_color(i, cv2.COLOR_HSV2BGR) 
            bgr_col.append(bgr)
        bgr_cols.append(bgr_col)
    result = []
    for j in bgr_cols: 
        result_arr = []
        for i in j: 
            a = np.full(patch, i, dtype=np.uint8)
            result_arr.append(a)
        c = np.hstack(result_arr)
        result.append(c)
    abcd = np.vstack(result)   
    print(abcd.shape) 
    
    if annotate: 
        abcd = cv2.putText(abcd, 'saturation: 100 (full)', (70, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        abcd = cv2.putText(abcd, 'value: 0 ', (70, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255,255), 2)
        abcd = cv2.putText(abcd, 'value: 100 ', (70, 1770), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        abcd = cv2.putText(abcd, 'hue: 0', (70, 1870), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        abcd = cv2.putText(abcd, 'hue: 360', (1680, 1870), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255,255), 2)
    if show:   
        cv2.imshow(f'HSV Matrix', abcd)
    if save: 
        os.chdir(path)
        cv2.imwrite(f'HSV_Matrix_s100_.jpg', abcd)

def space_color_matrix_hsl(patch, colors_width_count, path, annotate=True, show=False, save=True):
    h = np.linspace(0, 360, colors_width_count)
    l = np.linspace(0, 1, colors_width_count)
    s = np.linspace(0, 0, colors_width_count)    
    cols = []
    for l_el in l:
        li = np.full(colors_width_count, l_el)
        col = np.array(list(zip(h,li,s)))
        cols.append(col)    
    bgr_cols = []
    for j in cols:
        bgr_col = []
        for i in j: 
            bgr = convert_color(i, cv2.COLOR_HLS2BGR) 
            bgr_col.append(bgr)
        bgr_cols.append(bgr_col)   
    result = []
    for j in bgr_cols: 
        result_arr = []
        for i in j: 
            a = np.full(patch, i, dtype=np.uint8)
            result_arr.append(a)
        c = np.hstack(result_arr)
        result.append(c)
    
    abcd = np.vstack(result)    
    
    if annotate: 
        abcd = cv2.putText(abcd, 'saturation: 0 (empty)', (70, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        abcd = cv2.putText(abcd, 'luminance: 0 ', (70, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255,255), 2)
        abcd = cv2.putText(abcd, 'luminance: 100 ', (70, 1770), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        abcd = cv2.putText(abcd, 'hue: 0', (70, 1870), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        abcd = cv2.putText(abcd, 'hue: 360', (1680, 1870), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255,255), 2)  
    if show: 
        cv2.imshow(f'HLS Matrix', abcd)
    if save: 
        os.chdir(path)
        cv2.imwrite(f'HLS_Matrix_s0.jpg', abcd)
        
    
#%%
    

# Basic Matrix Color Plotting

    four_color_matrix([135, 173, 145], [135, 173, 25], [135, 120, 145], [85, 173, 25], PATCH, OUTPUT_FOLDER)
    
    space_color_matrix_lab(PATCH, COLORS_WIDTH_COUNT, OUTPUT_FOLDER)
    space_color_matrix_lch(PATCH, COLORS_WIDTH_COUNT, OUTPUT_FOLDER)
    space_color_matrix_rgb(PATCH, COLORS_WIDTH_COUNT, OUTPUT_FOLDER)
    space_color_matrix_hsv(PATCH, COLORS_WIDTH_COUNT, OUTPUT_FOLDER)
    space_color_matrix_hsl(PATCH, COLORS_WIDTH_COUNT, OUTPUT_FOLDER)

