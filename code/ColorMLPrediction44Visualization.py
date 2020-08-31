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

Plots LAB Space in a Matrix with annotated VIAN color categories
visualize a and b at constant luminance = 100 

"""


# import modules 

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import sys
sys.path.append(r'D:\thesis\code\04_Final')
from ColorConversion00 import *
from ColorConversion00 import convert_color

#%%
# declare variables 


PATH = r'D:\thesis\input_test_gridpoints'
PATH2 = r'D:\thesis\input_color_name_dictionaries\system_VIAN'
FILE = 'lab_vian_colors_testcolors216.xlsx'
FILE2 = 'labbgr_vian_colors_avg.xlsx'
FEATURE = "cielab" 
FEATURE2 = "lab"
LABEL = 'VIAN_color_category'
LABEL2 = 'vian_color'

DATA = 'VIAN'
ANNOTATE_COLORS = "testcolors" # testcolors or classcenter
LUMINANCE = 0
PATCH = (100, 100, 3)

SAVE_PATH = r'D:\thesis\output_images'
SAVE_FILE = f'LAB_{DATA}_testcolors216_l{LUMINANCE}_label_{ANNOTATE_COLORS}.jpg'

#%%
def make_lab_colors(data, feature, feature2, annotate_colors): 
    if annotate_colors == "classcenter": 
        l = [eval(t)[0] for t in data[feature2].tolist()]
        a = [eval(t)[1] for t in data[feature2].tolist()] 
        b = [eval(t)[2] for t in data[feature2].tolist()]
        l2 = [eval(t)[0] for t in data[feature].tolist()]
    elif annotate_colors == "testcolors": 
        l = [eval(t)[0] for t in data[feature].tolist()]
        a = [eval(t)[1] for t in data[feature].tolist()] 
        b = [eval(t)[2] for t in data[feature].tolist()]
        l2 = None
    return l, a, b, l2

def make_matrix_colors(luminance_fixed, l, a, b, labels, l2=None):   
    ll = []
    al = []
    bl = []
    lumlab = []
    
    for i in range(len(l)):
        if l[i] == luminance_fixed: 
            ll.append(l[i])
            al.append(a[i])
            bl.append(b[i]) 
            lumlab.append(labels[i])
    return ll, al, bl, lumlab

def make_matrix(colors): 
    matrix = []   
    matrix.append(colors[:6])
    matrix.append(colors[6:12]) 
    matrix.append(colors[12:18]) 
    matrix.append(colors[18:24]) 
    matrix.append(colors[24:30]) 
    matrix.append(colors[30:36]) 
    return matrix

def lab2bgr(matrix): 
    """convert numpy of lab colors to bgr colors"""
    bgr_cols = []
    for j in matrix:
        bgr_col = []
        for lab in j: 
            print(lab)
            rgb = convert_color(lab, "LAB", "RGB", lab2rgb)
            bgr = convert_color(rgb, "RGB", "BGR", rgb2bgr)
            bgr_col.append(bgr)
        bgr_cols.append(bgr_col)
    return bgr_cols


def bgr2patches(bgr_cols): 
    """ put bgr colors into patches """
    result = []
    for j in bgr_cols: 
        result_arr = []
        for i in j: 
            a = np.full(PATCH, i, dtype=np.uint8)
            result_arr.append(a)
        c = np.hstack(result_arr)
        result.append(c)
    return result 

def annotate_abcd(abcd, lumlab): 
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    FONT_COLOR = BLACK 
    abcd = cv2.putText(abcd, f'{lumlab[:6]}', (0, 50), cv2.FONT_HERSHEY_SIMPLEX, .8, FONT_COLOR, 1)
    abcd = cv2.putText(abcd, f'{lumlab[6:12]}', (0, 150), cv2.FONT_HERSHEY_SIMPLEX, .8, FONT_COLOR, 1)
    abcd = cv2.putText(abcd, f'{lumlab[12:18]}', (0, 250), cv2.FONT_HERSHEY_SIMPLEX, .9, FONT_COLOR, 1)
    abcd = cv2.putText(abcd, f'{lumlab[18:24]}', (0, 350), cv2.FONT_HERSHEY_SIMPLEX, .95, FONT_COLOR, 1)
    abcd = cv2.putText(abcd, f'{lumlab[24:30]}', (0, 450), cv2.FONT_HERSHEY_SIMPLEX, .85, FONT_COLOR, 1)
    abcd = cv2.putText(abcd, f'{lumlab[30:36]}', (0, 550), cv2.FONT_HERSHEY_SIMPLEX, .85, FONT_COLOR, 1)
    return abcd

#%%    
if __name__ == '__main__':
   
    
    # load data 
    os.chdir(PATH)
    testcolors = pd.read_excel(FILE)
    print(testcolors.head())
    os.chdir(PATH2)
    classcenter = pd.read_excel(FILE2, index_col=[0])
    print(classcenter.head())
    
    
    # merge data
    data = testcolors.merge(classcenter, left_on='VIAN_color_category', right_on='vian_color', how='left')
    print(data.head())
    
#%%
    length = len(data[LABEL].tolist())
    labels = data[LABEL].tolist()
    print(length, labels)
    
    l, a, b, l2 = make_lab_colors(data, FEATURE, FEATURE2, ANNOTATE_COLORS)
    ll, al, bl, lumlab = make_matrix_colors(LUMINANCE, l, a, b, labels, l2)

    li = np.full(36, ll[0])
    col = np.array(list(zip(li,al,bl)))
    
    matrix = make_matrix(col.tolist())
    bgr_cols = lab2bgr(matrix)
    result = bgr2patches(bgr_cols)
     
    abcd = np.vstack(result)   
    print(abcd.shape) 
    
    abcd = annotate_abcd(abcd, lumlab)

    # show matrix
    # cv2.imshow(f'LAB Matrix', abcd)
    
    # save matrix
    os.chdir(SAVE_PATH)
    cv2.imwrite(SAVE_FILE, abcd)