# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 19:42:03 2020

@author: lsamsi

Color Patch Visualization

!!! WARNING: Visuazlizations only happen in RGB !!!
For example, normalize hsv to 8-bit image range 0-1 for viewing 

"""

# import modules
import os
import matplotlib.pyplot as plt 
from matplotlib.colors import hsv_to_rgb

import numpy as np
import pandas as pd
import cv2


# declare variables
ITTEN_PATH = r'D:\thesis\input_color_name_dictionaries\system_ITTEN'
ITTEN12_RGB_FILE = 'rgbhsv_12.csv'
ITTEN12_LAB_FILE = 'lablchrgb_12_handcorrected.csv'
ITTEN6_RGB_FILE = 'rgbhsvlch_6.csv'

OUTPUT_PATH = r'D:\thesis\code\output_images'

#%%


# convert color 
def convert_color(col, conversion=cv2.COLOR_BGR2Lab): #default 
    color = cv2.cvtColor(np.array([[col]], dtype=np.float32), conversion)[0, 0]
    return color


def display_color(color, path, origin=None, save=False):
    """helper function: convert_color """
    if origin == 'BGR': 
        rgb_color = convert_color(color, cv2.COLOR_BGR2RGB)
    elif origin == 'LAB': 
        rgb_color = convert_color(color, cv2.COLOR_LAB2RGB)*255
    else: 
        rgb_color = color
    square = np.full((10, 10, 3), rgb_color, dtype=np.uint8) / 255.0
    plt.figure(figsize = (5,2))
    plt.imshow(square) 
    plt.axis('off')
    if save: 
        os.chdir(path)
        plt.savefig(f'{rgb_color}.png', transparent=True)
    plt.show() 


def convert_array(nparray, origin, target='RGB'): 
    """helper function: convert_color """
    rgb_colors = []
    for col in nparray: 
        if origin == 'BGR':        
            rgb_color = convert_color(col, cv2.COLOR_BGR2RGB)*255
        if origin == 'LAB':     
            rgb_color = convert_color(col, cv2.COLOR_LAB2RGB)*255
        if origin == 'HSV':     
            rgb_color = convert_color(col, cv2.COLOR_HSV2RGB)*255
        rgb_colors.append(rgb_color)
    return rgb_colors


def display_color_grid(palette, origin='RGB', colorbar_count=10):
    """helper function: convert_array, convert_color 
    - display color palette as bar """
    if origin == 'BGR':
        rgbcolors = convert_array(palette, 'BGR')
    if origin == 'LAB': 
        rgbcolors = convert_array(palette, 'LAB')
    if origin == 'HSV': 
        rgbcolors = convert_array(palette, 'HSV')
    x= 0
    for r in rgbcolors: 
        if len(rgbcolors[x:x+colorbar_count]) == colorbar_count:
            palette = np.array(rgbcolors[x:x+colorbar_count])[np.newaxis, :, :]
            plt.figure(figsize=(colorbar_count*2,5))
            plt.imshow(palette.astype('uint8'))
            plt.axis('off')
            plt.show()
            x += colorbar_count
        else: 
            if x == len(palette): 
                break
            else: 
                palette = np.array(rgbcolors[x:])[np.newaxis, :, :]
                plt.figure(figsize=(colorbar_count*2,2))
                plt.imshow(palette.astype('uint8'))
                plt.axis('off')
                plt.show()
                break


def visualize_one_rgbcolor(bgr_color, imagesize, use_cv2=False): 
    if use_cv2: 
        a = np.full(imagesize, bgr_color, dtype=np.uint8)
        print(a.shape) 
        cv2.imshow('A BGR-color', a)
    else: 
        fig = plt.figure(figsize = imagesize)
        plt.imshow(a) 
        plt.axis('off')

def show_three_rgbcolors(color1, color2, color3, rainbowtriple=None): 
    square_1 = np.full((10, 10, 3), color1, dtype=np.uint8) / 255.0
    square_2 = np.full((10, 10, 3), color2, dtype=np.uint8) / 255.0
    square_3 = np.full((10, 10, 3), color3, dtype=np.uint8) / 255.0  
    fig = plt.figure(figsize = (10,4))
    plt.subplot(3, 1, 1)
    plt.imshow(square_1) 
    plt.axis('off')
    plt.subplot(3, 1, 2)
    plt.imshow(square_2)
    plt.axis('off')
    plt.subplot(3, 1, 3)
    plt.imshow(square_3)
    plt.axis('off')
    if rainbowtriple==0: 
        txt_1 = 'Red: \nRGB: {color1} \nHSV: (0°,s,v)'
        txt_2 = 'Green: \nRGB: {color2} \nHSV: (120°,s,v)'
        txt_3 = 'Blue: \nRGB: {color3}  \nHSV: (240°,s,v)'
    if rainbowtriple==1:
        txt_1 = f'Yellow: \nRGB: {color1} \nHSV: (60°,s,v)'
        txt_2 = f'Cyan: \nRGB: {color2}  \nHSV: (180°,s,v)'
        txt_3 = f'Magenta: \nRGB: {color3}  \nHSV: (300°,s,v)'   
    if rainbowtriple==2:
        txt_1 = f'Orange: \nRGB: {color1} \nHSV: (30°,s,v)'
        txt_2 = f'Green-blue: \nRGB: {color2}  \nHSV: (150°,s,v)'
        txt_3 = f'Purple: \nRGB: {color3}  \nHSV: (270°,s,v)'
    if rainbowtriple==3: 
        txt_1 = f'Green-yellow: \nRGB: {color1} \nHSV: (90°,s,v)'
        txt_2 = f'Blue-yellow: \nRGB: {color2}  \nHSV: (210°,s,v)'
        txt_3 = f'Red-yellow: \nRGB: {color3}  \nHSV: (330°,s,v)'
    fig.text(.57, .76, txt_1)
    fig.text(.57, .49, txt_2)
    fig.text(.57, .23, txt_3)
    plt.suptitle('Color Patches in RGB', ha='left')
    plt.show()
    
def show_two_hsvcolors(hsv_color1, hsv_color2, use_cv2=False): 
    if use_cv2==False: 
        upper_square = np.full((10, 10, 3), hsv_color1, dtype=np.uint8) / 255.0
        lower_square = np.full((10, 10, 3), hsv_color2, dtype=np.uint8) / 255.0
        plt.subplot(1, 2, 1)
        plt.imshow(hsv_to_rgb(upper_square)) 
        plt.title('upper limit')
        plt.subplot(1, 2, 2)
        plt.imshow(hsv_to_rgb(lower_square))
        plt.title('lower limit')
        plt.suptitle('Color Patches in HSV')
        plt.show()
    else: 
        a = np.full((10, 10, 3), hsv_color1, dtype=np.uint8)
        b = np.full((10, 10, 3), hsv_color2, dtype=np.uint8)
        c = np.vstack((a, b))
        cv2.imshow('Two colors', c)
        
def show_two_bgrcolors(bgrcolor1, bgrcolor2, patchsize):  
    a = np.full(patchsize, bgrcolor1, dtype=np.uint8)
    b = np.full(patchsize, bgrcolor2, dtype=np.uint8)
    c = np.vstack((a, b))
    cv2.imshow('Two Colors', c)


def plot_itten12_hsvcolors(file, path, save=False):  
    squares = []
    for i in range(len(file)): 
        square = np.full((10, 10, 3), eval(file['RGB'][i]), dtype=np.uint8) / 255.0
        squares.append(square)
    fig = plt.figure(figsize = (30,5))
    for i in range(1,13):
        plt.subplot(1, 12, i)
        plt.imshow(squares[i-1]) 
        plt.axis('off')
        color = file['name'][i-1]
        rgb = file['RGB'][i-1] 
        hsv = file['HSV'][i-1] 
        plt.title(f'{color}', size=20)
    plt.suptitle('Color Patches in HSV in 30°-hue steps around the Color Wheel', size=40)
    if save: 
        os.chdir(path)
        fig.savefig('hsv_colorwheel_patches.png')
    plt.show()


def plot_itten12_lchcolors(file, path, save=False): 
    squares= []
    for i in range(len(file['RGB'])):
        color = eval(file['RGB'][i])
        square = np.full((10, 10, 3), color, dtype=np.uint8)
        squares.append(square) 
    fig = plt.figure(figsize = (30,5))    
    for i in range(1,13):  
        plt.subplot(1, 12, i)
        plt.imshow(squares[i-1]) 
        plt.axis('off')
        color = file['name'][i-1]
        plt.title(f'{color}', size=20)
    plt.suptitle('Color Patches in LCH in 30°-hue steps around the Color Wheel', size=40)
    if save: 
        os.chdir(path)
        fig.savefig('lch_colorwheel_patches.png')
    plt.show()

def plot_itten6_rgbcolors(file, path, save=False): 
    squares= []
    for i in range(len(file['RGB'])):
        color = eval(file['RGB'][i])
        square = np.full((10, 10, 3), color, dtype=np.uint8)
        squares.append(square)    
    fig = plt.figure(figsize = (20,5))    
    for i in range(1,7):  
        plt.subplot(1, 6, i)
        plt.imshow(squares[i-1]) 
        plt.axis('off')
        color = file['name'][i-1]
        plt.title(f'{color}', size=30)
    plt.suptitle('6 Basic Colors in RGB', size=40)
    if save: 
        os.chdir(path)
        fig.savefig('lch_6basiccolors_patches.png')
    plt.show()

   
#%%

if __name__ == '__main__': 

    display_color([22, 144, 23], OUTPUT_PATH, origin=None, save=False)
    
    # RGB
    # single colors
    visualize_one_rgbcolor([135, 173, 145], (190, 266, 3)) 
    
  
    # multiple colors
    show_three_rgbcolors((255,0,0) , (0,255,0), (0, 0, 255), rainbowtriple=0)
    show_three_rgbcolors((255,255,0), (0,255,255), (255, 0, 255), rainbowtriple=1)
    show_three_rgbcolors((255,128,0), (0,255,128), (128, 0, 255), rainbowtriple=2)
    show_three_rgbcolors((128,255,0), (0,128, 255), (255, 0, 128), rainbowtriple=3)
 
    # BGR
    show_two_bgrcolors([135, 173, 145], [135, 173, 25], (190, 266, 3))
    
    # HSV  
    light_orange = (18,255,255) 
    dark_orange = (1,190,200)
    light_white = (49.999916076660156, 0.31578928232192993, 0.2235294133424759)
    dark_white = (51.724082946777344, 0.4603172242641449, 0.24705882370471954)
    
    show_two_hsvcolors(light_orange, dark_orange, use_cv2=False)

    # 12 colors 
    
    # load data
    os.chdir(ITTEN_PATH)
    itten12_rgbhsv = pd.read_csv(ITTEN12_RGB_FILE)  
    itten12_lablchrgb = pd.read_csv(ITTEN12_LAB_FILE) 
       
    plot_itten12_hsvcolors(itten12_rgbhsv, OUTPUT_PATH, save=False)
    plot_itten12_lchcolors(itten12_lablchrgb, OUTPUT_PATH, save=False)
    
    # 6 colors
    itten6_rgbhsvlch = pd.read_csv(ITTEN6_RGB_FILE)
    
    plot_itten6_rgbcolors(itten6_rgbhsvlch, OUTPUT_PATH, save=False)



