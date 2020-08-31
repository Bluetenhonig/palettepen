# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 19:42:03 2020

@author: Linda Samsinger

Builds Dataframes: 12 Color Wheel Colors
Color spaces: RGB, HSV, LAB, LCH 
Goal: HSV is not uniform, LAB is uniform (perceptually)

"""

# import modules
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timeit

sys.path.append(r'D:\thesis\code\04_Final')
from ColorConversion00 import convert_color
from ColorConversion00 import *

# declare variables 
SAVE_PATH = r'D:\thesis\input_color_name_dictionaries\system_ITTEN'
SAVE_FILE = 'rgbhsv_12.csv'
SAVE_FILE2 = 'lablchrgb_12.csv'
start_time = timeit.default_timer()

#%%

# define functions

def make_rgb_hsv_colorwheel_colors(): 
    """ Color Wheel 30°-steps in different color spaces """
    # RGB-Colors
    lst1 = [(255,0,0) ,(255,128,0) ,(255,255,0),(128,255,0),(0,255,0),(0,255,128),(0,255,255),(0,128, 255),(0, 0, 255),(128, 0, 255),(255, 0, 255),(255, 0, 128)]
    lst2 = []
    lst3 = ['red', 'orange', 'yellow', 'green-yellow', 'green', 'green-blue', 'cyan', 'blue-yellow','blue','purple','magenta','red-yellow']
    # HSV-Colors
    for i in range(len(lst1)): 
        lst2.append(convert_color(lst1[i], "RGB", "HSV", rgb2hsv))
        
    df = pd.DataFrame()
    df['RGB'] = lst1
    df['HSV'] = lst2
    df['name'] = lst3
    return df 


def create_lab_lch_colorwheel_colors(): 
    """LCH-Color Wheel in 30°-steps
    get 12 hues of 30°-steps for L*CH's H-channel """
    lablch_twelve = dict()
    for i in np.linspace(0,360,num=12, endpoint=False): 
        lch_colors = (50,100,i)
        lab_colors = lab2lch(lch_colors)
        lablch_twelve[lch_colors] = lab_colors
    return lablch_twelve


def make_lab_lch_colorwheel_colors(lablch_twelve): 
    """ build pd frame with lab lch for 12 30°-step hues """
    lst = []
    lst2 = []
    lst3 = []
    lst4 = ['fuchsia', 'red', 'terracotta','olive', 'kelly','leaf', 'teal','atoll', 'azure','blue','purple','lilac']

    for i in lablch_twelve.items(): 
        # {'LCH': i[0]}
        lst.append(i[0]) 
        lst2.append(i[1])      
    for i in range(len(lst2)): 
        lst3.append(convert_color(lst2[i], "RGB", "LAB", rgb2lab))
        
    df = pd.DataFrame()
    df['LCH'] =lst
    df['Lab'] =lst2
    df['RGB'] = lst3
    df['name'] = lst4
    
    return df

#%%

if __name__ == '__main__':
    
    # make rgb and hsv color wheel colors
    df = make_rgb_hsv_colorwheel_colors()
    # save rgb-hsv color wheel dictionary
    os.chdir(SAVE_PATH)
    df.to_csv(SAVE_FILE)

#%%
    # make lab and lch color wheel colors from rgb 
    lablch_twelve = create_lab_lch_colorwheel_colors()
    df = make_rgb_hsv_colorwheel_colors(lablch_twelve)
    
    # save lab-lch color wheel dictionary 
    df.to_csv(SAVE_FILE2)

    # code runtime 
    secs = timeit.default_timer() - start_time
    mins = secs/60
    #evaluates how long it took to run the code
    print('It took {:05.2f} minutes to run this code.'.format(mins) ) 