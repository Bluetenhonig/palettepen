# -*- coding: utf-8 -*-
"""
Created on Tue May  5 13:40:42 2020

@author: Linda Samsinger 

Determine Color Class Center and save as Data set to Excel file: 
Each VIAN color category has many Color Thesaurus values.
Find the VIAN color class center by averaging over all Color Thesaurus values. 

"""

#import modules
import os
import sys
import numpy as np 
import pandas as pd
import timeit

sys.path.append(r'D:\thesis\code\04_Final')
from ColorConversion00 import convert_color
from ColorConversion00 import *

# declare variables 
SYSTEM = 'VIAN' 
PATH = r'D:\thesis\input_color_name_dictionaries\system_VIAN'
FILE = 'SRGBLABhsvhslLCHHEX_Eng_VIANHuesColorThesaurus.xlsx'
SAVE_PATH = r'D:\thesis\input_color_name_dictionaries\system_VIAN'
SAVE_FILE = "labbgr_vian_colors_avg2.csv"
CAT_COLUMN = 'VIAN_color_category'
RGB_COLUMN = 'srgb'
start_time = timeit.default_timer()

def get_color_category(data, label):   
    lst = data[label].unique()
    lst = lst.tolist()
    return lst

def get_bgr_lab_avg(data, label, col_rgb): 
    bgr_cols = []
    lab_cols = []
    for i in range(len(lst)): 
        avgcolcatrgb = np.mean([np.array(eval(n)) for n in data[col_rgb][data[label] ==lst[i]]], axis=0).tolist()
        rgb = list(np.array(avgcolcatrgb).astype(int))
        bgr = [int(el/255) for el in convert_color(rgb, "RGB", "BGR", rgb2bgr)]
        bgr_cols.append([np.round(l) for l in bgr])
        lab = convert_color(np.array(avgcolcatrgb)/255, "RGB", "LAB")
        lab_cols.append([np.round(l) for l in list(lab)])
    return bgr_cols, lab_cols
    


if __name__ == '__main__':
     # set directory 
    os.chdir(PATH)
    
    # load data 
    data = pd.read_excel(FILE, sep=" ", index_col=0)
    print(data[[CAT_COLUMN, RGB_COLUMN]].head())
    print(data.groupby(CAT_COLUMN)[RGB_COLUMN].count().sort_values())
   
    # get color category
    lst = get_color_category(data, CAT_COLUMN)
    # compute average of color category (bgr, lab) for given rgb
    bgr_cols, lab_cols = get_bgr_lab_avg(data, CAT_COLUMN, RGB_COLUMN)
    
    # resulting dataframe: color category, lab, bgr 
    avg = pd.DataFrame({f'{SYSTEM}_color': lst
                        , 'lab': lab_cols
                        , 'bgr': bgr_cols
                        })
    print(avg.head())
    
    # save to file
    os.chdir(SAVE_PATH)
    avg.to_csv(SAVE_FILE, index=0) 
    
    # code runtime 
    secs = timeit.default_timer() - start_time
    mins = secs/60
    #evaluates how long it took to run the code
    print('It took {:05.2f} minutes to run this code.'.format(mins) ) 