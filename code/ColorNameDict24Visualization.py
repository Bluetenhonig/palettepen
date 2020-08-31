# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 09:11:42 2020

@author: lsamsi

After the original color name dictionary (CND) is sourced, 
it is saved in a folder with the source name as folder name. 

The CND needs to be extended to a fully-fledged color name dictionary (FFCND)
and then into an Extended-FFCND (EFFCND).

The preprocessing phase will extend the CND to a EFFCND dataframe.  

The preprocessing steps: 
    
1. original CND (from source) 
+ color conversion 
2. processed CND = FFCND
columns = [id, lang, name, srgb, srgb_r, srgb_g, srgb_b, hsv, hsv_h, hsv_s, hsv_v, lab, lab_l, lab_a, lab_b, hex]
filename = "ffcnd_"+source+".xlsx"
+ basic color classification 
3. processed FFCND = EFFCND (with 1 system of basic colors)
columns =  [id, lang, name, srgb, srgb_r, srgb_g, srgb_b, hsv, hsv_h, hsv_s, hsv_v, lab, lab_l, lab_a, lab_b, hex, cat1, cat2]
filename = "effcnd_"+source+"_"+system+".xlsx" 
  
"""
# load modules
import os
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from statistics import mean
import sys
sys.path.append(r'D:\thesis\code')
from ColorConversion00 import convert_color
from ColorConversion00 import *


# declare variables
PATH = r'D:\thesis\input_color_name_dictionaries\thesaurus\datasets'
FILE = 'effcnd_thesaurus_basicvian.xlsx'



#%%
def load_data(path, file):
    os.chdir(path)        
    data = pd.read_excel(file, sep=" ", index_col=0)
    data.info()
    data = data.dropna()
    data = data.reset_index()
    return data
    
def plot_color(color, size): 
    """ helper function for plot_feature_label_color and plot_color_names
    plot image """
    image = np.full((size, size, 3), color, dtype=np.uint8) / 255
    plt.imshow(image) 
    plt.axis('off')
    plt.show()

def plot_feature_label_color(data): 
    """ plot color name feature and basic color label 
    finding basic color's values by averaging 
    all color names belonging to that basic color"""
    for color_id in range(len(data['srgb'])):
        print(f'label: {color_id}')
        plot_color(eval(data['srgb'].iloc[color_id]), 10)
        basic_color = data['cat1'].iloc[color_id]
        print(f'feature: {color_id}')
        try: 
            plot_color(eval(data['srgb'][data['name']==basic_color].iloc[color_id]),10)
        except: 
            colsofcat = data['cielab'][data['cat1']==basic_color]
            len_colsofcat = len(colsofcat)
            ls_ = []
            as_ = []
            bs_ = []
            for i in range(len_colsofcat):
                r = eval(colsofcat.iloc[i])[0]
                ls_.append(r)
                g = eval(colsofcat.iloc[i])[1]
                as_.append(g)
                b = eval(colsofcat.iloc[i])[2]
                bs_.append(b)
            
            la = int(mean(ls_))
            aa = int(mean(as_))
            ba = int(mean(bs_))
            srgb = convert_color([la,aa,ba], "LAB", "RGB",lab2rgb)
            plot_color(srgb, 10)
        
def plot_color_names(data): 
    """ plot all color names in EFFCND """
    for i in range(len(data['srgb'])): 
        print(i)
        print(f'{data["name"][i]}')
        plot_color(eval(data['srgb'].iloc[i]), 10)


#%%

if __name__ == '__main__': 


    data = load_data(PATH, FILE)
    plot_color_names(data)
    plot_feature_label_color(data)





