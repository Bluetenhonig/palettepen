# -*- coding: utf-8 -*-
"""
Created on Sun May 24 19:06:51 2020

@author: lsamsi


After the original color name dictionary (CND) is sourced, 
it is saved in a folder with the source name as folder name. 

The CND needs to be extended to a fully-fledged color name dictionary (FFCND)
and then into an extended FFCND (EFFCND).

The preprocessing will extend the CND to a EFFCND dataframe.  

The preprocessing steps: 
0. raw CND
+ English 
  
1. original CND (from source) 
+ color space values 

2. processed CND = FFCND
columns = [id, lang, name, srgb, srgb_r, srgb_g, srgb_b, hsv, hsv_h, hsv_s, hsv_v, lab, lab_l, lab_a, lab_b, hex]
filename = "ffcnd_"+source+".xlsx"
+ basic color classification 

3. processed FFCND = EFFCND (with 1 system of basic colors)
columns =  [id, lang, name, image, srgb, srgb_r, srgb_g, srgb_b, hsv, hsv_h, hsv_s, hsv_v, lab, lab_l, lab_a, lab_b, hex, cat1, cat2]
filename = "effcnd_"+source+"_"+system+".xlsx" 

Step Before: Original CND 
Goal: Extend CND to all color space values + add new color name 
Step After: Last Word Classification

"""

# import modules
import os
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import sys 
from timeit import default_timer as timer
sys.path.append(r'D:\thesis\code')
from ColorConversion00 import convert_color
from ColorConversion00 import *



PATH = r'D:\thesis\input_images\google\ultramarine'
FILE = 'ultramarine.csv'

DICT_PATH = r'D:\thesis\input_color_name_dictionaries\thesaurus\datasets'
DICT_FILE = 'effcnd_thesaurus_basicvian.xlsx'
DICT_FILE2 = 'eeffcnd_thesaurus_basicvian_upinterval.xlsx'

SAVE_PATH = r'D:\thesis\input_color_name_dictionaries\thesaurus\datasets'
SAVE_FILE = 'effcnd_thesaurus_basicvian.xlsx'
SAVE_FILE2 = 'eeffcnd_thesaurus_basicvian_upinterval.xlsx'

LABEL = 'cielab'

#%%

def add_new_row(data, col_decl): 
    """ add new row to data set, e.g. for ultramarine """
    
    new_row_id = data.shape[0]
    col_decl.columns = ['cat1', 'srgb', 'cielab', 'hsv', 'LCH', 'hex']
    new_row = col_decl[['cat1', 'srgb', 'cielab', 'hsv', 'LCH', 'hex']]
    data = data.append(new_row.to_dict('records') , ignore_index=True)

    data["id"].iloc[new_row_id] = new_row_id
    data["lang"].iloc[new_row_id] = "eng"
    data["name"].iloc[new_row_id] = "ultramarine"
    
    data['srgb_R'].iloc[-1] = int(data['srgb'].iloc[-1][1:3])
    data['srgb_G'].iloc[-1] = int(data['srgb'].iloc[-1][7:9])
    data['srgb_B'].iloc[-1] = int(data['srgb'].iloc[-1][12:16])

    data['cielab_L'].iloc[-1] = eval(data['cielab'].iloc[-1])[0]
    data['cielab_a'].iloc[-1] = eval(data['cielab'].iloc[-1])[1]
    data['cielab_b'].iloc[-1] = eval(data['cielab'].iloc[-1])[2]
    data['hsv_H'].iloc[-1] = eval(data['hsv'].iloc[-1])[0]
    data['hsv_S'].iloc[-1] = eval(data['hsv'].iloc[-1])[1]
    data['hsv_V'].iloc[-1] = eval(data['hsv'].iloc[-1])[2]
    data['hsl_H'].iloc[-1] = None
    data['hsl_S'].iloc[-1] = None
    data['hsl_L'].iloc[-1] = None
    data['LCH_L'].iloc[-1] = eval(data['LCH'].iloc[-1])[0]
    data['LCH_C'].iloc[-1] = eval(data['LCH'].iloc[-1])[1]
    data['LCH_H'].iloc[-1] = eval(data['LCH'].iloc[-1])[2]
    return data 
    
def add_row_to_data(dict_file, dict_path, new_row_file, path, save=False):
    """ add new row of color label category to dictionary """
    os.chdir(path)
    col_decl = pd.read_csv(new_row_file, index_col=0)
    print(col_decl.info())
    os.chdir(dict_path)
    data = pd.read_excel(dict_file, sep=" ", index_col=0)       
    data = add_new_row(data, col_decl)
    if save: 
        data.to_excel(dict_file, sep=" ", index_col=0)        
    return data 

def make_subset(data, dict_path, label): 
    os.chdir(dict_path)
    data = pd.read_excel(data, sep=" ", index_col=0) 
    data_c = data[data[label].notnull()]
    data = data[data[label].isnull()]
    data_small = data[['srgb_R', 'srgb_G', 'srgb_B']]
    basic_colors_rgb = data_small.values.tolist()
    return data, data_c, basic_colors_rgb


def convert_fromrgb_toall(data, basic_colors_rgb): 

    basic_colors_rgb = [[int(a),int(b),int(c)] for a,b,c in basic_colors_rgb]
    basic_colors_lab  = []
    basic_colors_hsv  = []
    basic_colors_hsl  = []
    basic_colors_lch  = []
    basic_colors_hex  = []
    
    for rgb in basic_colors_rgb:
        lab = convert_color(rgb, "RGB", "LAB", rgb2lab) 
        hsv = convert_color(rgb, "RGB", "HSV", rgb2hsv) 
        hsl = convert_color(rgb, "RGB", "HSL", rgb2hsl) 
        lch = convert_color(lab, "LAB", "LCH")
        r,g,b = np.array(rgb)
        hex_val = convert_color((int(r),int(g),int(b)), "RGB", "HEX") 
        
        basic_colors_lab.append(lab)
        basic_colors_hsv.append(hsv)
        basic_colors_hsl.append(hsl)
        basic_colors_lch.append(lch)  
        basic_colors_hex.append(hex_val)
        
    data['srgb_R'] = [i[0] for i in basic_colors_rgb]
    data['srgb_G'] = [i[1] for i in basic_colors_rgb]
    data['srgb_B'] = [i[2] for i in basic_colors_rgb]    
    data['hsv'] = basic_colors_hsv
    data['hsv_H'] = [i[0] for i in basic_colors_hsv]
    data['hsv_S'] = [i[1] for i in basic_colors_hsv]
    data['hsv_V'] = [i[2] for i in basic_colors_hsv]
    data['cielab'] = basic_colors_lab
    data['cielab_L'] = [i[0] for i in basic_colors_lab]
    data['cielab_a'] = [i[1] for i in basic_colors_lab]
    data['cielab_b'] = [i[2] for i in basic_colors_lab]
    data['hsl'] = basic_colors_hsl
    data['hsl_H'] = [i[0] for i in basic_colors_hsl]
    data['hsl_S'] = [i[1] for i in basic_colors_hsl]
    data['hsl_L'] = [i[2] for i in basic_colors_hsl]
    data['LCH'] = basic_colors_lch
    data['LCH_L'] = [i[0] for i in basic_colors_lch]
    data['LCH_C'] = [i[1] for i in basic_colors_lch]
    data['LCH_H'] = [i[2] for i in basic_colors_lch]
    data['hex'] = basic_colors_hex
    
    return data 

def convert_fromlab_toall(data, label):
    basic_colors_cielab = [eval(lab) for lab in data[label]]
    basic_colors_rgb  = []
    basic_colors_hsv  = []
    basic_colors_hsl  = []
    basic_colors_lch  = []
    basic_colors_hex  = []
    
    for lab in basic_colors_cielab:
        rgb = convert_color(lab, "LAB", "RGB", lab2rgb) 
        hsv = convert_color(rgb, "RGB", "HSV", rgb2hsv) 
        hsl = convert_color(rgb, "RGB", "HSL", rgb2hsl) 
        lch = convert_color(lab, "LAB", "LCH")
        r,g,b = np.array(rgb)
        hex_val = convert_color((int(r),int(g),int(b)), "RGB", "HEX") 
        
        basic_colors_rgb.append(rgb)
        basic_colors_hsv.append(hsv)
        basic_colors_hsl.append(hsl)
        basic_colors_lch.append(lch)  
        basic_colors_hex.append(hex_val)
        
    data['srgb'] = basic_colors_rgb
    data['srgb_R'] = [i[0] for i in basic_colors_rgb]
    data['srgb_G'] = [i[1] for i in basic_colors_rgb]
    data['srgb_B'] = [i[2] for i in basic_colors_rgb]    
    data['hsv'] = basic_colors_hsv
    data['hsv_H'] = [i[0] for i in basic_colors_hsv]
    data['hsv_S'] = [i[1] for i in basic_colors_hsv]
    data['hsv_V'] = [i[2] for i in basic_colors_hsv]
    data['hsl'] = basic_colors_hsl
    data['hsl_H'] = [i[0] for i in basic_colors_hsl]
    data['hsl_S'] = [i[1] for i in basic_colors_hsl]
    data['hsl_L'] = [i[2] for i in basic_colors_hsl]
    data['LCH'] = basic_colors_lch
    data['LCH_L'] = [i[0] for i in basic_colors_lch]
    data['LCH_C'] = [i[1] for i in basic_colors_lch]
    data['LCH_H'] = [i[2] for i in basic_colors_lch]
    data['hex'] = basic_colors_hex
    
    return data

def convert_allrows_fromrgb_toother(data, other_cs = 'HEX'): 
    basic_colors_rgb = data[['srgb_R', 'srgb_G', 'srgb_B']].values.tolist()
    basic_colors_rgb = [[int(a),int(b),int(c)] for a,b,c in basic_colors_rgb]
    if other_cs == 'HEX': 
        basic_colors_hex  = []
        for rgb in basic_colors_rgb:
            r,g,b = np.array(rgb)
            hex_val = convert_color((int(r),int(g),int(b)), "RGB", "HEX") 
            basic_colors_hex.append(hex_val)
        data['hex'] = basic_colors_hex
        return data 
    
    
#%%    
if __name__ == '__main__':
    
    # task: add new row (information about new color category) to data 
    data_new = add_row_to_data(DICT_FILE, DICT_PATH, FILE, PATH, save=False)

    # task: convert from RGB to all other cs where no LAB 
    data, data_c, basic_colors_rgb = make_subset(DICT_FILE2, DICT_PATH, LABEL)
    if data.empty == True:
        print('Nothing to convert.')
    
    data_all_cs = convert_fromrgb_toall(data, basic_colors_rgb)
    print(data_all_cs)
    
    # task: convert from LAB to all other cs where no other cs 
    data_c.info()
    data_c_nans = data_c[(data_c.isnull().sum(axis=1) >=2)]
    data_c_no_nans = pd.concat([data_c, data_c_nans]).drop_duplicates(keep=False)
    data_c_full = convert_fromlab_toall(data_c_nans, LABEL)

    data_c = pd.concat([data_c_no_nans, data_c_full]).reset_index(drop=True)
    data = pd.concat([data_c, data_all_cs]).reset_index(drop=True)

    # task: convert from all RGB to other cs
    data = convert_allrows_fromrgb_toother(data, other_cs = 'HEX')
    

#%% 

    # save file 
    os.chdir(SAVE_PATH)
    data.to_excel(SAVE_FILE2, index=True)
