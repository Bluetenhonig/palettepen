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



#%%


# VIAN's 28 colors 
# convert srgb, lab into hsv, hsl and lch values 

# import modules
import os
import matplotlib.pyplot as plt #2.1.2
import numpy as np
import pandas as pd
import cv2

# add new color name to color name dictionary 
# load new color name 
FOLDER_PATH = r'D:\thesis\input_images\google\ultramarine'
os.chdir(FOLDER_PATH)
col_decl = pd.read_csv('ultramarine.csv', index_col=0)

# load color name dictionary 
PATH = r'D:\thesis\input_color_name_dictionaries\thesaurus\datasets'
FILE = 'effcnd_thesaurus_vian.xlsx'
OUTPUT_FILE = 'effcnd_thesaurus_vian.xlsx'
os.chdir(PATH)
data = pd.read_excel(FILE, sep=" ", index_col=0)

#%%
# add new row to data set(for ultramarine)
new_row_id = data.shape[0]
col_decl.columns = ['cat1', 'srgb', 'cielab', 'hsv', 'LCH', 'hex']
new_row = col_decl[['cat1', 'srgb', 'cielab', 'hsv', 'LCH', 'hex']]
data = data.append(new_row.to_dict('records') , ignore_index=True)
# fill in what's missing for new color's row 
data["id"].iloc[new_row_id] = new_row_id
data["lang"].iloc[new_row_id] = "eng"
data["name"].iloc[new_row_id] = "ultramarine"

data['srgb_R'].iloc[-1] = int(data['srgb'].iloc[-1][1:3])
data['srgb_G'].iloc[-1] = int(data['srgb'].iloc[-1][7:9])
data['srgb_B'].iloc[-1] = int(data['srgb'].iloc[-1][12:16])

# fill up other cells with to fill up last row 
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


#%%

# color conversion from rgb to all other color spaces
import sys 
sys.path.append(r'D:\thesis\code')
import ColorConversion 
from ColorConversion import convert_color

# convert rgb into all other cs 
data_small = data[['srgb_R', 'srgb_G', 'srgb_B']]
basic_colors_rgb = data_small.values.tolist()

basic_colors_hsv  = []
basic_colors_hsl  = []
basic_colors_lch  = []
basic_colors_hex  = []

for i in basic_colors_rgb:
    rgb = np.array(i)
    rgbi = np.array([[rgb/ 255]], dtype=np.float32)
    hsv = cv2.cvtColor(rgbi, cv2.COLOR_RGB2HSV)
    hsv = hsv[0, 0]
    hsv = hsv.tolist()
    basic_colors_hsv.append(hsv)
    
for i in basic_colors_rgb:
    rgb = np.array(i)
    rgbi = np.array([[rgb/ 255]], dtype=np.float32)
    hsl = cv2.cvtColor(rgbi, cv2.COLOR_RGB2HLS) 
    hsl = hsl[0, 0]
    hsl = hsl.tolist()
    basic_colors_hsl.append(hsl)
    
for i in basic_colors_rgb:
    rgb = np.array(i)
    rgbi = np.array([[rgb/ 255]], dtype=np.float32)
    lab = cv2.cvtColor(rgbi, cv2.COLOR_RGB2Lab)
    lch = convert_color(lab, "LAB", "LCH")[0, 0]
    lch = lch.tolist()
    basic_colors_lch.append(lch)
    
for i in basic_colors_rgb:
    r,g,b = np.array(i)
    hex_val = convert_color((int(r),int(g),int(b)), "RGB", "HEX")
    basic_colors_hex.append(hex_val)
    
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
data['HEX'] = basic_colors_hex


#%% 

# save file 

os.chdir(PATH)
data.to_excel(OUTPUT_FILE, index=True)
