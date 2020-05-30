# -*- coding: utf-8 -*-
"""
Created on Tue May  5 13:40:42 2020

@author: lsamsi

Determine Color Class Center: 
Each VIAN color category has many Color Thesaurus values.
Find the VIAN color class center by averaging over all Color Thesaurus values. 

"""
import os
import pandas as pd
import cv2

# current directory to add to PATH to make other files importable as modules 

# basic function for color conversion 
def convert_color(col, conversion=cv2.COLOR_BGR2Lab): #default 
    """" converts BGR to LAB by default supports all color spaces except lch 
    """
    color = cv2.cvtColor(np.array([[col]], dtype=np.float32), conversion)[0, 0]
    return color

print(convert_color((0.5,0.2,0.3), cv2.COLOR_RGB2Lab))

#%%

########### VIAN Colors ###########
PATH = r'D:\thesis\code\pd28vianhues'
FILE = 'SRGBLABhsvhslLCHHEX_EngAll_VIANHuesColorThesaurus.xlsx'
PATCH = (200, 200, 3)

# set directory 
os.chdir(PATH)

# load data 
data = pd.read_excel(FILE, sep=" ", index_col=0)
    
# convert numpy of srgb colors to bgr colors
lst = data['VIAN_color_category'].unique()
lst = lst.tolist()
print(f"There are {len(lst)} color categories.")

bgr_cols = []
lab_cols = []
for i in range(len(lst)):
    # take the average 
    avgcolcatrgb = np.mean([np.array(eval(n)) for n in data['srgb'][data['VIAN_color_category'] ==lst[i]]], axis=0).tolist()
    bgr = convert_color(np.array(avgcolcatrgb), cv2.COLOR_RGB2BGR)
    bgr_cols.append([np.round(l) for l in bgr.tolist()])
    lab = convert_color(np.array(avgcolcatrgb)/255, cv2.COLOR_RGB2Lab)
    lab_cols.append([np.round(l) for l in lab.tolist()])


# save to file
os.chdir(r'D:\thesis\code\pd28vianhues')
avg = pd.DataFrame({'vian_color': lst
                    , 'lab': lab_cols
                    , 'bgr': bgr_cols
                    })
    
avg.to_csv("labbgr_vian_colors_avg.csv", index=0) 