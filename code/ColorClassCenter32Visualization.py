# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 20:23:20 2020

@author: Linda Samsinger

Basic Colors: 9 (Boynton-2), 28 (VIAN)
Color Space: HSV, LCH

A color space matrix is visualized using gridpoints. 
The basic color centers are marked as black dots on the matrix. 
The blackened basic color patches are annotated. 

"""

#import modules
import os
import sys
import numpy as np 
import pandas as pd
import cv2 
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
# hand-adjusted HSV and LCH values in dictionary 
basic_ten = {
              'red': {'rgb': [255,0,0], 'hsv': [0,1,1], 'lch': [50,100,40]}
            , 'orange': {'rgb': [255,128,0], 'hsv': [30,1,1], 'lch': [70,100,60]}
            , 'yellow': {'rgb': [255,255,0], 'hsv': [60,1,1], 'lch': [100,100,100]}
            , 'green': {'rgb': [0,255,0], 'hsv': [120,1,1], 'lch': [90,100,130]}
            , 'blue': {'rgb': [0,0,255], 'hsv': [240,1,1], 'lch': [30,100,300]}
            , 'pink': {'rgb': [255,0,128], 'hsv': [330,1,1], 'lch': [50,100,0]} # redefined it by looking at the end result color wheel, pink is usually rgb-defined differently
            , 'magenta': {'rgb': [255,0,255], 'hsv': [300,1,1], 'lch': [60,100,330]}
            , 'brown': {'rgb': [165,42,42], 'hsv': [0,1,.65], 'lch': [10,100,70]} # only brown cannot be found on the hsv-end result 
            , 'cyan': {'rgb': [0,255,255], 'hsv': [180,1,1], 'lch': [100,100,220]}
            , 'violet': {'rgb': [128,0,255], 'hsv': [270,1,1], 'lch': [40,100,310]}            
            }

start_time = timeit.default_timer()


#%%

def sample_color_space(min1, max1, div1, min2, max2, div2, min3, max3, div3): 
    h = np.linspace(min1, max1, div1) 
    s = np.linspace(min2, max2, div2)  
    v = np.linspace(min3, max3, div3) 
    return h,s,v 


def create_cols(channel1, channel2, channel3, space='lab'): 
    cols = []
    if space == 'lch': 
        cols = []
        for l_el in channel1:
            li = np.full(37, l_el)
            col = np.array(list(zip(li,channel2,channel3)))
            cols.append(col)
        return cols
    for l_el in channel3:
        vi = np.full(37, l_el)
        col = np.array(list(zip(channel1,channel2,vi)))
        cols.append(col)
    return cols


def center_as_black(cols, col_space): 
    # replace color class centers with black color 
    cols_blackds = []
    for c in cols:
        cols_blackd = []
        for color in c: 
            for key, value in basic_ten.items(): 
                # find basic colors in all hsv colors 
                if np.array_equal(np.array(basic_ten[key][col_space]), color): 
                    color = np.array([0,0,0])
                else: 
                    pass
            cols_blackd.append(color)
        cols_blackds.append(cols_blackd)
    return cols_blackds

def center_as_black_match(cols, col_space): 
    # replace color class centers with black color 
    matches = []
    cols_blackds = []
    for c in cols:
        cols_blackd = []
        for lch in c: 
            for i in df[col_space].tolist(): 
                # find basic colors in all hsv colors 
                if np.array_equal(np.array(eval(i)), lch): 
                    print("match")
                    matches.append(list(eval(i)))
                    lch = np.array([0,0,0])
                else: 
                    pass
            cols_blackd.append(lch)
        cols_blackds.append(cols_blackd)
    print(f"{len(matches)} matches found.")
    return cols_blackds, matches


def hsv_to_bgr(cols):  
    # convert numpy of hsv colors to bgr colors
    bgr_cols = []
    for j in cols:
        bgr_col = []
        for hsv in j:
            h, s, v = hsv
            rgb = convert_color((h,s*100,v*100), "HSV", "RGB", hsv2rgb) 
            bgr = convert_color(rgb, "RGB", "BGR", rgb2bgr)
            bgr_col.append(bgr)
        bgr_cols.append(bgr_col)
    # len(bgr_cols[0])  20 x 20 
    return bgr_cols

def lch_to_bgr(cols): 
    # convert numpy of lch colors to bgr colors
    bgr_cols = []
    for j in cols:
        bgr_col = []
        for lch in j: 
            lab = lch2lab(lch)
            bgr = convert_color(lab, "LAB", "BGR", lab2bgr) 
            bgr_col.append(bgr)
        bgr_cols.append(bgr_col)
    # len(bgr_cols[0])  20 x 20 
    return bgr_cols
    
    
def col_to_patches(cols, patch): 
    # put bgr colors into patches
    result = []
    for j in cols: 
        result_arr = []
        for i in j: 
            a = np.full(patch, i, dtype=np.uint8)
            result_arr.append(a)
        c = np.hstack(result_arr)
        result.append(c)
    return result


def annotate_hsv(img): 
    # annotate 8 basic colors 
    img = cv2.putText(img, 'red', (20, 1050), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2) # img, text, (x,y), font, size, text color, thickness
    img = cv2.putText(img, 'orange', (310, 1050), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255,255), 2)
    img = cv2.putText(img, 'yellow', (610, 1050), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2)
    img = cv2.putText(img, 'green', (1210, 1050), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2)
    img = cv2.putText(img, 'cyan', (1810, 1050), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255,255), 2)
    img = cv2.putText(img, 'blue', (2420, 1050), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    img = cv2.putText(img, 'violet', (2710, 1050), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    img = cv2.putText(img, 'magenta', (3000, 1050), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255,255), 2)
    img = cv2.putText(img, 'pink', (3320, 1050), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255,255), 2)

def annotate_lch(img): 
    # annotate with 8 basic colors 
    img = cv2.putText(img, 'red', (420, 550), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2) # img, text, (x,y), font, size, text color, thickness
    img = cv2.putText(img, 'orange', (610, 750), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255,255), 2)
    img = cv2.putText(img, 'yellow', (1010, 1050), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2)
    img = cv2.putText(img, 'brown', (710, 150), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255,255), 2)
    img = cv2.putText(img, 'green', (1310, 950), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2)
    img = cv2.putText(img, 'cyan', (2210, 1050), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255,255), 2)
    img = cv2.putText(img, 'blue', (3020, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    img = cv2.putText(img, 'violet', (3110, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    img = cv2.putText(img, 'magenta', (10, 550), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255,255), 2)
    img = cv2.putText(img, 'pink', (3320, 650), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255,255), 2)


def lab_to_lch(df, lab_column): 
    # convert lab to lch    
    lchs = []
    for lab in df['lab'].tolist(): 
        lch = lab2lch(eval(lab), h_as_degree = True)
        l,c,h = lch
        # round to the nearest tenth, chroma held constant at 100 
        lch = round(l,-1), 100.0, round(h,-1)
        lchs.append(str(list(lch)))   
    return lchs 

def get_duplicates(lchs):  
    # find duplicates 
    # print([x for n, x in enumerate(lchs) if x in lchs[:n]])
    # get indexes of duplicates 
    dupindx = [i for i, x in enumerate(lchs) if lchs.count(x) > 1]
    dupl_vians = ', '.join(df['vian_color'].iloc[dupindx].tolist())
    print(f"LAB VIAN color duplicates: {dupl_vians}.")
    return dupindx, dupl_vians


def remove_duplicates(dupindx, lchs, colors): 
    # remove other duplicates from the list
    for i in dupindx[1:]: 
        del lchs[i]
        del colors[i]
    return lchs, colors 

def get_matchcolors(matches):    
    matchcol = []
    for match in matches:  
        key = df['vian_color'][df['lch']==str(match)].iloc[0]
        matchcol.append(key)     
    matchcolors = list(zip(matches, matchcol))
    return matchcolors


#%%
    
if __name__ == '__main__':
    # basic colors in black (exclude brown): 9
    
    # Plot HSV Space in a Matrix
    #declare variables
    PATCH = (100, 100, 3)
    SAVE_PATH = r'D:\thesis\output_images\FINAL'
    SAVE_FILE = 'HSV_Wheel_Colors2.jpg'
    
    # get hsv gridpoints
    h,s,v = sample_color_space(0, 360, 37, 1, 1, 37, 0, 1, 11)
    hsv_cols = create_cols(h,s,v)
    # replace color centers with black color
    cols_has_blackds = center_as_black(hsv_cols, 'hsv')
    col_with_blackds_bgr = hsv_to_bgr(cols_has_blackds)
    # put colors into patches
    result = col_to_patches(col_with_blackds_bgr, PATCH)
    # stack patches together to image
    img = np.vstack(result)   
    print(img.shape)
    # annotate image
    annotate_hsv(img)
    
    #show image
    # cv2.imshow(SAVE_FILE, img)
    
    #save iamge
    os.chdir(SAVE_PATH)
    cv2.imwrite(SAVE_FILE, img)

#%%
    # basic colors in black (exclude brown): 9

    # Plot LCH Space in a Matrix 
    
    # declare variables  
    PATCH = (100, 100, 3)
    SAVE_PATH = r'D:\thesis\output_images\FINAL'
    SAVE_FILE = f'LCH_Matrix_Colors.jpg'
    
    
    # get lch gridpoints
    l, c, h = sample_color_space(0, 100, 11, 100, 100, 37, 0, 360, 37)
    lch_cols = create_cols(l,c,h)
    # replace color centers with black color
    cols_has_blackds = center_as_black(lch_cols, 'lch')  
    col_with_blackds_bgr = lch_to_bgr(cols_has_blackds)
    # put colors into patches
    result = col_to_patches(col_with_blackds_bgr, PATCH)
    # stack patches together 
    img = np.vstack(result)   
    print(img.shape)
    # annotate image
    annotate_lch(img)
    
    #show image
    # cv2.imshow(SAVE_FILE, img)
    
    #save iamge
    os.chdir(SAVE_PATH)
    cv2.imwrite(SAVE_FILE, img)



#%%
    ### VIAN COLORS in LCH ###
    # VIAN basic colors in black: 28 
    
    # Plot LCH Space in a Matrix
    
    # declare variables 
    PATH = r'D:\thesis\input_color_name_dictionaries\system_VIAN'
    FILE = 'labbgr_vian_colors_avg.csv'
    SAVE_PATH = r'D:\thesis\output_images\FINAL'
    SAVE_FILE = 'LCH_Matrix_VIAN_Colors.jpg'
    PATCH = (100, 100, 3)
    
    
    os.chdir(PATH)
    df = pd.read_csv(FILE)
    
    print(df.columns)
    print(df.head)

#%%
    # preprocessing
    # convert lab to lch values
    lchs = lab_to_lch(df, 'lab')  
    df['lch'] = lchs 
    # remove duplicates
    dupindx, dupl_vians = get_duplicates(lchs)   
    color_cats = df['vian_color'].tolist()
    lchs, color_cats = remove_duplicates(dupindx, lchs, color_cats)
    # make dictionary: color category - lch value
    col2lch = dict()
    
    for i, col in enumerate(color_cats): 
        col2lch[col] = lchs[i]


#%%
    # get lch gridpoints 
    l, c, h = sample_color_space(0, 100, 11, 100, 100, 37, 0, 360, 37)
    lch_cols = create_cols(l,c,h, space='lch')
    # replace color centers with black color
    cols_has_blackds, matches = center_as_black_match(lch_cols, 'lch')
    col_with_blackds_bgr = lch_to_bgr(cols_has_blackds) 
    
    # put colors into patches
    result = col_to_patches(col_with_blackds_bgr, PATCH)
    # stack patches together 
    img = np.vstack(result)   
    print(img.shape)
    # annotate VIAN colors separately using matchcolors 
    matchcolors = get_matchcolors(matches)
    print(matchcolors)
  
    
    #show image
    # cv2.imshow(SAVE_FILE, img)
    
    #save iamge
    os.chdir(SAVE_PATH)
    cv2.imwrite(SAVE_FILE, img)
    
    # code runtime 
    secs = timeit.default_timer() - start_time
    mins = secs/60
    #evaluates how long it took to run the code
    print('It took {:05.2f} minutes to run this code.'.format(mins) ) 






