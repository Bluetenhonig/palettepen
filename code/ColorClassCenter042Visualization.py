# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 20:23:20 2020

@author: Linda Samsinger

Plot epicenters of color classes in HSV/HSL and LCH space. 
We take Boynton's 11 color categories subtracting gray, brown, black and white. 
There remain 8 basic colors: orange, brown, pink, magenta, yellow, green, red, blue 
"""

# load modules 
import numpy as np
import cv2 

#%%
#################
### BASIC TEN ###
################# 

# to define Samsinger's 10 basic color epicenters
# based on RGB definition of these colors and conversion, see dataframe constructed in ColorConversion.py 

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



#%%

# Plot HSV Space in a Matrix 

# only 9 out of 10 basic colors possible in HSV: brown is excluded 

def convert_color(col, conversion=cv2.COLOR_BGR2Lab): #default 
    """" converts BGR to LAB by default, supports all color spaces except lch 
    """
    color = cv2.cvtColor(np.array([[col]], dtype=np.float32), conversion)[0, 0]
    return color*255



PATCH = (100, 100, 3)


# original CS: hsv 
h = np.linspace(0, 360, 37) # hue: 0-360 
s = np.linspace(1, 1, 37) # constant saturation: 100 - exclude 'brown' 
v = np.linspace(0, 1, 11) #  value: 0-100

# create hsv colors (tuple-3)
cols = []
for l_el in v:
    vi = np.full(37, l_el)
    col = np.array(list(zip(h,s,vi)))
    cols.append(col)
# len(cols[0]) 11 x 37 

# replace color class centers with black color 
cols_blackds = []
for c in cols:
    cols_blackd = []
    for hsv in c: 
        for key, value in basic_ten.items(): 
            # find basic colors in all hsv colors 
            if np.array_equal(np.array(basic_ten[key]['hsv']), hsv): 
                hsv = np.array([0,0,0])
            else: 
                pass
        cols_blackd.append(hsv)
    cols_blackds.append(cols_blackd)

  
# convert numpy of hsv colors to bgr colors
bgr_cols = []
for j in cols_blackds:
    bgr_col = []
    for i in j: 
        bgr = convert_color(i, cv2.COLOR_HSV2BGR) 
        bgr_col.append(bgr)
    bgr_cols.append(bgr_col)
# len(bgr_cols[0])  20 x 20 



# put bgr colors into patches
result = []
for j in bgr_cols: 
    result_arr = []
    for i in j: 
        a = np.full(PATCH, i, dtype=np.uint8)
        result_arr.append(a)
    c = np.hstack(result_arr)
    result.append(c)

# stack patches together 
abcd = np.vstack(result)   
print(abcd.shape) #(3700, 100, 3)

# annotate with 8 basic colors 
abcd = cv2.putText(abcd, 'red', (20, 1050), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2) # abcd, text, (x,y), font, size, text color, thickness
abcd = cv2.putText(abcd, 'orange', (310, 1050), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255,255), 2)
abcd = cv2.putText(abcd, 'yellow', (610, 1050), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2)
abcd = cv2.putText(abcd, 'green', (1210, 1050), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2)
abcd = cv2.putText(abcd, 'cyan', (1810, 1050), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255,255), 2)
abcd = cv2.putText(abcd, 'blue', (2420, 1050), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
abcd = cv2.putText(abcd, 'violet', (2710, 1050), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
abcd = cv2.putText(abcd, 'magenta', (3000, 1050), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255,255), 2)
abcd = cv2.putText(abcd, 'pink', (3320, 1050), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255,255), 2)

# cv2.imshow(f'LAB Matrix', abcd)
import os
os.chdir(r'D:\thesis\images')
cv2.imwrite(f'HSV_Wheel_Colors2.jpg', abcd)



#%%

# Plot LCH Space in a Matrix 
def convert_color(col, conversion=cv2.COLOR_BGR2Lab): #default 
    """" converts BGR to LAB by default supports all color spaces except lch 
    """
    color = cv2.cvtColor(np.array([[col]], dtype=np.float32), conversion)[0, 0]
    return color*255

def lch2lab(lch, h_as_degree = True):
    """
    :param lch: np.ndarray (dtype:float32), l : {0, ..., 100}, c : {0, ..., 128}, h : {0, ..., 2*PI}  
    :return: lab: np.ndarray (dtype:float32) l : {0, ..., 100}, a : {-128, ..., 128}, l : {-128, ..., 128}
    """
    if not isinstance(lch, np.ndarray):
        lch = np.array(lch, dtype=np.float32)
    lab = np.zeros_like(lch, dtype=np.float32)
    lab[..., 0] = lch[..., 0]
    if h_as_degree:
        lch[..., 2] = lch[..., 2] *np.pi / 180
    lab[..., 1] = lch[..., 1]*np.cos(lch[..., 2])
    lab[..., 2] = lch[..., 1]*np.sin(lch[..., 2])
    return lab


PATCH = (100, 100, 3)

# original CS: lch 
l = np.linspace(0, 100, 11) # luminance dark-to-light: 0-100 
c = np.linspace(100, 100, 37) # constant chroma: 100
h = np.linspace(0, 360, 37) # hue angles around h-channel: 0-360

# create lch colors (tuple-3)
cols = []
for l_el in l:
    li = np.full(37, l_el)
    col = np.array(list(zip(li,c,h)))
    cols.append(col)
# len(cols[0]) 11 x 37 

# replace color class centers with black color 
cols_blackds = []
for c in cols:
    cols_blackd = []
    for lch in c: 
        for key, value in basic_ten.items(): 
            # find basic colors in all hsv colors 
            if np.array_equal(np.array(basic_ten[key]['lch']), lch): 
                lch = np.array([0,0,0])
            else: 
                pass
        cols_blackd.append(lch)
    cols_blackds.append(cols_blackd)

# convert numpy of lch colors to bgr colors
bgr_cols = []
for j in cols_blackds:
    bgr_col = []
    for i in j: 
        bgr = convert_color(lch2lab(i), cv2.COLOR_Lab2BGR) 
        bgr_col.append(bgr)
    bgr_cols.append(bgr_col)
# len(bgr_cols[0])  20 x 20 


# put bgr colors into patches
result = []
for j in bgr_cols: 
    result_arr = []
    for i in j: 
        a = np.full(PATCH, i, dtype=np.uint8)
        result_arr.append(a)
    c = np.hstack(result_arr)
    result.append(c)
    
abcd = np.vstack(result)   
print(abcd.shape) #(1100, 3700, 3)

# annotate with 8 basic colors 
abcd = cv2.putText(abcd, 'red', (420, 550), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2) # abcd, text, (x,y), font, size, text color, thickness
abcd = cv2.putText(abcd, 'orange', (610, 750), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255,255), 2)
abcd = cv2.putText(abcd, 'yellow', (1010, 1050), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2)
abcd = cv2.putText(abcd, 'brown', (710, 150), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255,255), 2)
abcd = cv2.putText(abcd, 'green', (1310, 950), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2)
abcd = cv2.putText(abcd, 'cyan', (2210, 1050), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255,255), 2)
abcd = cv2.putText(abcd, 'blue', (3020, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
abcd = cv2.putText(abcd, 'violet', (3110, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
abcd = cv2.putText(abcd, 'magenta', (10, 550), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255,255), 2)
abcd = cv2.putText(abcd, 'pink', (3320, 650), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255,255), 2)

# cv2.imshow(f'LAB Matrix', abcd)
import os
os.chdir(r'D:\thesis\images')
cv2.imwrite(f'LCH_Matrix_Colors.jpg', abcd)


#%%
###################
### VIAN COLORS ###
###################

import os 
import pandas as pd

os.chdir(r'D:\thesis\code\pd28vianhues')
df = pd.read_csv('labbgr_vian_colors_avg.csv')

df.columns

print(df.info()) 



#%%

# Plot LCH Space in a Matrix 
def convert_color(col, conversion=cv2.COLOR_BGR2Lab): #default 
    """" converts BGR to LAB by default supports all color spaces except lch 
    """
    color = cv2.cvtColor(np.array([[col]], dtype=np.float32), conversion)[0, 0]
    return color*255

def lch2lab(lch, h_as_degree = True):
    """
    :param lch: np.ndarray (dtype:float32), l : {0, ..., 100}, c : {0, ..., 128}, h : {0, ..., 2*PI}  
    :return: lab: np.ndarray (dtype:float32) l : {0, ..., 100}, a : {-128, ..., 128}, l : {-128, ..., 128}
    """
    if not isinstance(lch, np.ndarray):
        lch = np.array(lch, dtype=np.float32)
    lab = np.zeros_like(lch, dtype=np.float32)
    lab[..., 0] = lch[..., 0]
    if h_as_degree:
        lch[..., 2] = lch[..., 2] *np.pi / 180
    lab[..., 1] = lch[..., 1]*np.cos(lch[..., 2])
    lab[..., 2] = lch[..., 1]*np.sin(lch[..., 2])
    return lab

def lab2lch(lab, h_as_degree = False):
    """
    :param lab: np.ndarray (dtype:float32) l : {0, ..., 100}, a : {-128, ..., 128}, l : {-128, ..., 128}
    :return: lch: np.ndarray (dtype:float32), l : {0, ..., 100}, c : {0, ..., 128}, h : {0, ..., 2*PI}
    """
    if not isinstance(lab, np.ndarray):
        lab = np.array(lab, dtype=np.float32)
    lch = np.zeros_like(lab, dtype=np.float32)
    lch[..., 0] = lab[..., 0]
    lch[..., 1] = np.linalg.norm(lab[..., 1:3], axis=len(lab.shape) - 1)
    lch[..., 2] = np.arctan2(lab[..., 2], lab[..., 1])
    lch[..., 2] += np.where(lch[..., 2] < 0., 2 * np.pi, 0)
    if h_as_degree:
        lch[..., 2] = (lch[..., 2] / (2*np.pi)) * 360
    return lch

#%%

# convert lab to lch    
lchs = []
for lab in df['lab'].tolist(): 
    lch = lab2lch(eval(lab), h_as_degree = True)
    l,c,h = lch
    # round to the nearest tenth, chroma held constant at 100 
    lch = round(l,-1), 100.0, round(h,-1)
    lchs.append(str(list(lch)))
df['lch'] = lchs

# find duplicates 
print([x for n, x in enumerate(lchs) if x in lchs[:n]])
# get indexes of duplicates 
dupindx = [i for i, x in enumerate(lchs) if lchs.count(x) > 1]
dupl_vians = ', '.join(df['vian_color'].iloc[dupindx].tolist())
print(f"LAB VIAN color duplicates: {dupl_vians}.")
# remove other duplicate from the list

del lchs[3]
del lchs[18]

colors = df['vian_color'].tolist()
del colors[3]
del colors[18]

col2lch = dict()
for i, col in enumerate(colors): 
    col2lch[col] = lchs[i]

#%%
PATCH = (100, 100, 3)

# original CS: lch 
l = np.linspace(0, 100, 11) # luminance dark-to-light: 0-100 
c = np.linspace(100, 100, 37) # constant chroma: 100
h = np.linspace(0, 360, 37) # hue angles around h-channel: 0-360

# create lch colors (tuple-3)
cols = []
for l_el in l:
    li = np.full(37, l_el)
    col = np.array(list(zip(li,c,h)))
    cols.append(col)
# len(cols[0]) 11 x 37 

# replace color class centers with black color 
matches = []
cols_blackds = []
for c in cols:
    cols_blackd = []
    for lch in c: 
        for i in df['lch'].tolist(): 
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

matchcol = []
for match in matches:  
    key = df['vian_color'][df['lch']==str(match)].iloc[0]
    matchcol.append(key)  



# convert numpy of lch colors to bgr colors
bgr_cols = []
for j in cols_blackds:
    bgr_col = []
    for i in j: 
        bgr = convert_color(lch2lab(i), cv2.COLOR_Lab2BGR) 
        bgr_col.append(bgr)
    bgr_cols.append(bgr_col)
# len(bgr_cols[0])  20 x 20 


# put bgr colors into patches
result = []
for j in bgr_cols: 
    result_arr = []
    for i in j: 
        a = np.full(PATCH, i, dtype=np.uint8)
        result_arr.append(a)
    c = np.hstack(result_arr)
    result.append(c)
    
abcd = np.vstack(result)   
print(abcd.shape) #(1100, 3700, 3)

# annotate VIAN colors using matchcolors 
matchcolors = list(zip(matches, matchcol))
print(matchcolors)

# cv2.imshow(f'LAB Matrix', abcd)
import os
os.chdir(r'D:\thesis\images')
cv2.imwrite(f'LCH_Matrix_VIAN_Colors.jpg', abcd)