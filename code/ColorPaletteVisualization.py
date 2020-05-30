# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 19:42:03 2020

@author: lsamsi

=====================
Visualize Colors in a Color Palette 
=====================

!!! WARNING: Visuazlizations only happen in RGB !!!

"""

# import modules
import os
import matplotlib.pyplot as plt #2.1.2
import numpy as np
import pandas as pd
import cv2

# color picker: https://www.ginifab.com/feeds/pms/color_picker_from_image.php

# to specify
PATH = r'D:\thesis\code'

# change directory 
os.chdir(PATH)

#%%


# convert color 
def convert_color(col, conversion=cv2.COLOR_BGR2Lab): #default 
    color = cv2.cvtColor(np.array([[col]], dtype=np.float32), conversion)[0, 0]
    return color

# display color 
def display_color(color, origin=None):
    """helper function: convert_color """
    # convert color to RGB
    if origin == 'BGR': 
        rgb_color = convert_color(color, cv2.COLOR_BGR2RGB)
    elif origin == 'LAB': 
        rgb_color = convert_color(color, cv2.COLOR_LAB2RGB)*255
    else: 
        rgb_color = color
    square = np.full((10, 10, 3), rgb_color, dtype=np.uint8) / 255.0
     #display RGB colors patch 
    plt.figure(figsize = (5,2))
    plt.imshow(square) 
    plt.axis('off')
    plt.show() 

# convert numpy array of colors
def convert_array(nparray, origin, target='RGB'): 
    """helper function: convert_color """
    # convert to RGB
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

# display color palette as bar 
def display_color_grid(palette, origin='RGB', colorbar_count=10):
    """helper function: convert_array, convert_color """
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
            #plt.imshow(palette)
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


#%%

########### ColorPalette Extractor ###########

# load modules 
import os
import pandas as pd

# to specify
PATH = r'D:\thesis\videos\frames'
FILE = r'frame125_bgr_palette_colors.csv'

# set directory 
os.chdir(PATH)
# load data
palette = pd.read_csv(FILE, index_col=0)
# define palette
lowest_CP = palette.loc['bgr_colors'][-1]
# convert type 
lowest_CP = eval(lowest_CP) # str2list
lowest_CP = np.array(lowest_CP) # list2numpy array
# show palette
display_color_grid(lowest_CP)
# analyze palette
print('Number of colors in palette: ', len(lowest_CP))

#%%

########### VIAN Colors ###########
PATH = r'D:\thesis\code\pd28vianhues'
FILE = 'labbgr_vian_colors_avg.csv'
PATCH = (300, 300, 3)

# set directory 
os.chdir(PATH)

# load data 
data = pd.read_csv(FILE)

# sort data (by luminance)
data = data.sort_values(by='lab')    

# convert numpy of srgb colors to bgr colors
lst = data['vian_color'].tolist()


bgr_cols = [eval(l) for l in data['bgr']]
lab_cols = [eval(l) for l in data['lab']]


col2name = {}
for i, bgr in enumerate(bgr_cols): 
    col2name[lst[i]] = bgr
    
# len(bgr_cols[0])  20 x 20 

#bgr_cols = []
#for lab in lab_cols: 
#    bgr = convert_color(np.array(lab), cv2.COLOR_Lab2BGR)
#    bgr_cols.append(bgr*255)

lst = [None] * len(bgr_cols)
for (key, value) in col2name.items(): 
    for i in range(len(bgr_cols)): 
        if bgr_cols[i] == value:
            lst[i] = key

# put bgr colors into patches
result = []
for j in bgr_cols: 
    a = np.full(PATCH, j, dtype=np.uint8)
    result.append(a)

 
ab = np.hstack((result[0], result[1], result[2], result[3], result[4], result[5], result[6]))  
cd = np.hstack((result[7], result[8], result[9], result[10], result[11], result[12], result[13])) 
ef = np.hstack((result[14], result[15], result[16], result[17], result[18], result[19], result[20])) 
gh = np.hstack((result[21], result[22], result[23], result[24], result[25], result[26], result[27])) 
abcd = np.vstack((ab, cd, ef, gh))   
print(abcd.shape) #(2000, 2000, 3)


for i, im in enumerate(lst[:7]):
        abcd = cv2.putText(abcd, f'{im}', ((10*(30*(i+1))-280), 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
for i, im in enumerate(lst[7:14]):
        print(im)
        abcd = cv2.putText(abcd, f'{im}', ((10*(30*(i+1))-280), 450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
for i, im in enumerate(lst[14:21]):
        abcd = cv2.putText(abcd, f'{im}', ((10*(30*(i+1))-280), 750), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
for i, im in enumerate(lst[21:27]):
        abcd = cv2.putText(abcd, f'{im}', ((10*(30*(i+1))-280), 1050), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
abcd = cv2.putText(abcd, f'{lst[27]}', ((10*(30*(6+1))-280), 1050), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)   

#%%    
#cv2.imshow(f'Matrix', abcd)
# save to file
import os
os.chdir(r'D:\thesis\images')
cv2.imwrite(f'VIAN_COLOR_AVGS_CATEGORIES_sorted.jpg', abcd)

# save to file
#os.chdir(r'D:\thesis\code\pd28vianhues')
#avg = pd.DataFrame({'vian_color': lst
#                    , 'lab': lab_cols
#                    , 'bgr': bgr_cols
#                    })
#    
#avg.to_csv("labbgr_vian_colors_avg.csv", index=0) 


#%%
   
########### ColorPalette KNN ###########
# color range detection for 8 color categories

# manually-adjusted color classification chart 
hue_range = {'red': {'pure': 355, 'range': [341,10]}
            , 'orange': {'pure': 25, 'range': [11,40]}
            , 'yellow': {'pure': 50, 'range': [41,60]}
            , 'green': {'pure': 115, 'range': [61,170]}
            , 'cyan': {'pure': 185, 'range': [171,200]}
            , 'blue': {'pure': 230, 'range': [201,250]}
            , 'violet': {'pure': 275, 'range': [251,290]}
            , 'magenta': {'pure': 315, 'range': [291,340]}
            }

saturation_range = {'gray': {'pure': 0, 'range': [0,5]} }
value_range = {'black': {'pure': 0, 'range': [0,10]}}

# to specify
PATH = r'D:\thesis\code\pd12seasons'
FILE = r'lab_hsvsorted_cp1_knn5_steps51.csv'
FILTER_SEASON = 'Light Spring'  # 'True Winter', None
SAMPLE = 5 # -1 total, 5 subsample
ORIGINAL_COLORSPACE = 'hsv'
FILTER_HUE = 'yellow'  # [0-360°], hue_range['red']['pure'],
FILTER_SAT = 0.4 # [0,1]  0 - gray, 100 - color (saturation)
FILTER_VAL = 1 # [0,1]  0 - black, 100 - color (value/brightness), 0 - black, 50 - color, 100 - white (lumination)

# set directory 
os.chdir(PATH)
# load data
palette = pd.read_csv(FILE, index_col=0)
# eda data
palette.index.value_counts()

# filter colors 
if FILTER_SEASON: 
    paletti = palette.loc[FILTER_SEASON] 
else: 
    paletti = palette
    
#palett = paletti[paletti['hue'] >= hue_range[FILTER_HUE]['pure']]
if FILTER_HUE == 'red': 
    palett1 = paletti[paletti['hue'] >= hue_range[FILTER_HUE]['range'][0]]
    palett2 = paletti[paletti['hue'] <= hue_range[FILTER_HUE]['range'][1]]
    palett = pd.concat([palett1, palett2])
else: 
    palett = paletti[paletti['hue'].between(hue_range[FILTER_HUE]['range'][0],hue_range[FILTER_HUE]['range'][1])]
#palett = palett[palette['sat'] <= FILTER_SAT]
#palett = palett[palette['val'] == FILTER_VAL]
    
# sample few from total colors
if SAMPLE >=0:
    dtpts = palett[ORIGINAL_COLORSPACE].sample(n=SAMPLE) # always same sample with random_state=1
    palett = [eval(l) for l in dtpts] # pd2list
else: 
    palett = [eval(l) for l in palett[ORIGINAL_COLORSPACE]] # pd2list

if palett == []: 
    print("There are no such colors in the palette.") 
# convert type    
palet = np.array(palett) # list2numpy array

# sort colors 
try: 
    #palet = palet[palet[:,0].argsort()] # sort numpy array by first element
    palet = palet[palet[:,2].argsort()]
    palet = palet[palet] # pd2list  
except: 
    pass
    

#%%
# waterfall 
# show palette
display_color_grid(palet, 'HSV')
# analyze palette
print(f'Number of colors in this {FILTER_HUE}-filtered palette: ', len(palett))    
print(f'Number of colors in "{FILTER_SEASON}"-palette: ', len(paletti))    


#%%
# color info 
# calculate number of colors of a category rel and abs for season 
def colorinfo(season): 
    color_abs = []
    color_ratio = {}
    for key, val in hue_range.items(): 
        FILTER_HUE = (hue_range[key]['range'][0], hue_range[key]['range'][1]) # [0-360°], hue_range['red']['pure'],
        paletti = palette.loc[season] 
        if key == 'red': 
            palett1 = paletti[paletti['hue'] >= FILTER_HUE[0]]
            palett2 = paletti[paletti['hue'] <= FILTER_HUE[1]]
            palett = pd.concat([palett1, palett2])
        else: 
            palett = paletti[paletti['hue'].between(FILTER_HUE[0],FILTER_HUE[1])]
        color_ratio[key] = round(len(palett)/len(paletti)*100, 2)
        color_abs.append(len(palett))
    
    colorinfo = pd.DataFrame(color_ratio.items(), columns=['color', 'percent'])
    colorinfo['absolute'] = color_abs
    colorinfo = colorinfo.sort_values(by=['percent'], ascending=False).reset_index(drop=True)
    
    return colorinfo 

info = colorinfo('Dark Winter')
print(f'"{FILTER_SEASON}"-color palette info: \n', info)

#%%

# top-3 most frequent colors per season 
seasons = palette.index.unique().tolist()
seasons.sort()

top3 = {}
for season in seasons: 
    top3[season] = colorinfo(season)['color'][:3].tolist()

for key, value in top3.items(): 
    print(key, value)

top3s  = pd.DataFrame(top3).T

top3s.columns=['top1', 'top2', 'top3']

# most frequent colors in top1 for all seasons
top3s['top1'].value_counts()
# interesting insight: across all seasons, red (+magenta) can be worn by
# all seasons the most frequently, then blue+cyan, then green 

# analysis per season 
# spring season: defined by 3x green, 2x blue, magenta, orange, red, yellow
top3s.iloc[[0,4,-3],:]
# summer season: defined by 3x green, 2x blue, 2x cyan, magenta, red
top3s.iloc[[5,7,-2],:]
# autumn season:  defined by 3x orange, 2xred, 2x yellow, 2x green
top3s.iloc[[2,-6,-4],:]
# winter season: defined by 3xblue, 2xcyan, 2xmagenta, 2xred
top3s.iloc[[1,3,-1],:]