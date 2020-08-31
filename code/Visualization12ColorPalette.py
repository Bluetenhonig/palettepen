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


#%%

#%%

#%%

#%%

# color palettes using seaborn
import seaborn as sns

flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
sns.palplot(sns.color_palette(flatui))

#%%

#visualize color dictionary

import os 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

os.chdir(r'D:\thesis\input_color_name_dictionaries\thesaurus\datasets')
data = pd.read_excel(r'ffcnd_thesaurus.xlsx', col_index=0)

EFFCNDThesaurus_COLORS = {}

for ids in range(data[['name', 'hex']].shape[0]):
    EFFCNDThesaurus_COLORS[data[['name']].iloc[ids][0]] = data[['hex']].iloc[ids][0]


def plot_colortable(colors, title, sort_colors=True, emptycols=0):

    cell_width = 212
    cell_height = 22
    swatch_width = 48
    margin = 12
    topmargin = 40

    # Sort colors by hue, saturation, value and name.
    if sort_colors is True:
        by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(color))),
                         name)
                        for name, color in colors.items())
        names = [name for hsv, name in by_hsv]
    else:
        names = list(colors)

    n = len(names)
    ncols = 4 - emptycols
    nrows = n // ncols + int(n % ncols > 0)

    width = cell_width * 4 + 2 * margin
    height = cell_height * nrows + margin + topmargin
    dpi = 72

    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.subplots_adjust(margin/width, margin/height,
                        (width-margin)/width, (height-topmargin)/height)
    ax.set_xlim(0, cell_width * 4)
    ax.set_ylim(cell_height * (nrows-0.5), -cell_height/2.)
    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)
    ax.set_axis_off()
    ax.set_title(title, fontsize=24, loc="left", pad=10)

    for i, name in enumerate(names):
        row = i % nrows
        col = i // nrows
        y = row * cell_height

        swatch_start_x = cell_width * col
        swatch_end_x = cell_width * col + swatch_width
        text_pos_x = cell_width * col + swatch_width + 7

        ax.text(text_pos_x, y, name, fontsize=14,
                horizontalalignment='left',
                verticalalignment='center')

        ax.hlines(y, swatch_start_x, swatch_end_x,
                  color=colors[name], linewidth=18)

    return fig

#plot_colortable(mcolors.BASE_COLORS, "Base Colors",
#                sort_colors=False, emptycols=1)
#plot_colortable(mcolors.TABLEAU_COLORS, "Tableau Palette",
#                sort_colors=False, emptycols=2)

#sphinx_gallery_thumbnail_number = 3
#plot_colortable(mcolors.CSS4_COLORS, "CSS Colors")

# Optionally plot the XKCD colors (Caution: will produce large figure)
#xkcd_fig = plot_colortable(mcolors.XKCD_COLORS, "XKCD Colors")
#xkcd_fig.savefig("XKCD_Colors.png")

plot_colortable(EFFCNDThesaurus_COLORS, "EFFCND Thesaurus-VIAN")


plt.show()