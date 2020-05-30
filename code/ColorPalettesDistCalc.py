# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 10:40:15 2020

@author: Linda Samsinger

=====================
Color Palettes with Same Color 
=====================

For a given color, find all color palettes with the same color in them. 
Filter color palettes which contain the same color. 
"""


########### ColorPalette Search ###########

# import modules
import os
import matplotlib.pyplot as plt 
import numpy as np
import cv2
import pandas as pd

# to specify: USER SPECIFICATION (VIAN)
# filters
SEARCH_VIAN_COLOR = 'lavender' # basic color (desired format: lab)
PALETTE_DEPTH = 'row 20' # ['row 1','row 20'] (top-to-bottom hierarchy)
THRESHOLD_RATIO = 0 # [0,100], %color pixel, a threshold of 5 means that lavender must take up at least 5% of the image for a given depth
COLORBAR_COUNT = 10

#%%
# load color palettes
# palette/s
PALETTE_PATH = r'D:\thesis\videos\frames'
EXTENSION = '.csv'
PALETTE_FILE = 'frame125_bgr_palette.csv'
#FILES = ['frame250.jpg', 'frame375.jpg']     # for a list of images to process 
# load files from directory 
#FILES = ['frame12625_bgr_palette.csv', 'frame125_bgr_palette.csv']
FILES = []
for r, d, f in os.walk(PALETTE_PATH): # r=root, d=directories, f = files
    for file in f:
        if EXTENSION in file:
            FILES.append(file) 
            


#%%

### Processing ###

# convert color 
def convert_color(col, conversion=cv2.COLOR_BGR2Lab): #default 
    color = cv2.cvtColor(np.array([[col]], dtype=np.float32), conversion)[0, 0]
    return color

# convert numpy array of colors
def convert_array(nparray, origin, target='RGB'): 
    """helper function: convert_color """
    # convert to RGB
    converted_colors = []
    for col in nparray: 
        if origin == 'BGR' and target == 'RGB':        
            converted_color = convert_color(col, cv2.COLOR_BGR2RGB)
        if origin == 'LAB' and target == 'RGB':     
            converted_color = convert_color(col, cv2.COLOR_LAB2RGB)*255
        if origin == 'RGB' and target == 'LAB':     
            converted_color = convert_color(col, cv2.COLOR_RGB2LAB)
        if origin == 'HSV' and target == 'RGB':     
            converted_color = convert_color(col, cv2.COLOR_HSV2RGB)*255
        if origin == 'RGB' and target == 'HSV':     
            converted_color = convert_color(col, cv2.COLOR_RGB2HSV)
        converted_colors.append(converted_color)
    return converted_colors

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
        return rgbcolors
            


#%%
# load function
                
def load_palette(path, file):     
    # set directory 
    os.chdir(path)
    # load data
    palette = pd.read_csv(file, index_col=0)
    return palette
    
 
def show_palette(palette, depth): 
    # define palette
    CP_subset = palette.loc['bgr_colors'][depth]
    # convert type 
    CP_subset = eval(CP_subset) # str2list
    CP_subset = np.array(CP_subset) # list2numpy array
    # show palette
    rbgcolors = display_color_grid(CP_subset, 'BGR')
    # analyze palette
    print('Number of colors in palette: ', len(CP_subset))
    return CP_subset, rgbcolors

#%%
# all palettes    
cp_pool = []

# load palette
for FILE in FILES: 
    palette = load_palette(PALETTE_PATH, FILE) 
    cp_pool.append(palette)
    
# show palette
#cp_row, rgb = show_palette(cp_pool[0], PALETTE_DEPTH)

# pool of color palettes 
# remove extension in file names 
palet_names = [f[:-4] for f in FILES]  
print(f"Number of palettes: {len(palet_names)}")
print("Names of palettes: \n", ', '.join(palet_names), '.')



#%%
# add other cs values to palette data

def get_palettecolvals(palette, depth, target_cs='hsv'): 
    """ convert palette's bgr colors into any color space values """
    bgr_array = np.array(eval(palette.loc['bgr_colors'][depth]))
    rgb_array = convert_array(bgr_array, origin='BGR', target='RGB')
    if target_cs == 'rgb':
        rgb_list = [list(i) for i in rgb_array]
        return rgb_list
    elif target_cs == 'hsv':
        hsv_array =[]
        for i in rgb_array:
            rgb = np.array(i)
            rgbi = np.array([[rgb/ 255]], dtype=np.float32)
            hsv = cv2.cvtColor(rgbi, cv2.COLOR_RGB2HSV)
            hsv = hsv[0, 0]
            hsv_array.append(hsv)
        hsv_list = [list(i) for i in hsv_array]
        return hsv_list 
    elif target_cs == 'lab':
        lab_array =[]
        for idn, rgb in enumerate(rgb_array):
            rgb = np.array(rgb)
            rgbi = np.array([[rgb/ 255]], dtype=np.float32)
            lab = cv2.cvtColor(rgbi, cv2.COLOR_RGB2LAB)
            lab = lab[0, 0]
            lab_array.append(lab)
        lab_list = [i.tolist() for i in lab_array]
        return lab_list 

def add_cs2palett(cs_list, cs='hsv'):
    # add color space values to palette
    palettini = pd.DataFrame()
    palettini[f'{cs}_colors'] = cs_list
    palettini[f'{cs[0]}'] = [i[0] for i in cs_list]
    palettini[f'{cs[1]}'] = [i[1] for i in cs_list]
    palettini[f'{cs[2]}'] = [i[2] for i in cs_list]
    palettini[f'{cs}'] = palettini[[f'{cs[0]}', f'{cs[1]}', f'{cs[2]}']].apply(lambda row: ','.join(row.values.astype(str)), axis=1)
    return palettini


#%%
# add ratio width info to palette data

def get_paletteratiovals(palette, depth): 
    """ convert palette's bgr colors into any color space values """
    ratio_array = np.array(eval(palette.loc['ratio_width'][depth]))
    ratio_list = ratio_array.tolist()
    return ratio_list 

def add_ratio2palett(df, ratio_list):
    # add ratio width to palette
    palettini['ratio_width'] = ratio_list
    return palettini

#%%

# palette: convert bgr to lab, add ratio_wdith for lab, make new DataFrame with lab color values, aggregate to list of DataFrames for all palettes
palettinis = []
for i, palette in enumerate(cp_pool): 
    lab_list = get_palettecolvals(palette, PALETTE_DEPTH, 'lab')
    ratio_list = get_paletteratiovals(palette, PALETTE_DEPTH)
    palettini = add_cs2palett(lab_list, 'lab')
    try: 
        palettini = add_ratio2palett(palettini, ratio_list)
        
    except ValueError:
        print("Oops! Cases reported where the number of ratio_width values are unequal to number of bgr_colors: for these colors no ratio width can be analyzed.")        
    # sort values  by ratio width 
    try: 
        palettini = palettini.sort_values(by='ratio_width', ascending=False)
    except:
        pass
    palettinis.append(palettini)
    

#%%
from scipy.spatial import distance
from statistics import mean

labcols = []
for i in range(len(palettinis)): 
    labs = palettinis[i]['lab_colors'].tolist()
    labcols.append(labs)

cps = pd.DataFrame({'palette_name': palet_names
                        , 'lab': labcols})
x = []
for i in range(len(cps)): 
    el = cps.iloc[i][0], cps.iloc[i][1]
    x.append(el)

# possible long exec time     
import itertools
paircombi = list(itertools.combinations(x, 2))
print(f"Number of pairwise combinations: {len(paircombi)}.")

# calculate minimum euclidean of pair 
def mindist_pairpoint(A,B,dist = 'euclidean'):
    min_dist = []
    min_pair =  []
    for a in A:
        dist = []
        pair = []
        for b in B:
            dist.append(distance.euclidean(a,b))
            pair.append((a,b))
        min_dist.append(min(dist))
        min_id = dist.index(min(dist)) 
        min_pair.append(pair[min_id])
    return min_pair, min_dist

def get_pbond(min_dist):
    avg = round(mean(min_dist),4)
    return avg 


def pairwise_dist(pair, i): 
    pair_name = pair[0][0], pair[1][0]
    min_pair, min_dist = mindist_pairpoint(paircombi[i][0][1], paircombi[i][1][1])
    pbond = get_pbond(min_dist)
    print(f"Number of matched pair minimums:{len(min_pair)}")
    return pair_name, pbond



pair_names = []
pbonds = []
for i, pair in enumerate(paircombi):
    print(i)
    pair_name, pbond = pairwise_dist(paircombi[i],i)
    pair_names.append(pair_name)
    pbonds.append(pbond)
    
 
cp_pairs = pd.DataFrame({ 'pair1': [i[0] for i in pair_names],
                         'pair2': [i[1] for i in pair_names],
                        'pair': pair_names, 
                         'pbond': pbonds})
 
# save pair bonds dataframe
os.chdir(r'D:\thesis\code\pd4cpbonds')
cp_pairs.to_csv("palette_pair_pbonds", index=False)    

#%%
# get n-closest possible palette-pair from a pool of palettes  

# load pair bonds dataframe
os.chdir(r'D:\thesis\code\pd4cpbonds')
cp_pairs = pd.read_csv("palette_pair_pbonds")    

# find minimum pbond 
def min_pbond(pbonds, number = 0):
    # set base 
    minimum = min(pbonds)
    while number > 0: 
        new_pbonds = list(filter(lambda a: a != minimum, pbonds))
        # use recursion  
        minimum = min_pbond(new_pbonds, number-1)
        return minimum
    return minimum 

pbonds = sorted(cp_pairs['pbond'].tolist())
pbonds[:3] #[0.4764, 0.7084, 0.7225]
cp_min_pbonds = min_pbond(pbonds)
gold_pair = list(eval(cp_pairs['pair'][cp_pairs['pbond'] == cp_min_pbonds].iloc[0]))
golden_pair = ', '.join(list(eval(cp_pairs['pair'][cp_pairs['pbond'] == cp_min_pbonds].iloc[0])))
print(f"Palettes {golden_pair} are the closest to each other.")
NUMBER = 1 
cp_2min_pbonds = min_pbond(pbonds, number = NUMBER)
silver_pair = ', '.join(list(eval(cp_pairs['pair'][cp_pairs['pbond'] == cp_2min_pbonds].iloc[0])))
print(f"Palettes {silver_pair} are {NUMBER + 1}. closest to each other.")
NUMBER = 2
cp_3min_pbonds = min_pbond(pbonds, number = NUMBER)
bronze_pair = ', '.join(list(eval(cp_pairs['pair'][cp_pairs['pbond'] == cp_3min_pbonds].iloc[0])))
print(f"Palettes {bronze_pair} are {NUMBER + 1}. closest to each other.")

# display palette 
rgbs = display_color_grid(cps['lab'][cps['palette_name']==gold_pair[0]].iloc[0], 'LAB')
rgbs = display_color_grid(cps['lab'][cps['palette_name']==gold_pair[1]].iloc[0], 'LAB')


#%%
# get top-n closest palettes for given palette

# Search request - Finding result 
SEARCHKEY_PALETTE = "frame3500_bgr_palette"
TOPN = 10 
# get pair-partner for given palette and sort pbonds 
cp_pairs['pair1'] = [eval(i)[0] for i in cp_pairs['pair']]
cp_pairs['pair2'] = [eval(i)[1] for i in cp_pairs['pair']]
# get symmetrical values too 
gold_pbonds1 = cp_pairs[cp_pairs['pair1']== SEARCHKEY_PALETTE]
gold_pbonds2 = cp_pairs[cp_pairs['pair2']== SEARCHKEY_PALETTE]
gold_pbonds = gold_pbonds1.append(gold_pbonds2)
gold_pbonds = gold_pbonds.sort_values(by='pbond')
gold_pbonds = gold_pbonds.reset_index(drop=True)

def get_sym_goldpal(df, SEARCHKEY_PALETTE): 
    pair1 = gold_pbonds[['pbond','pair2']][gold_pbonds['pair1']==SEARCHKEY_PALETTE]
    pair1['alt_pair'] = pair1['pair2']
    pair2 = df[['pbond','pair1']][df['pair2']==SEARCHKEY_PALETTE]
    pair2['alt_pair'] = pair2['pair1']
    pair = pair1.append(pair2)
    pair = pair.sort_values(by='pbond')
    return pair 
        
# show top-n closest pbonds for a given palette 
gold_palettes = get_sym_goldpal(gold_pbonds, SEARCHKEY_PALETTE)['alt_pair'][:TOPN].reset_index(drop=True)
print("-------------------------")
print(f"Task: Find most similar color palettes")
print(f"Searching color palette: {SEARCHKEY_PALETTE}")
print(f"Total number of gold palettes: {len(gold_pbonds)}")
print(f"Top-{TOPN} gold palettes: \n{gold_palettes}")
print("-------------------------")


# TODO: give weights to each lab color val  
#%%
# golden waterfall 
# show all found palettes
    
if not any(gold_palettes): 
    print(f"No palettes found.")
else: 
    print("-------------------------")
    print(f"Display palettes most similar to {SEARCHKEY_PALETTE}:")
    display_color_grid(cps['lab'][cps['palette_name']==SEARCHKEY_PALETTE].iloc[0], 'LAB')
    print("-------------------------")
    for gold in gold_palettes:
        print(f"Palette: {gold}")
        display_color_grid(cps['lab'][cps['palette_name']==gold].iloc[0], 'LAB')

    
 