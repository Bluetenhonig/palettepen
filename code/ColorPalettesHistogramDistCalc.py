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
# load images 
# images
PALETTE_PATH = r'D:\thesis\input_videos\frames'
EXTENSION = '.jpg'
#IMAGE_FILE = 'frame125.jpg'
#FILES = ['frame250.jpg', 'frame375.jpg']     # for a list of images to process 
# load files from directory 
#FILES = ['frame12625_bgr_palette.csv', 'frame125_bgr_palette.csv']
FILES = []
for r, d, f in os.walk(PALETTE_PATH): # r=root, d=directories, f = files
    for file in f:
        if EXTENSION in file:
            FILES.append(file) 
  
# set directory 
os.chdir(PALETTE_PATH)

#%%            
import cv2
import numpy as np
from matplotlib import pyplot as plt

def show_hist(file): 
    img = cv2.imread(file)
    plt.plot(cv2.calcHist([img],[0],None,[256],[0,256]))
    color = ('b','g','r')
    for i,col in enumerate(color):
        histr = cv2.calcHist([img],[i],None,[256],[0,256])
        plt.plot(histr,color = col)
        plt.xlim([0,256])
    plt.show()

def save_hist(file): 
    img = cv2.imread(file)
    plt.plot(cv2.calcHist([img],[0],None,[256],[0,256]))
    color = ('b','g','r')
    for i,col in enumerate(color):
        histr = cv2.calcHist([img],[i],None,[256],[0,256])
        plt.plot(histr,color = col)
        plt.xlim([0,256])
    plt.savefig(f"{file}_histogram.jpg")
    plt.show()

#%%
# all images    
image_pool = []

# show histogram of images
for FILE in FILES: 
    show_hist(FILE)
    
# save histogram of images 
for FILE in FILES: 
    save_hist(FILE)


#%%
# compare histograms 
# based on : https://docs.opencv.org/2.4/doc/tutorials/imgproc/histograms/histogram_comparison/histogram_comparison.html

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

    
 