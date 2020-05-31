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
# load VIAN colors
### Color-Thesaurus EPFL ###
SEARCH_COLORS_PATH = r'D:\thesis\code\pd28vianhues'
SEARCH_COLORS_FILE = 'SRGBLABhsvhslLCHHEX_Eng_VIANHuesColorThesaurus.xlsx'

# set directory 
os.chdir(SEARCH_COLORS_PATH)

# load data 
data = pd.read_excel(SEARCH_COLORS_FILE, sep=" ", index_col=0)
data.head()
data.info()

vian_hues = [
        'blue'
        , 'cyan'
        , 'green'
        , 'magenta'
        , 'orange'
        , 'pink'
        , 'red'
        , 'yellow'
        , 'beige'
        , 'black'
        , 'brown'
        , 'copper'
        , 'cream'
        , 'gold'
        , 'grey'
        , 'purple'
        , 'rust'
        , 'silver'
        , 'white'
        , 'amber'
        , 'lavender'
        , 'sepia'
        , 'apricot'
        , 'bronze'
        , 'coral'
        , 'peach'
        , 'ultramarine'
        , 'mustard'
        ]


#%%
from sklearn.neighbors import KNeighborsClassifier
# to specify
ML_MODELS_PATH = r'D:\thesis\machine_learning\models'

names = [
        "Nearest Neighbors"
         , "Linear SVM"
         ]
ML_MODELS_FILE = f'model_{names[0]}_KNN5_p2_train721_cat28.sav'

# load the model from disk
import os
import pickle
os.chdir(ML_MODELS_PATH)
clf = pickle.load(open(ML_MODELS_FILE, 'rb'))

# use machine learning classifier for color prediction
def categorize_color(color_lab, clf): 
    # lab to VIAN color 
    label = clf.predict([color_lab]) #lab: why? The CIE L*a*b* color space is used for computation, since it fits human perception
    label = label.tolist()[0]         
    #print('Label: ', label) 
    return label 


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
print(f"Searching a total of {len(palet_names)} palettes. ")
print("Searching your chosen color in palettes: \n", ', '.join(palet_names), '.')



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
    palettinis.append(palettini)
    
#%%
# for palette colors predict VIAN colors  
palette_vian_colors = []
for palid, palette in enumerate(palettinis): 
    vian_col_pred = []   
    for colid, color in enumerate(palette['lab_colors']): 
        viancolpred = categorize_color(color, clf)
        vian_col_pred.append(viancolpred)
    palette_vian_colors.append(vian_col_pred)
    palette['VIAN_color_prediction'] = vian_col_pred


#%%
    
# match VIAN colors to palette colors 

def vian_filter_palette(index, cp, palett_name, searchkey, threshold= None):
    # filter by threshold floor 
    if threshold: 
      try:
          cp = cp[cp['ratio_width'] >= threshold]
      except: 
          pass
    # filter by search key 
    if cp['VIAN_color_prediction'][cp['VIAN_color_prediction'].str.match(searchkey)].any():
#        print('Match')
        match = (index, cp, palett_name) 
        return match
    else:
        pass
#        print('No match')

#%%

# Search request - Finding result 
        
# find same color across color palettes    
print(f"Number of palettes to search: {len(palettinis)}")
print(f"VIAN color to search: {SEARCH_VIAN_COLOR}")
print(f"Threshold floor set to: {THRESHOLD_RATIO}")

# filtered color palettes 
gold_palettes = []
for i, palette in enumerate(palettinis): 
    gold = vian_filter_palette(i, palette, palet_names[i][:-12], SEARCH_VIAN_COLOR, THRESHOLD_RATIO)
    gold_palettes.append(gold)

gold_palettes = [i for i in gold_palettes if i]
print(f"Number of palettes found: {len(gold_palettes)}")

#%%
# golden waterfall 
# show all found palettes
print("-------------------------")
print(f"Number of palettes to search: {len(palettinis)}")
print(f"VIAN color to search: {SEARCH_VIAN_COLOR}")
print(f"Threshold floor set to: {THRESHOLD_RATIO}")
print(f"Number of palettes found: {len(gold_palettes)}")
print("-------------------------")
# no filtered color palettes    
if not any(gold_palettes): 
    print(f"No palettes contain searchkey VIAN color '{SEARCH_VIAN_COLOR}'.")
else: 
    print(f"Following palettes contain color '{SEARCH_VIAN_COLOR}':")
    for i, palette in enumerate(gold_palettes):
        colors_count = len(palette[1])
        # read names of gold color palettes
        print(f"{i+1}. {palet_names[i]}")
#        print(f"{i+1}. {palet_names[i]} - {COLORBAR_COUNT} out of {colors_count} colors")
        # display gold color palettes where colorbar_count=10
        display_color_grid(palette[1]['lab_colors'], 'LAB', COLORBAR_COUNT)
        # read number of gold colors
        gold_colors = palette[1][palette[1]['VIAN_color_prediction'].str.match(SEARCH_VIAN_COLOR)]
        gold_colors_count = len(gold_colors)
#        print(f"Number of gold colors for palette: {gold_colors_count} out of {len(palette[1])}")
        # display gold colors 
#        display_color_grid(gold_colors['lab_colors'], 'LAB')
    
