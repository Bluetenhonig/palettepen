# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 16:54:26 2020

=====================
Color Palettes with Same Color Contrast  
=====================

For a given color contrast, find all color palettes with the same color color contrast in them. 
Filter color palettes which contain the same color contrast . 
"""



########### ColorPalette Search ###########

# import modules
import os
import matplotlib.pyplot as plt 
import numpy as np
import cv2
import pandas as pd

# to specify: USER SPECIFICATION 
# filters
# you want all palettes with a certain contrast 
SEARCH_COLOR_CONTRAST = 'cs' 
PALETTE_DEPTH = 'row 20' # ['row 1','row 20'] (top-to-bottom hierarchy)
COLORBAR_COUNT = 10

COLOR_CONTRASTS = ['coh', 'ldc', 'cwc', 'cc', 'cc: g-r', 'cc: b-o', 'cc: v-y', 'cs']

#%%
# load color palettes dataframe 
# palette/s
PALETTE_PATH = r'D:\thesis\film_colors_project\sample-dataset\screenshots\7'
EXTENSION = 'bgr_palette.csv'
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
# load color name dictionary 
from sklearn import preprocessing
            
### Color-Thesaurus EPFL ###
SEARCH_COLORS_PATH = r'D:\thesis\input_color_name_dictionaries\thesaurus\datasets'
SEARCH_COLORS_FILE = 'effcnd_thesaurus_itten.xlsx'

# set directory 
os.chdir(SEARCH_COLORS_PATH)

# load data 
data = pd.read_excel(SEARCH_COLORS_FILE, sep=" ", index_col=0)
data.head()
data.info()

lab2pt = data['cat'].tolist() #list(df.index)
le = preprocessing.LabelEncoder()
le.fit(lab2pt) # fit all cat1 colors  



#%%
from sklearn.neighbors import KNeighborsClassifier
# to specify
ML_MODELS_PATH = r'D:\thesis\machine_learning\models'

names = [
        "Nearest Neighbors"
         , "Linear SVM"
         ]
ML_MODELS_FILE = f'model_THESAURUS_ITTEN_K-Nearest Neighbors_KNN21_p2_train721_cat6_testacc0.849.sav'

# load the model from disk
import os
import pickle
os.chdir(ML_MODELS_PATH)
clf = pickle.load(open(ML_MODELS_FILE, 'rb'))

# use machine learning classifier for color prediction
def categorize_color(color_lab, clf): 
    # lab to color category
    label = clf.predict([color_lab]) #lab: why? The CIE L*a*b* color space is used for computation, since it fits human perception
    label = le.inverse_transform(label)
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

#display_color_grid(palette['lab_colors'], 'LAB', COLORBAR_COUNT)

def sort_color_grid(palette): 
    palette = palette.sort_values(by=['ratio_width'], ascending=False)
    return palette 
        # sort by hue, sat, value 
#        hsvcolors = convert_array(rgbcolors, 'RGB', 'HSV')
#        hsvs = [list(l) for l in hsvcolors]
#        # post hsv
#        palette['hsv'] = hsvs
#        # extract hue
#        palette['hue'] = palette.hsv.map(lambda x: int(round(x[0])))
#        # extract saturation
#        palette['sat'] = palette.hsv.map(lambda x: int(round(x[1])))
#        # extract value
#        palette['val'] = palette.hsv.map(lambda x: int(round(x[2])))
#        #sort by one column only: hue
#        #palette = palette.sort_values(by=['hue'])
#        # sort by multiple columns
#        palette = palette.sort_values(by=['hue', 'sat', 'val'])  
#        rgbcolors = convert_array(palette, 'HSV', 'RGB')
     
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
#Searching a total of 569 palettes.
print("Searching your chosen color in first five examples of palettes: \n", ', '.join(palet_names[:5]), '.')



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
    hsv_list = get_palettecolvals(cp_pool[0], PALETTE_DEPTH, 'hsv')
    palettini2 = add_cs2palett(hsv_list, 'hsv')
    len(palettini)
    palettini = pd.concat([palettini, palettini2], axis=1)
    try: 
        palettini = add_ratio2palett(palettini, ratio_list)
        palettini = sort_color_grid(palettini)
    except ValueError:
        print("Oops! Cases reported where the number of ratio_width values are unequal to number of bgr_colors: for these colors no ratio width can be analyzed.")        
    palettinis.append(palettini)
    
#%%
# for palette colors predict colors cats 

palette_colors_cats = []
for palid, palette in enumerate(palettinis): 
    col_pred = []   
    for colid, color in enumerate(palette['lab_colors']):         
        colpred = categorize_color(color, clf)
        col_pred.append(colpred)
    palette_colors_cats.append(col_pred)
    palette['color_cat_prediction'] = col_pred

#%%
    
# match color categories to palette colors 

def make_contrast_palette(index, cp, palett_name):
    contrasts = [] 
    # make lumens
    palette['lumens'] = None
    palette['lumens'][palette['l'] > 75] = 'light'
    palette['lumens'][palette['l'] < 25] = 'dark'
    lumens = palette[['lumens', 'ratio_width']].groupby('lumens').agg('sum').sort_values(by='ratio_width', ascending=False)
    # make saturation
    palette['tone'] = None
    palette['tone'][palette['s'] > .75] = 'saturated'
    palette['tone'][palette['s'] < .25] = 'desaturated'
    tone = palette[['tone', 'ratio_width']].groupby('tone').agg('sum').sort_values(by='ratio_width', ascending=False)
  # make hue 
    tafel = palette[['ratio_width', 'color_cat_prediction']].groupby('color_cat_prediction').agg('sum').sort_values(by='ratio_width', ascending=False)
    try: 
        tafel = tafel.drop(['nan'])
    except: 
        pass
    # contrast of hue 
    if len(tafel.index) > 2: 
#        print('Match')
        contrasts.append('coh')
    # light-dark contrast
    if 'dark' and 'light' in lumens.index: 
#        print('Match')
        contrasts.append('ldc')      
    # cold-warm contrast
    if ('green' or 'blue' or 'violet') and ('red' or 'orange' or 'yellow') in tafel.index: 
#        print('Match')
        contrasts.append('cwc')
    # complementary contrast
    if ('green' and 'red') or ('blue' and 'orange') or ('violet' and 'yellow') in tafel.index:
#        print('Match')
        contrasts.append('cc')
        if ('green' and 'red') in tafel.index:
            contrasts.append('cc: g-r')
        if ('blue' and 'orange') in tafel.index:
            contrasts.append('cc: b-o')
        if ('violet' and 'yellow') in tafel.index:
            contrasts.append('cc: v-y')        
    # contrast of saturation 
    if 'saturated' and 'desaturated' in tone.index: 
#        print('Match')
        contrasts.append('cs') 
    match = (index, cp, palett_name, contrasts, tafel, lumens, tone) 
    return match


#%%

# Classify color palette into contrast categories (Task 3)
        
# find same color across color palettes    
print(f"Number of palettes to search: {len(palettinis)}")

# filtered color palettes 
ctrstd_palettes = []
for i, palette in enumerate(palettinis):    
    ctrstd = make_contrast_palette(i, palette, palet_names[i][:-12])
    ctrstd_palettes.append(ctrstd)

# sample palettes classified into color contrast categories 
#for i in range(20):
#    print(ctrstd_palettes[i][-2])
#    print(ctrstd_palettes[i][-1])


#%%
# Find a color palette's color contrasts

PALETTE_NUMBER = '45487'

for pltt in ctrstd_palettes: 
    if PALETTE_NUMBER == pltt[2]:
        print(f"Color contrasts of palette number '{PALETTE_NUMBER}': \n +++", ', '.join(pltt[3]), '+++')
        print(pltt[-3])
        print(pltt[-2])
        print(pltt[-1])

        
#%%

# match color contrasts to palettes 
   
def filter_palette(index, cp, palett_name, searchkey): 
    if searchkey in cp[-4]:
#        print('Match')
        match = (index, cp, palett_name) 
        return match
    else:
        pass
#        print('No match')

#%%
    
# Search request - Finding the result (not part of Task 3, but still done)

# find same color across color palettes    
print(f"Number of palettes to search: {len(palettinis)}")
print(f"Color category to search: {SEARCH_COLOR_CONTRAST}")  

# filtered color palettes 
gold_palettes = []
for i, palette in enumerate(ctrstd_palettes): 
    gold = filter_palette(i, palette, palet_names[i][:-12], SEARCH_COLOR_CONTRAST)
    gold_palettes.append(gold)
    
gold_palettes = [i for i in gold_palettes if i]
print(f"Number of palettes found: {len(gold_palettes)}")

    
#%%
# golden waterfall 
# show all found palettes
print("-------------------------")
print(f"Number of palettes to search: {len(palettinis)}")
print(f"Color category to search: {SEARCH_COLOR_CONTRAST}")
print(f"Number of palettes found: {len(gold_palettes)}")
print("-------------------------")
# no filtered color palettes    
if not any(gold_palettes): 
    print(f"No palettes contain color contrast '{SEARCH_COLOR_CONTRAST}'.")
else: 
    print(f"Following palettes contain color contrast '{SEARCH_COLOR_CONTRAST}':")
    for i, palette in enumerate(gold_palettes[:5]):
        colors_count = len(palette[1])
        # read names of gold color palettes
        print(f"{i+1}. {palet_names[i]}")
#        print(f"{i+1}. {palet_names[i]} - {COLORBAR_COUNT} out of {colors_count} colors")
        # display gold color palettes where colorbar_count=10
#        display_color_grid(palette[1][1]['lab_colors'], 'LAB', COLORBAR_COUNT)
        
        # read gold statistics 
        if SEARCH_COLOR_CONTRAST in ['coh', 'cwc', 'cc', 'cc: g-r', 'cc: b-o', 'cc: v-y']: 
            gold_tafel = palette[1][-3]
            print(gold_tafel)
        if SEARCH_COLOR_CONTRAST == 'ldc':  
            gold_lumens = palette[1][-2]
            print(gold_lumens)
        if SEARCH_COLOR_CONTRAST == 'cs': 
            gold_tone = palette[1][-1]
            print(gold_tone)


