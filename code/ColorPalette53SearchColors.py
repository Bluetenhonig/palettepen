# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 10:40:15 2020

@author: Linda Samsinger

=====================
Color Palettes with Same Color 
=====================

For a given basic or dictionary color category, find all color palettes with the same input color. 
Machine learning is used to predict the color category of a palette color for indexing. 
A collection of hierarchical palettes can be searched using the palette's specified level in the hierarchy, 
the basic or dictionary color category and a threshold. Palette where the color category takes up an area
of at least a specified threshold size ratio in the image will be shown in the results. 

"""


# import modules
import os
import matplotlib.pyplot as plt 
import numpy as np
import cv2
import pickle
import pandas as pd
from timeit import default_timer as timer
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier

# USER SPECIFICATION 
SEARCH_COLOR_CAT = 'lavender'
PALETTE_DEPTH = 'row 20' 
THRESHOLD_RATIO = 0 
COLORBAR_COUNT = 10


# declare variables 
PALETTE_PATH = r'D:\thesis\film_colors_project\sample-dataset\screenshots\7\D0_100_bgr_palette_csv'
PALETTE_EXTENSION = 'bgr_palette.csv'
     
DICT_PATH = r'D:\thesis\input_color_name_dictionaries\thesaurus\datasets'
# basic colors 
DICT_FILE = 'eeffcnd_thesaurus_basicvian_upinterval.xlsx'
SOURCE = 'THESAURUS'
SYSTEM = 'VIAN'
# dictionary colors 
#DICT_FILE = 'eeffcnd_thesaurus_dictinterval.xlsx'
#METHOD = 'INTERVAL' 

MODEL_PATH = r'D:\thesis\machine_learning\models'
MODEL_FILE = f'model_THESAURUS_VIAN_GaussianNB_cat28_testacc0.75.sav'

assert SYSTEM.lower() in DICT_FILE and SYSTEM in MODEL_FILE or METHOD.lower() in DICT_FILE and METHOD in MODEL_FILE 

#%%

def load_all_palettes_in_folder(path, file_extension): 
    """ load all palettes contained in a folder """
    palette_files = []
    for r, d, f in os.walk(path): 
        for file in f:
            if file_extension in file:
                palette_files.append(file) 
    return palette_files

def load_palette(path, file):     
    """ load palettes"""
    os.chdir(path)
    palette = pd.read_csv(file, index_col=0)
    return palette
    
 
def show_palette(palette, depth): 
    CP_subset = palette.loc['bgr_colors'][depth] 
    CP_subset = eval(CP_subset) 
    CP_subset = np.array(CP_subset)
    display_color_grid(CP_subset, 'BGR')
    print('Number of colors in palette: ', len(CP_subset))
    return CP_subset


def get_palettes_pool(palette_path, palette_files): 
    """ loads all palettes from a path"""   
    cp_pool = []
    for file in palette_files: 
        palette = load_palette(palette_path, file) 
        cp_pool.append(palette)
    return cp_pool
  
def rename_palette_files(palette_files, input_color_category): 
    """ for pool of color palettes
    removes extension in file names """
    if palette_files: 
        palet_names = [f[:-4] for f in palette_files]  
        print(f"Searching your chosen color '{input_color_category.upper()}' in a total of {len(palet_names)} palettes. ")
        print(f"First five out of {len(palet_names)} palettes: \n", ', '.join(palet_names[:5]), '.')
        return palet_names 



def convert_color(col, conversion=cv2.COLOR_BGR2Lab): 
    """ color converter """
    color = cv2.cvtColor(np.array([[col]], dtype=np.float32), conversion)[0, 0]
    return color

def convert_array(nparray, origin, target='RGB'): 
    """helper function: convert_color
    converty numpy array of colors to other color spaces"""
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

def categorize_color(data, color_lab, clf): 
    """ helper function for get_palette_color_categories_names_pred
    - predicts color category using a machine learning classifier
    - original color needs to be in lab """ 
    try: 
        lab2pt = data['cat1'].tolist() 
    except: 
        lab2pt = data['cat'].tolist() 
    le = preprocessing.LabelEncoder()
    le.fit(lab2pt) 
    list(le.classes_) 
    label = clf.predict([color_lab])
    label = le.inverse_transform(label)
    label = label.tolist()[0]         
    return label 


def sort_color_grid(palette): 
    """ helper function for get_palette_color_categories_names_pred """
    palette = palette.sort_values(by=['ratio_width'], ascending=False)
    return palette 
  
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
            elif x == len(rgbcolors): 
                break 
            else: 
                palette = np.array(rgbcolors[x:])[np.newaxis, :, :]
                plt.figure(figsize=(colorbar_count*2,2))
                plt.imshow(palette.astype('uint8'))
                plt.axis('off')
                plt.show()
                break
    
            

def get_palettecolor_lab(palette, depth, target_cs='hsv'): 
    """ helper function for get_palette_in_lab_add_ratio
    convert palette's bgr colors into any other color space value """
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
    """ helper function for get_palette_in_lab_add_ratio
    adds other cs values to palette data """
    palettini = pd.DataFrame()
    palettini[f'{cs}_colors'] = cs_list
    palettini[f'{cs[0]}'] = [i[0] for i in cs_list]
    palettini[f'{cs[1]}'] = [i[1] for i in cs_list]
    palettini[f'{cs[2]}'] = [i[2] for i in cs_list]
    palettini[f'{cs}'] = palettini[[f'{cs[0]}', f'{cs[1]}', f'{cs[2]}']].apply(lambda row: ','.join(row.values.astype(str)), axis=1)
    return palettini

def get_paletteratiovals(palette, depth): 
    """ helper function for get_palette_in_lab_add_ratio
    convert palette's bgr colors into any color space values """
    ratio_array = np.array(eval(palette.loc['ratio_width'][depth]))
    ratio_list = ratio_array.tolist()
    return ratio_list 

def add_ratio2palett(palettini, ratio_list):
    """ helper function for get_palette_in_lab_add_ratio,
    adds ratio width to palette data """
    palettini['ratio_width'] = ratio_list
    return palettini

def get_palette_in_lab_add_ratio(cp_pool): 
    """ palette: convert bgr to lab, 
    add ratio_width for lab, 
    make new DataFrame with lab color values, 
    aggregate to list of DataFrames for all palettes
    """
    palettinis = []
    for i, palette in enumerate(cp_pool): 
        lab_list = get_palettecolor_lab(palette, PALETTE_DEPTH, 'lab')
        ratio_list = get_paletteratiovals(palette, PALETTE_DEPTH)
        palettini = add_cs2palett(lab_list, 'lab')
        try: 
            palettini = add_ratio2palett(palettini, ratio_list)
        except ValueError:
            print("Oops! Cases reported where the number of ratio_width values are unequal to number of bgr_colors: for these colors no ratio width can be analyzed.")        
        palettinis.append(palettini)
    return palettinis 
 
def get_palette_color_categories_names_pred(data, palettinis, clf): 
    """ for palette colors predict colors cats
    - sorts palette color patches
    - predicts color category for colors in palette"""
    palette_colors_cats = []
    palettes = []
    for palid, palette in enumerate(palettinis):        
        palette = sort_color_grid(palette)
        col_pred = []   
        for colid, color in enumerate(palette['lab_colors']):         
            colpred = categorize_color(data, color, clf)
            col_pred.append(colpred)
        palette_colors_cats.append(col_pred)
        palette['color_cat_prediction'] = col_pred
        palettes.append(palette)
    return palette_colors_cats, palettes

def get_all_predicted_colors(palettes): 
    colors_pred = []
    for i in range(len(palettes)):
        colors= palettes[i]['color_cat_prediction'].tolist()
        colors_pred.append(colors)
    predicted_colors = set([l for i in colors_pred for l in i])
    return predicted_colors

def filter_palette(index, cp, palett_name, searchkey, threshold= None):
    """ machtes color categories to palette colors 
    - filters by a threshold ratio
    - filters by searched color category (user specification) """
    if threshold: 
      try:
          cp = cp[cp['ratio_width'] >= threshold]
      except: 
          pass
    if cp['color_cat_prediction'][cp['color_cat_prediction'].str.match(searchkey)].any():
        match = (index, cp, palett_name) 
        return match
    else:
        pass

def show_input_color_category(data, search_color_cat, threshold = None): 
    """  shows searched input color """ 
    try:  
        search_color_cat_rgb = eval(data['srgb'][data['cat1']== search_color_cat].iloc[0])
    except: 
        search_color_cat_rgb = eval(data['srgb'][data['cat']== search_color_cat].iloc[0])
    a = np.full((5, 5, 3), search_color_cat_rgb, dtype=np.uint8)
    plt.imshow(a) #now it is in RGB 
    plt.axis('off')
    plt.show()
    print(f"Threshold floor set to: {threshold}")  
    
def get_search_results_palettes_with_color_category(palet_names, palettinis, search_color_cat, threshold_ratio): 
    """ filters color palettes based on 
    - input color category 
    - threshold ratio """
    results_palettes = []
    for i, palette in enumerate(palettinis):     
        result = filter_palette(i, palette, palet_names[i][:-12], search_color_cat, threshold_ratio)
        results_palettes.append(result)
    results_palettes = [i for i in results_palettes if i]
    return results_palettes

def show_search_results_palettes_with_color_category(palet_names, all_palettes, results_palettes, search_color_cat, colorbar_count, display_palettes=False, display_palette_colors_match=True): 
    """ shows search results: palettes containing searched color category for threshold
    - reads palette's file names
    - displays search results' palette limited to colorbar count
    - number of palette colors matching searched color category out of all palette colors in the palette
    - displays palette colors matching searched color category"""
    print("-------------------------")
    print(f"Number of palettes to search: {len(all_palettes)}")
    print(f"Color category to search: {search_color_cat}")
    print(f"Threshold floor set to: {THRESHOLD_RATIO}")
    print(f"Number of palettes found: {len(results_palettes)}")
    print("-------------------------")   
    if not any(results_palettes): 
        print(f"No palettes contain searchkey color '{search_color_cat.upper()}'.")
    else: 
        print(f"Following palettes contain color '{search_color_cat}':")
        for i, palette in enumerate(results_palettes):
            print(f"{i+1}. {palet_names[i]}")
            if display_palettes: 
                display_color_grid(palette[1]['lab_colors'], 'LAB', colorbar_count)
            searched_color_cat_as_palette_colors = palette[1][palette[1]['color_cat_prediction'].str.match(search_color_cat)]
            count_searched_color_cat_as_palette_colors = len(searched_color_cat_as_palette_colors)
            print(f"Number of resulting colors for palette: {count_searched_color_cat_as_palette_colors} out of {len(palette[1])}")
            if display_palette_colors_match: 
                display_color_grid(searched_color_cat_as_palette_colors['lab_colors'], 'LAB')
  
    
#%%    
if __name__ == '__main__':
    
    start = timer()
    
    # load palettes
    palette_files = load_all_palettes_in_folder(PALETTE_PATH, PALETTE_EXTENSION)

    # load dictionary 
    os.chdir(DICT_PATH)
    data = pd.read_excel(DICT_FILE, sep=" ", index_col=0)

    # load model  
    os.chdir(MODEL_PATH)
    clf = pickle.load(open(MODEL_FILE, 'rb'))
    
    # processing 
    print('Processing...')
    all_palettes = get_palettes_pool(PALETTE_PATH, palette_files)
    palette_names = rename_palette_files(palette_files, SEARCH_COLOR_CAT)
    all_palettes_extended = get_palette_in_lab_add_ratio(all_palettes)
    palette_colors_cats, palettes = get_palette_color_categories_names_pred(data, all_palettes_extended, clf)
    set_of_all_predicted_colors = get_all_predicted_colors(palettes)


#%%        
    # Search request    
    print('---------')
    print(f"Number of palettes to search: {len(palettes)}")
    print(f"Color category to search: '{SEARCH_COLOR_CAT.upper()}'")    
    show_input_color_category(data, SEARCH_COLOR_CAT, THRESHOLD_RATIO)    
    results_palettes = get_search_results_palettes_with_color_category(palette_names, palettes, SEARCH_COLOR_CAT, THRESHOLD_RATIO)

    print(f"Number of palettes found: {len(results_palettes)}")     
    show_search_results_palettes_with_color_category(palette_names, palettes, results_palettes, SEARCH_COLOR_CAT, COLORBAR_COUNT, display_palettes=False, display_palette_colors_match=True)
    
    end = timer()
    duration = end - start
    print('duration: ', np.round(duration, 2), 'seconds')