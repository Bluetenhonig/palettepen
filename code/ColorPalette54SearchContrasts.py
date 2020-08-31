# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 16:54:26 2020

=====================
Color Palettes with Same Color Contrast  
=====================

For a given color contrast, find all color palettes with the same color color contrast in them. 
Filter color palettes which contain the same color contrast . 
"""


# import modules
import os
import matplotlib.pyplot as plt 
import numpy as np
import cv2
import pickle
import pandas as pd
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from timeit import default_timer as timer
import sys
sys.path.append(r'D:\thesis\code')
from ColorConversion00 import convert_color


# USER SPECIFICATION 
SEARCH_COLOR_CONTRAST = 'ce' 
PALETTE_DEPTH = 'row 20'
COLORBAR_COUNT = 10
COLOR_CONTRASTS = ['coh', 'ldc', 'cwc', 'cc', 'cc: g-r', 'cc: b-o', 'cc: v-y', 'cs', 'ce']
SEARCH_PALETTE = '45537'

# declare variables 
PALETTE_PATH = r'D:\thesis\film_colors_project\sample-dataset\screenshots\7\D0_100_bgr_palette_csv'
PALETTE_EXTENSION = 'bgr_palette.csv'

IMG_PATH = r'D:\thesis\film_colors_project\sample-dataset\screenshots\7\images'
IMG_EXTENSION = '.jpg'

MODEL_PATH = r'D:\thesis\machine_learning\models'
MODEL_FILE = f'model_THESAURUS_ITTEN_INTERVAL_KNeighborsClassifier_KNN23_p2_train1035_cat6_testacc0.87.sav'

DICT_PATH = r'D:\thesis\input_color_name_dictionaries\thesaurus\datasets'
DICT_FILE = 'eeffcnd_thesaurus_basicitten_upinterval.xlsx'
SOURCE = 'THESAURUS'
SYSTEM = 'ITTEN'
METHOD = 'INTERVAL'

assert SYSTEM in MODEL_FILE and SYSTEM.lower() in DICT_FILE and METHOD in MODEL_FILE and METHOD.lower() in DICT_FILE 

#%%

def load_all_palettes_in_folder(path, file_extension): 
    """ load all palettes contained in a folder """
    files = []
    for r, d, f in os.walk(path): 
        for file in f:
            if file_extension in file:
                files.append(file) 
    return files

def load_palette(path, file):     
    """ load CSV palettes"""
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


def load_dictionary(path, file):     
    """ load dictionary"""
    os.chdir(path)
    dictn = pd.read_excel(file, index_col=0)
    return dictn

def encode_labels(data): 
    lab2pt = data['cat1'].tolist() 
    le = preprocessing.LabelEncoder()
    le.fit(lab2pt) 
    set(le.transform(lab2pt)) 
    le.inverse_transform([3])
    le.classes_
    return lab2pt, le

def categorize_color(data, color_lab, clf): 
    """ machine learning classifier predicts lab color's color category"""
    label = clf.predict([color_lab]) 
    lab2pt, le = encode_labels(data)
    label = le.inverse_transform(label)
    label = label.tolist()[0]         
    return label 


def convert_color_quick(col, conversion=cv2.COLOR_BGR2Lab): 
    """ color converter """
    color = cv2.cvtColor(np.array([[col]], dtype=np.float32), conversion)[0, 0]
    return color

def convert_array(nparray, origin, target='RGB'): 
    """helper function: convert_color
    converty numpy array of colors to other color spaces"""
    converted_colors = []
    for col in nparray: 
        if origin == 'BGR' and target == 'RGB':        
            converted_color = convert_color_quick(col, cv2.COLOR_BGR2RGB)
        if origin == 'LAB' and target == 'RGB':     
            converted_color = convert_color_quick(col, cv2.COLOR_LAB2RGB)*255
        if origin == 'RGB' and target == 'LAB':     
            converted_color = convert_color_quick(col, cv2.COLOR_RGB2LAB)
        if origin == 'HSV' and target == 'RGB':     
            converted_color = convert_color_quick(col, cv2.COLOR_HSV2RGB)*255
        if origin == 'RGB' and target == 'HSV':     
            converted_color = convert_color_quick(col, cv2.COLOR_RGB2HSV)
        converted_colors.append(converted_color)
    return converted_colors

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

def rename_palette_files(palette_files): 
    """ remove extension in file names """
    file_name = [f[:-4] for f in palette_files]
    frame_name = [f[:-16] for f in palette_files]  
    print(f"Searching a total of {len(palette_files)} palettes. ")
    print("First five examples of palettes: \n", ', '.join(frame_name[:5]), '.')
    return frame_name

def get_palettecolvals(palette, depth, target_cs='hsv'): 
    """ convert palette's bgr colors into any color space values """
    bgr_array = np.array(eval(palette.loc['bgr_colors'][depth]))
    rgb_array = convert_array(bgr_array, origin='BGR', target='RGB')
    lab_array = convert_array(np.array(rgb_array)/255, origin='RGB', target='LAB')
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
    elif target_cs == 'lch':
        lch_array =[]
        for idn, lab in enumerate(lab_array):
            lch = convert_color(lab, "LAB", "LCH")
            lch = np.array(lch)
            lch_array.append(lch)
        lch_list = [i.tolist() for i in lch_array]
        return lch_list 

def add_cs2palett(cs_list, cs='hsv'):
    """ add color space values to palette """
    palettini = pd.DataFrame()
    palettini[f'{cs}_colors'] = cs_list
    palettini[f'{cs[0]}'] = [i[0] for i in cs_list]
    palettini[f'{cs[1]}'] = [i[1] for i in cs_list]
    palettini[f'{cs[2]}'] = [i[2] for i in cs_list]
    palettini[f'{cs}'] = palettini[[f'{cs[0]}', f'{cs[1]}', f'{cs[2]}']].apply(lambda row: ','.join(row.values.astype(str)), axis=1)
    return palettini

def get_paletteratiovals(palette, depth): 
    """ convert palette's bgr colors into any color space values """
    ratio_array = np.array(eval(palette.loc['ratio_width'][depth]))
    ratio_list = ratio_array.tolist()
    return ratio_list 

def add_ratio2palett(df, ratio_list):
    """ add ratio width to palette """
    df['ratio_width'] = ratio_list
    return df


def get_palette_in_lab_add_ratio(cp_pool): 
    """ palette: convert bgr to lab, 
    add ratio_width for lab, 
    make new DataFrame with lab color values, 
    aggregate to list of DataFrames for all palettes
    """
    palettinis = []
    print(len(cp_pool), 'to be processed.')
    for i, palette in enumerate(cp_pool): 
        lab_list = get_palettecolvals(palette, PALETTE_DEPTH, 'lab')
        ratio_list = get_paletteratiovals(palette, PALETTE_DEPTH)
        palettini = add_cs2palett(lab_list, 'lab')
        lch_list = get_palettecolvals(palette, PALETTE_DEPTH, 'lch')
        palettini2 = add_cs2palett(lch_list, 'lch')
        palettini2 = palettini2.drop(['l'], axis=1)
        len(palettini)
        palettini = pd.concat([palettini, palettini2], axis=1)
        try: 
            palettini = add_ratio2palett(palettini, ratio_list)
            palettini = sort_color_grid(palettini)
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


def make_contrast_palette(index, cp, palett_name):
    """ determine color contrasts from color palette's color categories of palette colors  """

    contrasts = []   
    
    cp['lumens'] = None
    cp['lumens'][cp['l'] > 55] = 'light'
    cp['lumens'][cp['l'] < 25] = 'dark'
    cp['tone'] = None
    cp['tone'][cp['c'] > 50] = 'saturated'
    cp['tone'][cp['c'] < 25] = 'desaturated'
    
    lumens = cp[['lumens', 'ratio_width']].groupby('lumens').agg('sum').sort_values(by='ratio_width', ascending=False)
    lumens_colors = list(lumens.index)
    if set(['dark','light']).issubset(lumens_colors) and lumens.loc['dark'].iloc[0]>40:
        contrasts.append('ldc') 
        
    tafel = cp[cp['lumens'] != 'dark'] 
    tafel = tafel[tafel['c'] > 30]
    tafel = tafel[['ratio_width', 'lumens', 'tone', 'color_cat_prediction']].groupby('color_cat_prediction').agg('sum').sort_values(by='ratio_width', ascending=False)    
    try: 
        tafel = tafel.drop(['nan'])
    except: 
        pass   
    if 'orange' in tafel.index.tolist(): 
        if tafel.loc['orange'].iloc[0] > 20: 
            tafel = tafel.drop(['orange'])  
#    tafel = tafel[tafel['ratio_width']>=1] 
    small_colors = tafel[tafel['ratio_width'] <= 30 ]
    small_colors = small_colors[small_colors['ratio_width'] >= 5 ].index.tolist()
    sat_colors = set(cp['color_cat_prediction'][cp['c'] >70].tolist())
    for small in small_colors: 
        if small not in sat_colors: 
            tafel.drop([small])           
    tafel_colors = list(tafel.index)
    if len(tafel_colors) > 2: 
        contrasts.append('coh')
           
    noblue = cp[cp['color_cat_prediction']!='blue']
    tafel = noblue[noblue['lumens'] != 'dark'] 
    tafel = tafel[tafel['c'] > 10]
    tafel = pd.concat([tafel, cp[cp['color_cat_prediction']=='blue']])
    noorange = tafel[tafel['color_cat_prediction'] =='orange']
    noorange = noorange[noorange['tone'] =='desaturated']  
    combined = tafel.append(noorange)
    tafel = combined[~combined.index.duplicated(keep=False)]
    tafel = tafel[['ratio_width', 'lumens', 'tone', 'color_cat_prediction']].groupby('color_cat_prediction').agg('sum').sort_values(by='ratio_width', ascending=False)    
    try: 
        tafel = tafel.drop(['nan'])
    except: 
        pass   
    orange = cp[cp['color_cat_prediction'] == 'orange']
    if orange['l'][orange['l'] >= 40].any() == False:   
        if 'orange' in tafel.index.tolist(): 
            if tafel.loc['orange'].iloc[0] > 20: 
                tafel = tafel.drop(['orange'])            
    tafel_colors = list(tafel.index)
    if any(x in ['green', 'blue', 'violet'] for x in tafel_colors) and any(x in ['red', 'orange', 'yellow'] for x in tafel_colors):
        contrasts.append('cwc')
 
    if set(['green','red']).issubset(tafel_colors) or set(['blue','orange']).issubset(tafel_colors) or set(['violet','yellow']).issubset(tafel_colors):
        contrasts.append('cc')      
        if set(['green','red']).issubset(tafel_colors):
            contrasts.append('cc: g-r')
        if set(['blue','orange']).issubset(tafel_colors):
            contrasts.append('cc: b-o')
        if set(['violet','yellow']).issubset(tafel_colors):
            contrasts.append('cc: v-y')        
    

    tone = cp[['tone', 'ratio_width']].groupby('tone').agg('sum').sort_values(by='ratio_width', ascending=False)
    tone_colors = list(tone.index)
    if set(['saturated','desaturated']).issubset(tone_colors):
        contrasts.append('cs') 
        
        total = tone['ratio_width'].iloc[0]+tone['ratio_width'].iloc[1]
        first_proprz = round(tone['ratio_width'].iloc[0]/total,2) 
        second_proprz = round(tone['ratio_width'].iloc[1]/total,2)
        if (first_proprz >= 0.98 and second_proprz <= 0.02 and second_proprz >= 0.005):
            contrasts.append('ce')
            
    match = (index, cp, palett_name, contrasts, tafel, lumens, tone) 
    return match

def categorize_palette_to_contrast(palettinis, palette_names): 
    ctrstd_palettes = []
    for i, palette in enumerate(palettinis):    
        ctrstd = make_contrast_palette(i, palette, palette_names[i])
        ctrstd_palettes.append(ctrstd)
    return ctrstd_palettes


def show_sample_contrast_classification(ctrstd_palettes):  
    """ first three palettes's contrast classification """
    for i in range(3):
        print(ctrstd_palettes[i][2])
        print(ctrstd_palettes[i][-4])
        print(ctrstd_palettes[i][-3])
        print(ctrstd_palettes[i][-2])
        print(ctrstd_palettes[i][-1])

def show_all_contrasts_palettenumber(palette_nr, ctrstd_palettes, palette_names, display_tables=True): 
    palette_names.index(palette_nr)
    for pltt in ctrstd_palettes: 
        if palette_nr == pltt[2]:
            print(pltt[-4])
            if display_tables: 
                print(pltt[-3])
                print(pltt[-2])
                print(pltt[-1])

def filter_palette(index, cp, palett_name, searchkey): 
    """ search color contrast in palettes """
    if searchkey in cp[-4]:
        match = (index, cp, palett_name) 
        return match
    else:
        pass

def get_search_results_palettes_with_color_contrast(palette_names, ctrstd_palettes, search_color_contrast): 
    """ filtered color palettes """
    results_palettes = []
    for i, palette in enumerate(ctrstd_palettes): 
        result = filter_palette(i, palette, palette_names[i][:-12], search_color_contrast)
        results_palettes.append(result)
        
    results_palettes = [i for i in results_palettes if i]
    print(f"Number of palettes found: {len(results_palettes)}")
    return results_palettes

def show_search_results_palettes_with_color_contrast(palette_names, palettes, results_palettes, search_color_contrast, colorbar_count, display_palettes=False, display_tables=True, topn=5):
    """ show all found palettes """
    
    if results_palettes == None: 
        print(f"No palettes contain color contrast '{SEARCH_COLOR_CONTRAST}'.")
        print("-------------------------")
    else: 
        print("-------------------------")
        print(f"First {topn} palettes containing color contrast '{SEARCH_COLOR_CONTRAST}':")
        for i, palette in enumerate(results_palettes[:topn]):
            print(f"{i+1}. {palette_names[i]}")
            if display_palettes:
                display_color_grid(palette[1][1]['lab_colors'], 'LAB', colorbar_count)
            if display_tables: 
                if search_color_contrast in ['coh', 'cwc', 'cc', 'cc: g-r', 'cc: b-o', 'cc: v-y', 'ce']: 
                    hue_tafel = palette[1][-3]
                    print(hue_tafel)
                if search_color_contrast == 'ldc':  
                    lumens_tafel = palette[1][-2]
                    print(lumens_tafel)
                if search_color_contrast == 'cs' or search_color_contrast == 'ce': 
                    tone_tafel = palette[1][-1]
                    print(tone_tafel)
        print("-------------------------")        
                    
#%%
if __name__ == '__main__':
      
    start = timer()
     
    # load palettes
    palette_files = load_all_palettes_in_folder(PALETTE_PATH, PALETTE_EXTENSION)
    print('Number of color palettes: ', len(sorted(palette_files)))

    # load images
    img_files = load_all_palettes_in_folder(IMG_PATH, IMG_EXTENSION)
    print('Number of images: ', len(sorted(img_files)))   
    assert len(palette_files) == len(img_files), 'Number of color palettes not equal to number of images.'
    
    # load model 
    os.chdir(MODEL_PATH)
    clf = pickle.load(open(MODEL_FILE, 'rb'))
   
    # load dictionary
    data = load_dictionary(DICT_PATH, DICT_FILE)
    data.head()
    
    # encode labels
    lab2pt, le =encode_labels(data)
    
    # processing 
    all_palettes = get_palettes_pool(PALETTE_PATH, palette_files)
    palette_names = rename_palette_files(palette_files)  
    all_palettes_extended = get_palette_in_lab_add_ratio(all_palettes)
    palette_colors_cats, palettes = get_palette_color_categories_names_pred(data, all_palettes_extended, clf)
    ctrstd_palettes = categorize_palette_to_contrast(palettes, palette_names)


#%%

    # Search request  1  
    print('---------')
    print(f"Palette ID to search: '{SEARCH_PALETTE.upper()}'")  
    show_all_contrasts_palettenumber(SEARCH_PALETTE, ctrstd_palettes, palette_names, display_tables=False)

#%%
    
    # Search request  2
    print('---------')
    print(f"Number of palettes to search: {len(all_palettes)}")
    print(f"Color category to search: {SEARCH_COLOR_CONTRAST}")  
    results_palettes = get_search_results_palettes_with_color_contrast(palette_names, ctrstd_palettes, SEARCH_COLOR_CONTRAST)        
    show_search_results_palettes_with_color_contrast(palette_names, palettes, results_palettes, SEARCH_COLOR_CONTRAST, COLORBAR_COUNT, display_palettes=False, display_tables=True, topn=10)
   

    end = timer()
    duration = end - start
    print('duration: ', np.round(duration, 2), 'seconds')