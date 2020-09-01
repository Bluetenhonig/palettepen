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
import pickle
import matplotlib.pyplot as plt 
import numpy as np
import cv2
import pandas as pd
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
import sys
sys.path.append(r'D:\thesis\code\04_Final')
from ColorConversion00 import convert_color
from ColorPalette54SearchContrasts import *
from ColorPalette54SearchContrasts import get_palettes_pool


# USER SPECIFICATION 
SEARCH_COLOR_CONTRAST = 'coh' 
PALETTE_DEPTH = 'row 20' 
COLORBAR_COUNT = 10
COLOR_CONTRASTS = ['coh', 'ldc', 'cwc', 'cc', 'cc: g-r', 'cc: b-o', 'cc: v-y', 'cs', 'ce']
print('Number of color contrasts: ', len(COLOR_CONTRASTS))

# declare variables
CSV_PALETTE_PATH = r'D:\thesis\film_colors_project\sample-dataset\screenshots\7\D0_100_bgr_palette_csv'
CSV_PALETTE_EXTENSION = 'bgr_palette.csv'
IMG_PALETTE_PATH = r'D:\thesis\film_colors_project\sample-dataset\screenshots\7\D100_lab_palette_img'
IMG_PALETTE_EXTENSION = 'lab_palette.jpg'
IMG_PATH = r'D:\thesis\film_colors_project\sample-dataset\screenshots\7\images'
IMG_EXTENSION = '.jpg'
DICT_PATH = r'D:\thesis\input_color_name_dictionaries\thesaurus\datasets'
DICT_FILE = 'eeffcnd_thesaurus_basicitten_upinterval.xlsx'
MODEL_PATH = r'D:\thesis\machine_learning\models'
MODEL_FILE = f'model_THESAURUS_ITTEN_INTERVAL_KNeighborsClassifier_KNN23_p2_train1035_cat6_testacc0.87.sav'

LABEL = 'cat1'

OUTPUT_PATH = r'D:\thesis\film_colors_project\sample-dataset'
OUTPUT_FILE = 'KNN21_dataset_2.xlsx'


#%%

def load_palette(path, file):     
    os.chdir(path)
    palette = pd.read_csv(file, index_col=0)
    return palette
    

def load_files(path, extension): 
    files = []
    for r, d, f in os.walk(path): 
        for file in f:
            if extension in file:
                files.append(file) 
    files = sorted(files)
    return files 

def load_data(path, file): 
    os.chdir(path)
    data = pd.read_excel(file, sep=" ", index_col=0)
    data.head()
    data.info()
    return data 

def make_label_encoder(label, data): 
    lab2pt = data[label].tolist()
    le = preprocessing.LabelEncoder()   
    le.fit(lab2pt) 
    set(le.transform(lab2pt)) 
    le.inverse_transform([3])
    le.classes_
    return le 

def categorize_color(color_lab, clf): 
    label = clf.predict([color_lab]) 
    label = le.inverse_transform(label)
    label = label.tolist()[0]         
    return label 

def load_model(path, file): 
    os.chdir(path)
    clf = pickle.load(open(file, 'rb'))
    return clf 

def get_contrasts_all_palettes(ctrstd_palettes): 
    """ save all predictions of contrasts for all color palettes"""         
    ids = []    
    coh = []
    ldc = []
    cwc = []
    cc = []
    gr = []
    bo = []
    vy = []
    cs = []
    ce = []    
    red = []
    orange = []
    yellow = []
    green  = []
    blue = []
    violet = []    
    dark = []
    light = []   
    saturated = []
    desaturated = []
       
    for i in range(len(ctrstd_palettes)):
        el = ctrstd_palettes[i][2]
        ids.append(el)
        if 'coh' in ctrstd_palettes[i][-4]: 
            coh.append(1)
        else: 
            coh.append(0)
        if 'ldc' in ctrstd_palettes[i][-4]: 
            ldc.append(1)
        else: 
            ldc.append(0)
        if 'cwc' in ctrstd_palettes[i][-4]: 
            cwc.append(1)
        else: 
            cwc.append(0)
        if 'cc' in ctrstd_palettes[i][-4]: 
            cc.append(1)
        else: 
            cc.append(0)
        if 'cc: g-r' in ctrstd_palettes[i][-4]: 
            gr.append(1)
        else: 
            gr.append(0)
        if 'cc: b-o' in ctrstd_palettes[i][-4]: 
            bo.append(1)
        else: 
            bo.append(0)
        if 'cc: v-y' in ctrstd_palettes[i][-4]: 
            vy.append(1)
        else: 
            vy.append(0)
        if 'cs' in ctrstd_palettes[i][-4]: 
            cs.append(1)
        else: 
            cs.append(0)
        if 'ce' in ctrstd_palettes[i][-4]: 
            ce.append(1)
        else: 
            ce.append(0)
        if 'red' in ctrstd_palettes[i][-3].index: 
            red.append(round(ctrstd_palettes[i][-3].loc['red'][0],2))
        else: 
            red.append(0)
        if 'orange' in ctrstd_palettes[i][-3].index: 
            orange.append(round(ctrstd_palettes[i][-3].loc['orange'][0],2))
        else: 
            orange.append(0)
        if 'yellow' in ctrstd_palettes[i][-3].index: 
            yellow.append(round(ctrstd_palettes[i][-3].loc['yellow'][0],2))
        else: 
            yellow.append(0)
        if 'green' in ctrstd_palettes[i][-3].index: 
            green.append(round(ctrstd_palettes[i][-3].loc['green'][0],2))
        else: 
            green.append(0)
        if 'blue' in ctrstd_palettes[i][-3].index: 
            blue.append(round(ctrstd_palettes[i][-3].loc['blue'][0],2))
        else: 
            blue.append(0)  
        if 'violet' in ctrstd_palettes[i][-3].index: 
            violet.append(round(ctrstd_palettes[i][-3].loc['violet'][0],2))
        else: 
            violet.append(0)  
        if 'dark' in ctrstd_palettes[i][-2].index: 
            dark.append(round(ctrstd_palettes[i][-2].loc['dark'][0],2))
        else: 
            dark.append(0) 
        if 'light' in ctrstd_palettes[i][-2].index: 
            light.append(round(ctrstd_palettes[i][-2].loc['light'][0],2))
        else: 
            light.append(0) 
        if 'saturated' in ctrstd_palettes[i][-1].index: 
            saturated.append(round(ctrstd_palettes[i][-1].loc['saturated'][0],2))
        else: 
            saturated.append(0) 
        if 'desaturated' in ctrstd_palettes[i][-1].index: 
            desaturated.append(round(ctrstd_palettes[i][-1].loc['desaturated'][0],2))
        else: 
            desaturated.append(0) 
    
    data = {'Palette_id': ids,
            'Contrast of hue': coh, 
            'Light-dark contrast': ldc,
            'Cold-warm contrast': cwc,
            'Complementary contrast': cc, 
            'Green-red': gr, 
            'Blue-orange': bo,
            'Violet-yellow': vy, 
            'Contrast of saturation': cs,
            'Contrast of extension': ce,
            
            'Red': red,
            'Orange': orange,
            'Yellow': yellow, 
            'Green': green, 
            'Blue': blue, 
            'Violet': violet, 
            
            'Dark': dark, 
            'Light': light, 
            
            'Saturated': saturated,
            'Desaturated': desaturated}  
    
    data = pd.DataFrame(data)
    data = data.sort_values(by='Palette_id', ascending=True).reset_index(drop=True)
    rnbw = list(zip([r if r == 0 else 1 for r in list(data['Red'])], [o if o == 0 else 1 for o in list(data['Orange'])],[y if y == 0 else 1 for y in list(data['Yellow'])], [g if g == 0 else 1 for g in list(data['Green'])], [b if b == 0 else 1 for b in list(data['Blue'])], [v if v == 0 else 1 for v in list(data['Violet'])]))
    data['colors'] = [sum(rnb) for rnb in rnbw]
    warm = list(zip([r for r in list(data['Red'])], [o for o in list(data['Orange'])],[y for y in list(data['Yellow'])]))
    kalt = list(zip([g for g in list(data['Green'])], [b for b in list(data['Blue'])], [v for v in list(data['Violet'])]))
    data['warm'] = [round(sum(wrm),2) for wrm in warm]
    data['cold'] = [round(sum(klt),2) for klt in kalt]
    total = [sum(el) for el in list(zip([sa for sa in list(data['Saturated'])], [de for de in list(data['Desaturated'])]))]
    sat_ratio = [round(el[0]/el[1],2) for el in list(zip([s for s in list(data['Saturated'])], [t for t in total]))]
    desat_ratio = [round(el[0]/el[1],2) for el in list(zip([d for d in list(data['Desaturated'])], [t for t in total]))]
    data['sat-ratio'] = sat_ratio
    data['desat-ratio'] = desat_ratio
    return data
   
 

def save_allpalettes_contrasts_excel(data, path, file): 
    os.chdir(path)
    data.to_excel(file) 
    
#%%

if __name__ == '__main__': 
    
    cvs_palettes = load_files(CSV_PALETTE_PATH, CSV_PALETTE_EXTENSION)
    img_palettes = load_files(IMG_PALETTE_PATH, IMG_PALETTE_EXTENSION)
    images = load_files(IMG_PATH, IMG_EXTENSION)
    

    assert len(cvs_palettes) == len(images) , 'Number of color palettes not equal to number of images.'
    assert len(img_palettes) == len(images) , 'Number of color palettes not equal to number of images.'
       

#%%

    # load data  
    data = load_data(DICT_PATH, DICT_FILE)
    le = make_label_encoder(LABEL, data)
    clf = load_model(MODEL_PATH, MODEL_FILE) 
    

#%%
    
    # processing data
    all_palettes = get_palettes_pool(CSV_PALETTE_PATH, csv_palettes)
    palette_names = rename_palette_files(csv_palettes)    
    all_palettes_extended = get_palette_in_lab_add_ratio(all_palettes)
    palette_colors_cats, palettes = get_palette_color_categories_names_pred(data, all_palettes_extended, clf)
    ctrstd_palettes = categorize_palette_to_contrast(palettes, palette_names)
    data = get_contrasts_all_palettes(ctrstd_palettes)
    
    # save data 
    save_allpalettes_contrasts_excel(data, OUTPUT_PATH, OUTPUT_FILE)    


