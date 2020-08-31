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
import pandas as pd

# to specify: USER SPECIFICATION 
# filters
# you want all palettes with a certain contrast 
SEARCH_COLOR_CONTRAST = 'coh' 
PALETTE_DEPTH = 'row 20' # ['row 1','row 20'] (top-to-bottom hierarchy)
COLORBAR_COUNT = 10

COLOR_CONTRASTS = ['coh', 'ldc', 'cwc', 'cc', 'cc: g-r', 'cc: b-o', 'cc: v-y', 'cs', 'ce']
print('Number of color contrasts: ', len(COLOR_CONTRASTS))
#%%
# load color palettes dataframe 
# palette/s
PALETTE_PATH = r'D:\thesis\film_colors_project\sample-dataset\screenshots\7\D0_100_bgr_palette_csv'
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

print('Number of color palettes: ', len(sorted(FILES)))

IMG_PATH = r'D:\thesis\film_colors_project\sample-dataset\screenshots\7\images'
EXTENSION = '.jpg'
IMGS = []
for r, d, f in os.walk(IMG_PATH): # r=root, d=directories, f = files
    for file in f:
        if EXTENSION in file:
            IMGS.append(file) 
            
print('Number of images: ', len(sorted(IMGS)))

assert len(FILES) == len(IMGS), 'Number of color palettes not equal to number of images.'

#%%
# load color name dictionary 
from sklearn import preprocessing
            
### Color-Thesaurus EPFL ###
SEARCH_COLORS_PATH = r'D:\thesis\input_color_name_dictionaries\thesaurus\datasets'
SEARCH_COLORS_FILE = 'eeffcnd_thesaurus_basicitten_upinterval.xlsx'

# set directory 
os.chdir(SEARCH_COLORS_PATH)

# load data 
data = pd.read_excel(SEARCH_COLORS_FILE, sep=" ", index_col=0)
data.head()
data.info()

lab2pt = data['cat1'].tolist() #list(df.index)
le = preprocessing.LabelEncoder()

le.fit(lab2pt) # fit all cat1 colors 
set(le.transform(lab2pt)) 
le.inverse_transform([3])
le.classes_


#%%
from sklearn.neighbors import KNeighborsClassifier
# to specify
ML_MODELS_PATH = r'D:\thesis\machine_learning\models'


ML_MODELS_FILE = f'model_THESAURUS_ITTEN_INTERVAL_KNeighborsClassifier_KNN23_p2_train1035_cat6_testacc0.87.sav'

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
def convert_colors(col, conversion=cv2.COLOR_BGR2Lab): #default 
    color = cv2.cvtColor(np.array([[col]], dtype=np.float32), conversion)[0, 0]
    return color

# convert numpy array of colors
def convert_array(nparray, origin, target='RGB'): 
    """helper function: convert_color """
    # convert to RGB
    converted_colors = []
    for col in nparray: 
        if origin == 'BGR' and target == 'RGB':        
            converted_color = convert_colors(col, cv2.COLOR_BGR2RGB)
        if origin == 'LAB' and target == 'RGB':     
            converted_color = convert_colors(col, cv2.COLOR_LAB2RGB)*255
        if origin == 'RGB' and target == 'LAB':     
            converted_color = convert_colors(col, cv2.COLOR_RGB2LAB)
        if origin == 'HSV' and target == 'RGB':     
            converted_color = convert_colors(col, cv2.COLOR_HSV2RGB)*255
        if origin == 'RGB' and target == 'HSV':     
            converted_color = convert_colors(col, cv2.COLOR_RGB2HSV)
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
palet_nms = [f[:-16] for f in FILES]  
print(f"Searching a total of {len(palet_names)} palettes. ")
#Searching a total of 569 palettes.
print("Searching your chosen color in first five examples of palettes: \n", ', '.join(palet_names[:5]), '.')



#%%
import sys
sys.path.append(r'D:\thesis\code')
from ColorConversion00 import convert_color

# add other cs values to palette data

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
print(len(cp_pool), 'to be processed.')
for i, palette in enumerate(cp_pool): 
    print(i, 'processed')
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

print('Palettes processed: ', len(palettinis))   
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
#    cp = palettinis[315] # 365, 439, 476, 478, 479, 485 (borderline)
    contrasts = []     
    # make lumens
    cp['lumens'] = None
    cp['lumens'][cp['l'] > 55] = 'light'
    cp['lumens'][cp['l'] < 25] = 'dark'
    lumens = cp[['lumens', 'ratio_width']].groupby('lumens').agg('sum').sort_values(by='ratio_width', ascending=False)
    # make saturation
    cp['tone'] = None
    cp['tone'][cp['c'] > 50] = 'saturated'
    cp['tone'][cp['c'] < 25] = 'desaturated'
    tone = cp[['tone', 'ratio_width']].groupby('tone').agg('sum').sort_values(by='ratio_width', ascending=False)
    # make hue 
#       remove too dark values
    tafel = cp[cp['lumens'] != 'dark']
      # remove too desaturated values
    tafel = tafel[tafel['tone'] != 'desaturated']   
    tafel = tafel[['ratio_width', 'lumens', 'tone', 'color_cat_prediction']].groupby('color_cat_prediction').agg('sum').sort_values(by='ratio_width', ascending=False)
#    # remove "brown" 
    if 'orange' in tafel.index.tolist(): 
        if tafel.loc['orange'].iloc[0] > 20: 
            tafel = tafel.drop(['orange'])
    
#    # small size needs to be saturated
    small_colors = tafel[tafel['ratio_width'] <= 5].index.tolist()
    sat_colors = set(cp['color_cat_prediction'][cp['tone'] != 'saturated'].tolist())
    for small in small_colors: 
        if small not in sat_colors: 
            tafel.drop([small])
           
    try: 
        tafel = tafel.drop(['nan'])
    except: 
        pass
    tafel_colors = list(tafel.index)
    lumens_colors = list(lumens.index)
    tone_colors = list(tone.index)
    # contrast of hue 
    if len(tafel_colors) > 2: 
#        print('Match')
        contrasts.append('coh')
    # light-dark contrast
    if set(['dark','light']).issubset(lumens_colors) and lumens.loc['dark'].iloc[0]>40:
#        print('Match')
        contrasts.append('ldc') 
        
        
    # keep only dark and saturated values
    tafel = cp[cp['lumens'] != 'dark'] 
    tafel = tafel[tafel['c'] > 30]
    tafel = tafel[['ratio_width', 'lumens', 'tone', 'color_cat_prediction']].groupby('color_cat_prediction').agg('sum').sort_values(by='ratio_width', ascending=False)
#    # remove "brown" 
    if 'orange' in tafel.index.tolist(): 
        if tafel.loc['orange'].iloc[0] > 20: 
            tafel = tafel.drop(['orange'])
    
#    # small size needs to be saturated
    small_colors = tafel[tafel['ratio_width'] <= 5].index.tolist()
    sat_colors = set(cp['color_cat_prediction'][cp['tone'] != 'saturated'].tolist())
    for small in small_colors: 
        if small not in sat_colors: 
            tafel.drop([small])
    tafel_colors = list(tafel.index)
    
    # cold-warm contrast
    if any(x in ['green', 'blue', 'violet'] for x in tafel_colors) and any(x in ['red', 'orange', 'yellow'] for x in tafel_colors):
#        print('Match')
        contrasts.append('cwc')
    # complementary contrast   
    if set(['green','red']).issubset(tafel_colors) or set(['blue','orange']).issubset(tafel_colors) or set(['violet','yellow']).issubset(tafel_colors):
#        print('Match')
        contrasts.append('cc')      
        if set(['green','red']).issubset(tafel_colors):
            contrasts.append('cc: g-r')
        if set(['blue','orange']).issubset(tafel_colors):
            contrasts.append('cc: b-o')
        if set(['violet','yellow']).issubset(tafel_colors):
            contrasts.append('cc: v-y')        
    # contrast of saturation 
    if set(['saturated','desaturated']).issubset(tone_colors):
#        print('Match')
        contrasts.append('cs')        
    # contrast of extension 
        total = tone['ratio_width'].iloc[0]+tone['ratio_width'].iloc[1]
        first_proprz = round(tone['ratio_width'].iloc[0]/total,2) # desaturated: >=0.98
        second_proprz = round(tone['ratio_width'].iloc[1]/total,2) # saturated: <=0.02
        if (first_proprz >= 0.98 and second_proprz <= 0.02 and second_proprz >= 0.005):
#        print('Match')
            contrasts.append('ce')
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
for i in range(3):
    print(ctrstd_palettes[i][2])
    print(ctrstd_palettes[i][-4])
    print(ctrstd_palettes[i][-3])
    print(ctrstd_palettes[i][-2])
    print(ctrstd_palettes[i][-1])

#%%
# save all predictions of contrasts for all color palettes
     
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

#%%

#data.iloc[0]
#data[['Red', 'Orange', 'Yellow', 'Green', 'Blue', 'Violet', 'warm', 'cold']]

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

#%%
os.chdir(r'D:\thesis\film_colors_project\sample-dataset')
#data.to_csv('KNN21_dataset.csv', index=False)
data.to_excel('KNN21_dataset_2.xlsx')
