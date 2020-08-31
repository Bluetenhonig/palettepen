# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 09:11:42 2020

@author: lsamsi

After the original color name dictionary (CND) is sourced, 
it is saved in a folder with the source name as folder name. 

The CND needs to be extended to a fully-fledged color name dictionary (FFCND)
and then into an extended FFCND (EFFCND).

The preprocessing will extend the CND to a EFFCND dataframe.  

The preprocessing steps: 
    
1. original CND (from source) 
2. processed CND = FFCND
columns = [id, lang, name, srgb, srgb_r, srgb_g, srgb_b, hsv, hsv_h, hsv_s, hsv_v, lab, lab_l, lab_a, lab_b, hex]
filename = "ffcnd_"+source+".xlsx"
3. processed FFCND = EFFCND (with 1 system of basic colors)
columns =  [id, lang, name, image, srgb, srgb_r, srgb_g, srgb_b, hsv, hsv_h, hsv_s, hsv_v, lab, lab_l, lab_a, lab_b, hex, cat1, cat2]
filename = "effcnd_"+source+"_"+system+".xlsx" 


Goal: extract all basic color names found already in the color names of the color name dictionary
1. get all border colors out based on two-base-word color names


Step Before: cat1 basic colors 
Goal: cat2 basic colors (appending)
Step AFter: visualization  
    
"""
# load modules
import os
import pandas as pd


#%%
### Color-Survey ###

XKCD_PATH = r'D:\thesis\input_color_name_dictionaries\xkcd'
XKCD_FILE = 'satfaces.xlsx'

EPFL_PATH = r'D:\thesis\input_color_name_dictionaries\thesaurus\datasets'
EPFL_FILE = 'effcnd_thesaurus_basicvian.xlsx'
EPFL_OUTPUT_FILE = 'effcnd_thesaurus_basicvian.xlsx' 


ITTEN_PATH = r'D:\thesis\input_color_name_dictionaries\thesaurus\datasets'
ITTEN_FILE = 'effcnd_thesaurus_basicitten.xlsx'
ITTEN_OUTPUT_FILE = 'effcnd_thesaurus_basicitten.xlsx' 


#%%

# Basic Colors
# 28 Vian colors 
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

# 6 Itten colors 
itten_hues = [
        'red'
        , 'orange'
        , 'yellow'
        , 'green'
        , 'blue'
        , 'violet'
        ]

def load_data(path, file):  
    os.chdir(path) 
    data = pd.read_excel(file, sep=" ", index_col=0)
    data = data.dropna()
    return data


def find_borderline_colors(data): 
    """ multilabel classification: find border colors 
    get two words only"""
    data['name2'] = data['name'] 
    datalst = data['name2'].str.split().tolist() 
    index = []
    datalstt = []
    catt = []
    for i, l in enumerate(datalst): 
        if len(l)==2: 
            index.append(i)
            ca = data['cat1'].iloc[i]
            datalstt.append(l)
            catt.append(ca)
    pack = (index, datalstt, catt)
    return pack

def vian_recode_borderlines(twowords):    
    """ border colors: bluegreen, bluegrey, greenblue, greyblue, orangered, yellowgreen """
    # recode blue: blueberry, bluish, darkblue, lightblue -> blue 
    twowords['word1'] = twowords['word1'].replace(['blueberry', 'bluish', 'darkblue', 'lightblue', 'bluey'], ['blue','blue', 'blue', 'blue', 'blue'])
    # recode blue: brownish -> brown 
    twowords['word1'] = twowords['word1'].replace(['brownish', 'browny'], ['brown', 'brown'])
    # recode green: darkgreen, greenish -> green
    twowords['word1'] = twowords['word1'].replace(['darkgreen', 'greenish'], ['green', 'green'])
    # recode grey: greyish -> grey 
    twowords['word1'] = twowords['word1'].replace(['greyish'], ['grey'])
    # recode lavender: lavendar -> lavender
    twowords['word1'] = twowords['word1'].replace(['lavendar'], ['lavender'])
    # recode orange: orangeish -> orange 
    twowords['word1'] = twowords['word1'].replace(['orangeish'], ['orange'])
    # recode pink: pinkish, pinky -> pink
    twowords['word1'] = twowords['word1'].replace(['pinkish', 'pinky'], ['pink', 'pink'])
    # recode red: reddish -> red
    twowords['word1'] = twowords['word1'].replace(['reddish', 'reddy'], ['red', 'red'])
    # recode yellow: yellowish -> yellow 
    twowords['word1'] = twowords['word1'].replace(['yellowish', 'yellowy'], ['yellow', 'yellow'])
    # recode gold: yellowish -> yellow 
    twowords['word1'] = twowords['word1'].replace(['golden'], ['gold'])
    # recode gold: yellowish -> yellow 
    twowords['word1'] = twowords['word1'].replace(['greeny'], ['green'])
    # recode gold: yellowish -> yellow 
    twowords['word1'] = twowords['word1'].replace(['orangey', 'orangish'], ['orange', 'orange'])
    # recode gold: yellowish -> yellow 
    twowords['word1'] = twowords['word1'].replace(['peachy'], ['peach'])
    # recode gold: yellowish -> yellow 
    twowords['word1'] = twowords['word1'].replace(['purpley', 'purplish', 'purply'], ['purple', 'purple', 'purple'])
    # recode gold: yellowish -> yellow 
    twowords['word1'] = twowords['word1'].replace(['rusty'], ['rust'])
    return twowords



def itten_recode_borderlines(twowords): 
    """  border colors: bluegreen, orangered, yellowgreen """ 
    # recode red: reddish -> red
    twowords['word1'] = twowords['word1'].replace(['reddish', 'reddy'], ['red', 'red'])
    # recode orange: orangeish -> orange 
    twowords['word1'] = twowords['word1'].replace(['orangeish'], ['orange'])
     # recode gold: yellowish -> orange 
    twowords['word1'] = twowords['word1'].replace(['orangey', 'orangish'], ['orange', 'orange'])
    # recode yellow: yellowish -> yellow 
    twowords['word1'] = twowords['word1'].replace(['yellowish', 'yellowy'], ['yellow', 'yellow'])
    #recode green: darkgreen, greenish -> green
    twowords['word1'] = twowords['word1'].replace(['darkgreen', 'greenish'], ['green', 'green'])
    # recode gold: greeny -> green 
    twowords['word1'] = twowords['word1'].replace(['greeny'], ['green'])
    # recode blue: blueberry, bluish, darkblue, lightblue -> blue 
    twowords['word1'] = twowords['word1'].replace(['blueberry', 'bluish', 'darkblue', 'lightblue', 'bluey'], ['blue','blue', 'blue', 'blue', 'blue'])
    # recode gold: purple -> violet 
    twowords['word1'] = twowords['word1'].replace(['purpley', 'purplish', 'purply'], ['violet', 'violet', 'violet'])
    return twowords


def filter_vian_cat2(twowords, vian_hues): 
    """ if word 1 is matchable to a vian color, cat2 is found """
    flt1 = twowords[twowords['word1'].isin(vian_hues)]
    flt2 = twowords[~twowords['word1'].isin(vian_hues)]
    return flt1, flt2

def filter_itten_cat2(twowords, itten_hues): 
    """ if word 1 is matchable to a vian color, cat2 is found """
    flt1 = twowords[twowords['word1'].isin(itten_hues)]
    flt2 = twowords[~twowords['word1'].isin(itten_hues)]
    flt1 =flt1.set_index('id')
    return flt1, flt2

def restructure(data, flt1): 
    data = data.drop(['name2'], axis=1)
    flt1 = flt1[['id','word1']]
    data = pd.merge(data, flt1.rename(columns={'word1':'cat2'}), on='id', how='left') 
    return data

def restructure_itten(data, flt1): 
    data = data.drop(['name2'], axis=1)
    flt1 = flt1[['word1']]
    data = pd.merge(data, flt1.rename(columns={'word1':'cat2'}), left_index=True, right_index=True, how='left') 
    return data

def reset_index(data): 
    data = data.reset_index(drop=True)
    data['id'] = data.index
    return data

def remove_same_cat(twowords): 
    """ remove word1 where is is the same category as word2 """
    ids  = []
    for i in range(twowords.shape[0]): 
        if twowords['word1'].iloc[i] == twowords['cat1'].iloc[i]:
            ids.append(i)
    twowords = twowords.drop(ids)
    return twowords

#%%

if __name__ == '__main__': 
    
    data = load_data(XKCD_PATH, XKCD_FILE)
    # epfl
    data = load_data(EPFL_PATH, EPFL_FILE)
    
    pack = find_borderline_colors(data)
    (index, datalstt, catt) = pack

    twowords = pd.DataFrame({'id': index,
                             'word1': [l[0] for l in datalstt], 
                             'word2': [l[1] for l in datalstt]})

    twowords = vian_recode_borderlines(twowords)
    flt1, flt2 = filter_vian_cat2(twowords, vian_hues)
    data = restructure(data, flt1)

#%%
    # save data 
    data.to_excel(EPFL_OUTPUT_FILE)

#%%
    # itten
    data = load_data(ITTEN_PATH, ITTEN_FILE)
    data = reset_index(data)
    
    pack = find_borderline_colors(data)
    (index, datalstt, catt) = pack
    
    twowords = pd.DataFrame({'id': index,
                             'word1': [l[0] for l in datalstt], 
                             'word2': [l[1] for l in datalstt], 
                             'cat1': catt})

    twowords = itten_recode_borderlines(twowords)
    twowords = remove_same_cat(twowords)
    flt1, flt2 = filter_itten_cat2(twowords, itten_hues)
    data = restructure_itten(data, flt1)
    

#%%

    # save data
    data.to_excel(ITTEN_OUTPUT_FILE)
