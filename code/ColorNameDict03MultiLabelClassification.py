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

### Color-name Dictionary CND ###

#%%
### Color-Survey ###

PATH = r'D:\thesis\input_color_name_dictionaries\survey'
FILE = 'satfaces.xlsx'

# set directory 
os.chdir(PATH)

# load data 
data = pd.read_excel(FILE, sep=" ")
data.tail()
data['thesaurus'].value_counts()

#%%

### Color-Thesaurus EPFL ###

PATH = r'D:\thesis\input_color_name_dictionaries\thesaurus\datasets'
FILE = 'effcnd_thesaurus_vian.xlsx'
OUTPUT_FILE = 'effcnd_thesaurus_vian.xlsx' 

# set directory 
os.chdir(PATH)

# load data 
data = pd.read_excel(FILE, sep=" ")
data.info()
data = data.dropna()
data['name2'] = data['name'] 


#%%
# multilabel classification: find border colors
datalst = data['name2'].str.split().tolist()
# get two words only
index = []
datalstt = []
for i, l in enumerate(datalst): 
    if len(l)==2: 
        index.append(i)
        datalstt.append(l)



twowords = pd.DataFrame({'id': index,
                         'word1': [l[0] for l in datalstt], 
                         'word2': [l[1] for l in datalstt]})


#%%

# preprocessing 

# for VIAN basic colors only 
# 
# recodings         
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


# border colors: bluegreen, bluegrey, greenblue, greyblue, orangered, yellowgreen

#%%
### Basic Colors ###

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

#%%


# if word 1 is matchable to a vian color, cat2 is found
flt1 = twowords[twowords['word1'].isin(vian_hues)]
flt2 = twowords[~twowords['word1'].isin(vian_hues)]


#%%

# restructure
data = data.drop(['name2'], axis=1)
flt1 = flt1[['id','word1']]
data = pd.merge(data, flt1.rename(columns={'word1':'cat2'}), on='id', how='left') 


#%%

# save data (if adobe is first color! )
data.to_excel(OUTPUT_FILE)

