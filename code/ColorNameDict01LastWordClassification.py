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
1. take only last of two words for color name
2. recode color names to basic colors 

Step Before: Preprocessing to get image array into dataframe 
Goal: last word classification of color names to basic colors 
Step AFter: Basic Color Classification, manual classification 
    
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
FILE = 'ffcnd_thesaurus.xlsx'
OUTPUT_FILE = 'ffcnd_thesaurus_lastword.xlsx' 

# set directory 
os.chdir(PATH)

# load data 
data = pd.read_excel(FILE, sep=" ")
data.info()
data = data.dropna()

# last word classification: take second word only for color name 
datalst = data['name'].str.split().tolist()
datalstt = [l[-1] for l in datalst]

data['name2cat1'] = datalstt

data['name2cat1'].value_counts()

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
# before preprocessing: 

# filter data where color name is equal to vian basic color 
flt1 = data[data['name2cat1'].isin(vian_hues)]
flt1['name2cat1'].value_counts()
flt1.shape[0]


#%%

# preprocessing 

# for VIAN basic colors only 
# 
# recodings from almost vian colors to vian colors     
# recode blue: blueberry, bluish, darkblue, lightblue -> blue (a Vian color)
data['name2cat1'] = data['name2cat1'].replace(['blueberry', 'bluish', 'darkblue', 'lightblue'], ['blue', 'blue', 'blue', 'blue'])
# recode blue: brownish -> brown (a Vian color)
data['name2cat1'] = data['name2cat1'].replace(['brownish'], ['brown'])
# recode green: darkgreen, greenish -> green
data['name2cat1'] = data['name2cat1'].replace(['darkgreen', 'greenish'], ['green', 'green'])
# recode grey: greyish -> grey 
data['name2cat1'] = data['name2cat1'].replace(['greyish'], ['grey'])
# recode lavender: lavendar -> lavender
data['name2cat1'] = data['name2cat1'].replace(['lavendar'], ['lavender'])
# recode orange: orangeish -> orange 
data['name2cat1'] = data['name2cat1'].replace(['orangeish'], ['orange'])
# recode pink: pinkish, pinky -> pink
data['name2cat1'] = data['name2cat1'].replace(['pinkish', 'pinky'], ['pink', 'pink'])
# recode red: reddish -> red
data['name2cat1'] = data['name2cat1'].replace(['reddish'], ['red'])
# recode yellow: yellowish -> yellow 
data['name2cat1'] = data['name2cat1'].replace(['yellowish'], ['yellow'])


# border colors: bluegreen, bluegrey, greenblue, greyblue, orangered, yellowgreen




#%%

# after preprocessing: 

flt2 = data[data['name2cat1'].isin(vian_hues)]
flt2['name2cat1'].value_counts()
better = round(((flt2.shape[0] -flt1.shape[0]) / flt1.shape[0])*100,2)

print('Recoding will yield a {} % increase in the number of successfully extracted basic color rows intrinsic to the color names.'.format(better))

#%%

# restructure
data = data.drop(['name2cat1'], axis=1)
flt2 = flt2[['id', 'name2cat1']]
data = pd.merge(data, flt2, on='id',  how='left')


#%%

# save data 
data.to_excel(OUTPUT_FILE)

