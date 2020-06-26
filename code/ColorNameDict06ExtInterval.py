# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 09:11:42 2020

@author: lsamsi



The extention will extend the EFFCND to an Extended-EFFCND (EEFCND) dataframe.  

The extentions steps: 
    
1. EFFCND: remove "cat1" and "cat2" and copy-past column "name" to column "cat"
2. for columns srgb_r, srgb_g, srgb_b calculate +- 1 and make 6 more rows for each new value:
a. srgb_r + 1, srgb_g, srgb_b
b. srgb_r - 1, srgb_g, srgb_b
c. srgb_r, srgb_g + 1, srgb_b
... 
3. concatenated three columns srgb_r, srgb_g, srgb_b to srgb
4. fill out remaining cells if possible (not possible for color name) using duplication or color conversion methods

columns =  [id, lang, name, image, srgb, srgb_r, srgb_g, srgb_b, hsv, hsv_h, hsv_s, hsv_v, lab, lab_l, lab_a, lab_b, hex, cat]
filename = "eeffcnd_"+source+"_"+method+".xlsx" where method: interval 


Goal: extend all unique color names with color values that can be categroized into them 


Step Before: EFFCND (dict colors classified into basic colors)
Goal: EEFCND (no basic colors ie basic colors equal to dict colors, new color values for each dict color)
Step AFter: visualization (?), train machine learning model 
    
"""
# load modules
import os
import pandas as pd
import numpy as np

### Color-name Dictionary CND ###


#%%

### Color-Thesaurus EPFL ###

PATH = r'D:\thesis\input_color_name_dictionaries\thesaurus\datasets'
FILE = 'effcnd_thesaurus_vian.xlsx'
OUTPUT_FILE = 'eeffcnd_thesaurus_interval.xlsx' 

# set directory 
os.chdir(PATH)

# load data 
data = pd.read_excel(FILE, sep=" ", index_col=0)
data.info()
data.columns

data = data.drop(['cat1', 'cat2'], axis=1)
data['cat'] = data['name']
data[data['name'] == 'ultramarine']

# unique values
duplname = data['name'].duplicated().sum()
print("Number of duplicated color names: ", duplname)
duplrgb = data['srgb'].duplicated().sum()
print("Number of duplicated rgb values: ", duplrgb)
dupllab = data['cielab'].duplicated().sum()
print("Number of duplicated lab values: ", dupllab)

# all double-singletons 
double = data[['name', 'srgb']][data['srgb'].duplicated()]
# all double-pairs 
dubl = pd.DataFrame(columns=['name', 'srgb'])
for rgb in double['srgb'].tolist(): 
    dfdubl = data[['name', 'srgb']][data['srgb']== rgb]
    dubl = dubl.append(dfdubl)

# make unique srgb values
data = data.drop(double.index)


#%%
# used excel macro for first 150 data rows

# build extended columns as lists  
r_rgb = data['srgb_R'].tolist()
g_rgb = data['srgb_G'].tolist()
b_rgb = data['srgb_B'].tolist()
rp1 = [r+1 if r!=255 else r for r in r_rgb] #TODO: if .. else r in list compreh 
rm1 = [r-1 if r!=0 else r for r in r_rgb]
gp1 = [g+1 if g!=255 else g for g in g_rgb]
gm1 = [g-1 if g!=0 else g for g in g_rgb]
bp1 = [b+1 if b!=255 else b for b in b_rgb]
bm1 = [b-1 if b!=0 else b for b in b_rgb]
lst_len = len(r_rgb)+ len(rp1) + len(rm1)+ len(gp1) + len(gm1)+ len(bp1) + len(bm1)

final_rgb_R = [None] * lst_len
indices = list(range(0, lst_len, 7))
for i, rgb in enumerate(r_rgb):
    final_rgb_R[indices[i]] = rgb
for ids in range(3,7): 
    ggindices = list(range(ids, lst_len, 7))
    for i, rgb in enumerate(r_rgb):
        final_rgb_R[ggindices[i]] = rgb   
        
rpindices = list(range(1, lst_len, 7))
for i, rp in enumerate(rp1):
    final_rgb_R[rpindices[i]] = rp
    
rmindices = list(range(2, lst_len, 7))
for i, rm in enumerate(rm1):
    final_rgb_R[rmindices[i]] = rm


#%%   


final_rgb_G = [None] * lst_len
for ids in range(0,3): 
    for i, rgb in enumerate(g_rgb): 
        final_rgb_G[indices[i]+ ids] = rgb
for ids in range(5,7): 
    for i, rgb in enumerate(g_rgb): 
        final_rgb_G[indices[i]+ ids] = rgb
gpindices = list(range(3, lst_len, 7))
for i, gp in enumerate(gp1):
    final_rgb_G[gpindices[i]] = gp 
gmindices = list(range(4, lst_len, 7))
for i, gm in enumerate(gm1):
    final_rgb_G[gmindices[i]] = gm 
    

#%%
final_rgb_B = [None] * lst_len
for ids in range(0,5):
    for i, rgb in enumerate(b_rgb): 
        final_rgb_B[indices[i]+ids] = rgb
bpindices = list(range(5, lst_len, 7))
for i, bp in enumerate(bp1):
    final_rgb_B[bpindices[i]] = bp 
bmindices = list(range(6, lst_len, 7))
for i, bm in enumerate(bm1):
    final_rgb_B[bmindices[i]] = bm 
 
    
#%%
lang = data['lang'].tolist()
final_lang = ['eng'] * lst_len

   
name = data['name'].tolist()
final_name = [None] * lst_len
for ids in range(0,7): 
    for i, na in enumerate(name): 
        if ids == 0:
            final_name[indices[i]+ids] = na 
        else: 
            final_name[indices[i]+ids] = na + f' {ids}' 


final_cat = [None] * lst_len
for ids in range(0,7): 
    for i, na in enumerate(name): 
        final_cat[indices[i]+ids] = na


final_srgb = [None] * lst_len
for i in range(lst_len): 
    final_srgb[i] = str([final_rgb_R[i], final_rgb_G[i], final_rgb_B[i]])



    
#%%

# restructure
data2 = pd.DataFrame({
        'id': range(lst_len),
        'lang': final_lang,
        'name': final_name,
        'srgb': final_srgb,
        'srgb_R': final_rgb_R,
        'srgb_G': final_rgb_G,
        'srgb_B': final_rgb_B,
        'hsv': None,
       'hsv_H': None
       , 'hsv_S': None
       , 'hsv_V': None
       , 'cielab': None
       , 'cielab_L': None
       , 'cielab_a': None
       , 'cielab_b': None
       ,'hsl': None
       , 'hsl_H': None
       , 'hsl_S': None
       , 'hsl_L': None
       , 'LCH': None
       , 'LCH_L': None
       , 'LCH_C': None
       , 'LCH_H': None
       , 'hex': None
        ,'cat': final_cat
        })


#%%

# unique values
duplname2 = data2['name'].duplicated().sum()
print("Number of duplicated color names: ", duplname2)
duplrgb2 = data2['srgb'].duplicated().sum()
print("Number of duplicated rgb values: ", duplrgb2)
dupllab2 = data2['cielab'].duplicated().sum()
print("Number of duplicated lab values: ", dupllab2)
  

# all double-singletons 
double2 = data2[['name', 'srgb']][data2['srgb'].duplicated()]
# all double-pairs 
dubl2 = pd.DataFrame(columns=['name', 'srgb'])
for rgb in double2['srgb'].tolist(): 
    dfdubl = data2[['name', 'srgb']][data2['srgb']== rgb]
    dubl2 = dubl2.append(dfdubl)


# make unique srgb values
dubl2 = dubl2.drop_duplicates()

dub = []
for name in dubl2['name']: 
    try: 
        el = int(name[-1]) 
    except: 
        el = 0 
    dub.append(el)

dubl2['number'] = dub 
len(dubl2)

# for equal rgb, get last name number's indices 
print("Removing duplicate rgb values")
# apply to count = 2
twobear = dubl2.groupby(['srgb'])['number'].count()[dubl2.groupby(['srgb'])['number'].count() ==2]
twobear = twobear.to_frame()
twobear['srgb'] = twobear.index
twobear = dubl2[dubl2['srgb'].isin(twobear['srgb'].tolist())]

idx = twobear.groupby(['srgb'])['number'].transform(max) == twobear['number']
bull = twobear[idx]

# apply to count = 3
threebear = dubl2.groupby(['srgb'])['number'].count()[dubl2.groupby(['srgb'])['number'].count() !=2]
threebear = threebear.to_frame()
threebear['srgb'] = threebear.index
threebear = dubl2[dubl2['srgb'].isin(threebear['srgb'].tolist())]

idx = threebear.groupby(['srgb'])['number'].transform(min) != threebear['number']
bull3 = threebear[idx]

# make unique srgb values ie remove last number's indices
# names with high ordinal nb to drop, names wt/o nb to stay 
data2 = data2.drop(bull.index)
data2 = data2.drop(bull3.index)

duplany = data2['srgb'].duplicated().any()
print("Number of duplicates: ", duplany)

data2['cat'].value_counts()
data2['srgb_R'].max()
data2['srgb_G'].max()
data2['srgb_B'].max()
#%%

# save data (if adobe is first color! )
data2.to_excel(OUTPUT_FILE)

