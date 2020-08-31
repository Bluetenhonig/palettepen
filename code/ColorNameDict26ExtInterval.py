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
from random import choice, sample


# declare variables 
PATH = r'D:\thesis\input_color_name_dictionaries\thesaurus\datasets'
FILE = 'effcnd_thesaurus_basicvian.xlsx'
OUTPUT_FILE = 'effcnd_thesaurus_basicvian.xlsx' 
#OUTPUT_FILE = 'eeffcnd_thesaurus_dictinterval.xlsx' 

DICT_PATH = r'D:\thesis\input_color_name_dictionaries\thesaurus\datasets'
DICT_FILE = 'effcnd_thesaurus_basicvian.xlsx'
DICT_OUTPUT_FILE = 'eeffcnd_thesaurus_basicvian_upinterval.xlsx' 
DICT_THRESHOLD = 10 

#%%

def load_data(path, file):     
    os.chdir(PATH)
    data = pd.read_excel(FILE, sep=" ", index_col=0)
    data = data.dropna(subset=['srgb'])
    data.info()
    data.columns
    return data 


def get_two_val_for_cat_one(data): 
    """ get two more values for color category with one value only 
    - determine instances of color categories with one item only""" 
    unikate = []
    unikate_vals = []
    for i, el in enumerate(list(data['cat1'].value_counts())): 
        if el == 1: 
            unikat = list(data['cat1'].value_counts().keys())[i]
            unikate.append(unikat)
    
    for i in range(len(unikate)): 
        unikat_val = eval(data['cielab'][data['cat1'] == unikate[i]].iloc[0])
        unikate_vals.append(unikat_val)
    return unikate, unikate_vals

def make_two_more_vals(unikate_vals): 
    """ make two new values  """
    unikat_names =  []
    unikates = []
    more_vals = []
    for i, u in enumerate(unikate_vals):
        if u[0] >100 or u[0] < 0: 
            print('true')
        elif u[1] > 128 or u[1] < -128 or u[2]> 128 or u[1] < -128: 
            print('true') 
        else: 
            upper = [el +.5 for el in unikate_vals[i]]
            unikates.append(unikate[i])
            unikat_names.append(f'{unikate[i]}{1}')
            more_vals.append(upper)
            lower = [el -.5 for el in unikate_vals[i]]
            unikates.append(unikate[i])
            unikat_names.append(f'{unikate[i]}{2}')
            more_vals.append(lower)
    return unikat_names, unikates, more_vals
      
def into_dataframe(unikat_names, unikates, more_vals): 
    data3 = pd.DataFrame({
            'id': None,
            'lang': None,
            'name': unikat_names,
            'srgb': None,
            'srgb_R': None,
            'srgb_G': None,
            'srgb_B': None,
            'hsv': None,
           'hsv_H': None
           , 'hsv_S': None
           , 'hsv_V': None
           , 'cielab': more_vals
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
            ,'cat1': unikates
            , 'cat2' : None
            })
    return data3 

def make_unique_vales(data): 
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
    return data

def build_columns_ext(data): 
    r_rgb = data['srgb_R'].tolist()
    g_rgb = data['srgb_G'].tolist()
    b_rgb = data['srgb_B'].tolist()
    rp1 = [r+1 if r!=255 else r for r in r_rgb] 
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
           
    return final_rgb_R, final_rgb_G, final_rgb_B, lst_len, indices


def final_columns(final_rgb_R, final_rgb_G, final_rgb_B, lst_len, indices): 
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
    return lang, final_lang, final_name, final_cat, final_srgb

def restructure(lang, final_lang, final_name, final_cat, final_srgb): 
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
    return data2


def make_df(data2, save=False): 
    """unique values, all double-singletons, all double-pairs, 
    make unique srgb values, for equal rgb, get last name number's indices,
    make unique srgb values ie remove last number's indices, 
    names with high ordinal nb to drop, names wt/o nb to stay 
    
    """ 
    duplname2 = data2['name'].duplicated().sum()
    print("Number of duplicated color names: ", duplname2)
    duplrgb2 = data2['srgb'].duplicated().sum()
    print("Number of duplicated rgb values: ", duplrgb2)
    dupllab2 = data2['cielab'].duplicated().sum()
    print("Number of duplicated lab values: ", dupllab2)

    double2 = data2[['name', 'srgb']][data2['srgb'].duplicated()]
    dubl2 = pd.DataFrame(columns=['name', 'srgb'])
    for rgb in double2['srgb'].tolist(): 
        dfdubl = data2[['name', 'srgb']][data2['srgb']== rgb]
        dubl2 = dubl2.append(dfdubl)
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
    print("Removing duplicate rgb values")
    twobear = dubl2.groupby(['srgb'])['number'].count()[dubl2.groupby(['srgb'])['number'].count() ==2]
    twobear = twobear.to_frame()
    twobear['srgb'] = twobear.index
    twobear = dubl2[dubl2['srgb'].isin(twobear['srgb'].tolist())]  
    idx = twobear.groupby(['srgb'])['number'].transform(max) == twobear['number']
    bull = twobear[idx]
    threebear = dubl2.groupby(['srgb'])['number'].count()[dubl2.groupby(['srgb'])['number'].count() !=2]
    threebear = threebear.to_frame()
    threebear['srgb'] = threebear.index
    threebear = dubl2[dubl2['srgb'].isin(threebear['srgb'].tolist())]
    
    idx = threebear.groupby(['srgb'])['number'].transform(min) != threebear['number']
    bull3 = threebear[idx]
    data2 = data2.drop(bull.index)
    data2 = data2.drop(bull3.index)   
    duplany = data2['srgb'].duplicated().any()
    print("Number of duplicates: ", duplany)
    
    data2['cat'].value_counts()
    data2['srgb_R'].max()
    data2['srgb_G'].max()
    data2['srgb_B'].max()
    if save: 
        data2.to_excel(OUTPUT_FILE)
    return data2

def show_valuecounts(data): 
    """show imbalanced classes """
    data['cat1'].value_counts()
    data['cat2'].value_counts()
    return data 

def split_data(data): 
    """ split dataset into one cats and double cats """
    datacats = data[data['cat2'].apply(lambda x: isinstance(x, str))]
    datacat = data[data['cat2'].apply(lambda x: isinstance(x, float))]

    assert datacats.shape[0] + datacat.shape[0] == data.shape[0]
    datacat['cat1'].value_counts().shape[0]
    datacat['cat1'].value_counts().iloc[0]
    datacats.groupby(['cat1','cat2']).size()
    max(datacats.groupby(['cat1','cat2']).size())
    return datacats, datacat


def make_missings_or_lowten(datacats, datacat, threshold): 
    colorlowten = []
    for key, nb in datacat['cat1'].value_counts().items(): 
        if nb <threshold: 
            colorlowten.append(key)           
    dict_missing_nb = {}
    for k,i in datacat['cat1'].value_counts().items(): 
        if k in colorlowten: 
            dict_missing_nb[k] = threshold - i    
    dict_missing_nb2 = {}
    for k,i in datacats.groupby(['cat1','cat2']).size().items(): 
        dict_missing_nb2[k] = max(datacats.groupby(['cat1','cat2']).size()) - i
    
    return colorlowten, dict_missing_nb, dict_missing_nb2



def get_missing_vals(dict_missing_nb, data): 
    """ one label """
    dict_missing_vals = {}
    for key, i in dict_missing_nb.items(): 
        if i == 0: 
            pass
        else: 
            new_values = []
            data[data['cat1'] == key]
            # generate new values 
            for i, el in enumerate(range(i)): 
                a= [eval(n) for n in data['srgb'][data['cat1'] == key]]   
                # random srgb color value of example ultramarine with replacement 
                random = choice(a)
                # random's color names
                data['name'][data['srgb']==''.join(str(random).split(' '))]
                # bootstrap color value by samplling without replacement 
                snatch = sample(random, 1)[0] 
                if snatch == 0: 
                    increment = 1
                elif snatch == 255: 
                    increment = -1         
                else: 
                    increment = choice([-1,1])         
                new_snatch = snatch +increment
                idx = random.index(snatch)
                indices = [0,1,2]
                indices= [i for i in indices if i != idx ]
                idel1 = indices[0]
                idel2 = indices[1]
                idx_values = []
                idel1_values = []
                idel2_values = []
                for val in new_values: 
                    idx_values.append(val[idx])
                    idel1_values.append(val[idel1])
                    idel2_values.append(val[idel2])
                try: 
                    if increment == 1 and max(idx_values)<255: 
                        new_snatch = max(idx_values) +increment
                    else: 
                        new_snatch = min(idx_values) +increment                                   
                except: 
                    pass
                
                new_value = []
                for el in random: 
                    if snatch == el: 
                        el = new_snatch
                        new_value.append(el)
                    else: 
                        new_value.append(el)   
                
                if new_value in new_values: 
                    new_value = None
                    mini = min(new_values)
                    maxi = max(new_values)
                    if mini[0] !=0 and mini[1] != 0 and mini[2] != 0:
                        new_value = [mini[0]-1, mini[1]-1, mini[2]-1]
                    elif maxi[0] !=255 and maxi[1] != 255 and maxi[2] != 255:
                        new_value = [maxi[0]+1, maxi[1]+1, maxi[2]+1]
                new_values.append(new_value)
            assert i == len(new_values)
            dict_missing_vals[key] = new_values
    return dict_missing_vals
        

def make_color_cat_lang_list(dict_missing_vals): 
    colorlist = []
    rgblist = []
    catlist = []
    langlist = []
    for key, item in dict_missing_vals.items():
        for j, i in enumerate(item): 
            #print(key, j+1)
            colorlist.append(key + str(j+1))
            rgblist.append(i)
            catlist.append(key)
            langlist.append('eng')           
    r_rgblist = [int(el[0]) for el in rgblist]
    g_rgblist = [int(el[1]) for el in rgblist]
    b_rgblist = [int(el[2]) for el in rgblist]
    return colorlist, langlist, rgblist, r_rgblist, g_rgblist, b_rgblist, catlist


def put_into_dataframe(colorlist, langlist, rgblist, r_rgblist, g_rgblist, b_rgblist, catlist): 
    data2 = pd.DataFrame({
            'id': range(len(langlist)),
            'lang': langlist,
            'name': colorlist,
            'srgb': rgblist,
            'srgb_R': r_rgblist,
            'srgb_G': g_rgblist,
            'srgb_B': b_rgblist,
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
            ,'cat1': catlist
            })
    
    assert data2['srgb_R'].min() > 0 and data2['srgb_G'].min() > 0 and data2['srgb_B'].min() > 0
    assert data2['srgb_R'].max() < 255 and data2['srgb_G'].max() < 255 and data2['srgb_B'].max() < 255
    return data2

def get_multi_missing_vals(dict_missing_nb2, data): 
    dict_missing_vals = {}
    for key, i in dict_missing_nb2.items(): 
        #print(key, i)
        if i == 0: 
            pass
        else: 
            new_values = []
            one = key[0]
            two = key[1]
            data['srgb'][(data['cat1']==one) & (data['cat2']==two)] #[241,197,166]
            # generate new values 
            for el in range(i): 
                a= [eval(f) for f in data['srgb'][(data['cat1']==one) & (data['cat2']==two)]]   
                # random srgb color value of example ultramarine with replacement 
                random = choice(a)
                # random's color names
                data['name'][data['srgb']==''.join(str(random).split(' '))]
                # bootstrap color value by samplling without replacement 
                snatch = sample(random, 1)[0] 
                if snatch == 0: 
                    increment = 1
                elif snatch == 255: 
                    increment = -1         
                increment = choice([-1,1])         
                new_snatch = snatch +increment
                idx = random.index(snatch)
                indices = [0,1,2]
                indices= [i for i in indices if i != idx ]
                idel1 = indices[0]
                idel2 = indices[1]
                idx_values = []
                idel1_values = []
                idel2_values = []
                for val in new_values: 
                    idx_values.append(val[idx])
                    idel1_values.append(val[idel1])
                    idel2_values.append(val[idel2])
                try: 
                    if increment == 1 and max(idx_values)<255: 
                        new_snatch = max(idx_values) +increment
                    else: 
                        new_snatch = min(idx_values) +increment                                   
                except: 
                    pass
                
                new_value = []
                for el in random: 
                    if snatch == el: 
                        el = new_snatch
                        new_value.append(el)
                    else: 
                        new_value.append(el)   
                
                if new_value in new_values: 
                    new_value = None
                    mini = min(new_values)
                    maxi = max(new_values)
                    if mini[0] !=0 and mini[1] != 0 and mini[2] != 0:
                        new_value = [mini[0]-1, mini[1]-1, mini[2]-1]
                    elif maxi[0] !=255 and maxi[1] != 255 and maxi[2] != 255:
                        new_value = [maxi[0]+1, maxi[1]+1, maxi[2]+1]
                #checked: no duplicate values found, random not in new_values  
                new_values.append(new_value)
            assert i == len(new_values)
            dict_missing_vals[key] = new_values
    return dict_missing_vals

def make_multi_color_cat_lang_list(dict_missing_vals): 
    colorlist = []
    rgblist = []
    catlist1 = []
    catlist2 = []
    langlist = []
    for key, item in dict_missing_vals.items():
        for j, i in enumerate(item): 
            keyn = ' '.join(key)+ str(j+1000)
            colorlist.append(keyn)
            rgblist.append(i)
            catlist1.append(key[0])
            catlist2.append(key[1])
            langlist.append('eng')
    
            
    r_rgblist = [int(el[0]) for el in rgblist]
    g_rgblist = [int(el[1]) for el in rgblist]
    b_rgblist = [int(el[2]) for el in rgblist]
    return colorlist, langlist, rgblist, r_rgblist, g_rgblist, b_rgblist, catlist1, catlist2 


def make_frame(colorlist, langlist, rgblist, r_rgblist, g_rgblist, b_rgblist, catlist1, catlist2): 
    data3 = pd.DataFrame({
            'id': range(len(langlist)),
            'lang': langlist,
            'name': colorlist,
            'srgb': rgblist,
            'srgb_R': r_rgblist,
            'srgb_G': g_rgblist,
            'srgb_B': b_rgblist,
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
            ,'cat1': catlist1
            , 'cat2' : catlist2
            })
    return data3

def merge_frames(data, data3, save=False): 
    dataset = pd.concat([data, data3]).reset_index(drop=True)
    dataset = dataset.sort_index()
    dataset['cat1'].value_counts()
    dataset.groupby(['cat1','cat2']).size()
    if save: 
        dataset.to_excel(OUTPUT_FILE)
    return dataset

#%%
    
if __name__ == '__main__': 
    
    data = load_data(PATH, FILE)


    unikate, unikate_vals = get_two_val_for_cat_one(data)
    unikat_names, unikates, more_vals = make_two_more_vals(unikate_vals)
    data3 = into_dataframe(unikat_names, unikates, more_vals)
    data = make_unique_vales(data)


# RGB
    final_rgb_R, final_rgb_G, final_rgb_B, lst_len, indices = build_columns_ext(data)
    lang, final_lang, final_name, final_cat, final_srgb = final_columns(final_rgb_R, final_rgb_G, final_rgb_B, lst_len, indices)

    data2 = restructure(lang, final_lang, final_name, final_cat, final_srgb)
    data2 = make_df(data2, save=False)
    
 
# upsampling    
    data = load_data(DICT_PATH, DICT_FILE)
    datacats, datacat = split_data(data)
    colorlowten, dict_missing_nb, dict_missing_nb2 = make_missings_or_lowten(datacats, datacat, DICT_THRESHOLD)
    # one label
    dict_missing_vals = get_missing_vals(dict_missing_nb, data)
    colorlist, langlist, rgblist, r_rgblist, g_rgblist, b_rgblist, catlist = make_color_cat_lang_list(dict_missing_vals)
    # multi-label
    dict_missing_vals = get_multi_missing_vals(dict_missing_nb2, data)
    colorlist, langlist, rgblist, r_rgblist, g_rgblist, b_rgblist, catlist1, catlist2 = make_multi_color_cat_lang_list(dict_missing_vals)
    data3 = make_frame(colorlist, langlist, rgblist, r_rgblist, g_rgblist, b_rgblist, catlist1, catlist2)

    dataset = merge_frames(data, data3, save=False)


