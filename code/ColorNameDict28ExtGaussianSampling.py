# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 09:11:42 2020

@author: lsamsi



The extention will extend the EFFCND to an Extended-EFFCND (EEFFCND) dataframe.  

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
Goal: EEFFCND (no basic colors ie basic colors equal to dict colors, new color values for each dict color)
Step AFter: visualization, train machine learning model 
    
Gaussian Sampling: 
    null hypothesis: x comes from a normal distribution
    
"""
import os 
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from scipy import stats
import scipy.stats
import sys
sys.path.append(r'D:\thesis\code')
from Visualization11Color import display_color
# imputation
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import statistics as s 
from statistics import stdev
from random import choice, sample

    
#%%

# declare variables 
DICT_PATH = r'D:\thesis\input_color_name_dictionaries\thesaurus\datasets'
DICT_FILE = 'effcnd_thesaurus_basicitten.xlsx'
DICT_OUTPUT_FILE = 'eeffcnd_thesaurus_basicitten_upinterval.xlsx' 
DICT_THRESHOLD = 10

LABEL = 'cat1'

#%%

def load_data(path, file):     
    os.chdir(path)
    data = pd.read_excel(file, sep=" ", index_col=0)
    data.info()
    data.columns
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

def intersection(lst1, lst2): 
    return set(lst1).intersection(lst2) 


def fill_dataframe(categories, means, stdevs, means2, stdevs2, means3, stdevs3, normals, normals2, normals3, sample_sizes): 
    results = pd.DataFrame({ 'category': categories
                            , 'pop_mean_L': means
                            , 'pop_std_L': stdevs
                            , 'pop_mean_a': means2
                            , 'pop_std_a': stdevs2
                            , 'pop_mean_b': means3
                            , 'pop_std_b': stdevs3
                            , 'pop_normal_L': normals
                            , 'pop_normal_a': normals2
                            , 'pop_normal_b': normals3
                            , 'sample_size': sample_sizes
        })
    return results

def make_lab_frame(lab_l, lab_a, lab_b, category): 
    df2 = pd.DataFrame({'cielab_L': list(lab_l)
                        , 'cielab_a': list(lab_a)
                        , 'cielab_b': list(lab_b)
                        , 'cielab': [list(i) for i in list(zip(lab_l, lab_a, lab_b))]
                        , 'cat1' : np.repeat(category, MORE)})
    
    df2 = df2.sort_index()
    return df2


def show_color(df2): 
    """display colors of newly sampled values """
    for i in range(len(df2['cielab'])): 
        display_color(df2['cielab'].iloc[i], None, origin='LAB')
        

def get_gaussian_distr(categories): 
    VALUES =['cielab_L', 'cielab_a', 'cielab_b']
    cat_list = []
    means = []
    stdevs = []
    means2 = []
    stdevs2 = []
    means3 = []
    stdevs3 = []
    normals = []
    normals2 = []
    normals3 = []
    sample_sizes = []
    newdata = pd.DataFrame()
       
    for category in categories: 
        cat_list.append(category)
        for key, item in dict_missing_nb.items(): 
            if key == category: 
                MORE = item
        sample_sizes.append(MORE)
        print(category, ': need', MORE, 'more values')
        impute = datacat[['cat1', 'cielab', 'cielab_L', 'cielab_a', 'cielab_b']][datacat['cat1']==category]
        imputelist = []
        for imp in range(len(impute)): 
            row = impute.iloc[imp]
            cielab = [row[2], row[3], row[4]]
            imputelist.append(cielab)
        
        
        df2 = pd.DataFrame([[category], [category]], columns=['cat1'])
        impute2 = impute.append(df2)
        
        imputelist2 = []
        for imp in range(len(impute2)): 
            row = impute2.iloc[imp]
            cielab = [row[2], row[3], row[4]]
            imputelist2.append(cielab)
            
        for i, value in enumerate(VALUES): 
            try: 
                k2, p = stats.normaltest(impute[value])            
                alpha = 1e-3
                print("p = {:g}".format(p))
                
                if p < alpha:
                    print("Other distribution: The null hypothesis can be rejected")
                    if i == 0: 
                        normals.append('n')
                    elif i ==1: 
                        normals2.append('n')
                    else: 
                        normals3.append('n')
                else:
                    print("Normal distribution: The null hypothesis cannot be rejected")
                    if i == 0: 
                        normals.append('y')
                    elif i ==1: 
                        normals2.append('y')
                    else: 
                        normals3.append('y')       
            except: 
                if i == 0: 
                    normals.append('nan: <8samples')
                elif i ==1: 
                    normals2.append('nan: <8samples')
                else: 
                    normals3.append('nan: <8samples')        
            plt.hist(impute['cielab'])
            plt.show()
           
        try: 
            lab_l = np.random.normal(s.mean(impute[VALUES[0]]), s.stdev(impute[VALUES[0]]), MORE)
            lab_a = np.random.normal(s.mean(impute[VALUES[1]]), s.stdev(impute[VALUES[1]]), MORE)
            lab_b = np.random.normal(s.mean(impute[VALUES[2]]), s.stdev(impute[VALUES[2]]), MORE)
        except: 
            print('Less than 2 data points')
        lab_l = np.clip(lab_l, 0, 100)
        lab_a = np.clip(lab_a, -128, 128) 
        lab_b = np.clip(lab_b, -128, 128)
        
        df2 = make_lab_frame(lab_l, lab_a, lab_b, category)

        show_color(df2)
        
        means.append(s.mean(impute[VALUES[0]])) 
        stdevs.append( s.stdev(impute[VALUES[0]]))
        means2.append(s.mean(impute[VALUES[1]]))
        stdevs2.append( s.stdev(impute[VALUES[1]]))
        means3.append(s.mean(impute[VALUES[2]]))
        stdevs3.append(s.stdev(impute[VALUES[2]]))
            
        impute2 = impute.append(df2)
       
        newdata = newdata.append(df2)
    
    results = fill_dataframe(categories, means, stdevs, means2, stdevs2, means3, stdevs3, normals, normals2, normals3, sample_sizes)
    return results, newdata

def fill_dataframe2(newdata):   
    data2 = pd.DataFrame({
            'id': range(newdata.shape[0]),
            'lang': ['eng'] *newdata.shape[0],
            'name': None,
            'srgb': None,
            'srgb_R': None,
            'srgb_G': None,
            'srgb_B': None,
            'hsv': None,
           'hsv_H': None
           , 'hsv_S': None
           , 'hsv_V': None
           , 'cielab': newdata['cielab'].tolist()
           , 'cielab_L': newdata['cielab_L'].tolist()
           , 'cielab_a': newdata['cielab_a'].tolist()
           , 'cielab_b': newdata['cielab_b'].tolist()
           ,'hsl': None
           , 'hsl_H': None
           , 'hsl_S': None
           , 'hsl_L': None
           , 'LCH': None
           , 'LCH_L': None
           , 'LCH_C': None
           , 'LCH_H': None
           , 'hex': None
            ,'cat1': newdata['cat1'].tolist()
            })
    return data2


def increment_randomize_with_repl(dict_missing_nb2):  
    dict_missing_vals = {}
    for key, i in dict_missing_nb2.items(): 
        if i == 0: 
            pass
        else: 
            new_values = []
            one = key[0]
            two = key[1]
            data['srgb'][(data['cat1']==one) & (data['cat2']==two)] 
            for el in range(i): 
                a= [eval(i) for i in data['srgb'][(data['cat1']==one) & (data['cat2']==two)]]   
                random = choice(a)
                data['name'][data['srgb']==''.join(str(random).split(' '))]
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
                new_values.append(new_value)
            assert i == len(new_values)
            dict_missing_vals[key] = new_values
    return dict_missing_vals
        
def make_frame_lists(dict_missing_vals): 
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
    return langlist, colorlist, rgblist, r_rgblist, g_rgblist, b_rgblist, catlist1, catlist2

def fill_dataframe3(langlist, colorlist, rgblist, r_rgblist, g_rgblist, b_rgblist, catlist1, catlist2): 
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

def merge_dataframe(data, data2, data3): 
    dataset = pd.concat([data, data2, data3]).reset_index(drop=True)
    dataset['lang'] = 'eng' 
    cols = data.columns.tolist()
    dataset = dataset[cols]
    dataset = dataset.sort_index()
    dataset['cat1'].value_counts()
    print('The imbalance is accounted for in the multilabels.')
    dataset.groupby(['cat1','cat2']).size()
    return dataset 

  
#%%
if __name__ == '__main__': 
    
    THRESHOLD = 116 
    
    data = load_data(DICT_PATH, DICT_FILE)
    
    datacats, datacat = split_data(data)
    colorlowten, dict_missing_nb, dict_missing_nb2 = make_missings_or_lowten(datacats, datacat, THRESHOLD)
    
#%%
    # one label case : gaussian upsampling 
    categories = sorted(intersection(list(set(datacat[LABEL])), list(dict_missing_nb.keys())))
    results, newdata = get_gaussian_distr(categories)
    data2 = fill_dataframe2(newdata)

#%%

    # multilabel case: random increment
    dict_missing_vals = increment_randomize_with_repl(dict_missing_nb2)
    langlist, colorlist, rgblist, r_rgblist, g_rgblist, b_rgblist, catlist1, catlist2 = make_frame_lists(dict_missing_vals)
    data3 = fill_dataframe3(langlist, colorlist, rgblist, r_rgblist, g_rgblist, b_rgblist, catlist1, catlist2)
      
#%%

    # merge data and data2 
    dataset = merge_dataframe(data, data2, data3)

#%%

    # save data 
    dataset.to_excel(DICT_OUTPUT_FILE)

