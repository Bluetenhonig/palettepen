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
import os 
import pandas as pd
import numpy as np 
import cv2
import matplotlib.pyplot as plt
# imputation
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
#from sklearn.impute import KNNImputer

#%%

### Color-Thesaurus EPFL ###
### UPSAMPLING ###

# goal: upsample basic colors data set using method interval 
PATH = r'D:\thesis\input_color_name_dictionaries\thesaurus\datasets'
FILE = 'effcnd_thesaurus_basicitten.xlsx'
OUTPUT_FILE = 'eeffcnd_thesaurus_basicitten_upinterval.xlsx' 

# set directory 
os.chdir(PATH)

# load data 
data = pd.read_excel(FILE, sep=" ", index_col=0)
#data.info()
data.columns

# show imbalanced classes 
data['cat1'].value_counts()
data['cat2'].value_counts()


#split dataset into one cats and double cats 
datacats = data[data['cat2'].apply(lambda x: isinstance(x, str))]
datacat = data[data['cat2'].apply(lambda x: isinstance(x, float))]

assert datacats.shape[0] + datacat.shape[0] == data.shape[0]



datacat['cat1'].value_counts().shape[0]
datacat['cat1'].value_counts().iloc[0]
datacats.groupby(['cat1','cat2']).size()
max(datacats.groupby(['cat1','cat2']).size())

THRESHOLD = 116 # = datacat['cat1'].value_counts().iloc[0]
colorlowten = []
for key, nb in datacat['cat1'].value_counts().items(): 
    if nb <THRESHOLD: 
        colorlowten.append(key)
        

dict_missing_nb = {}
for k,i in datacat['cat1'].value_counts().items(): 
    if k in colorlowten: 
        dict_missing_nb[k] = THRESHOLD - i

dict_missing_nb2 = {}
for k,i in datacats.groupby(['cat1','cat2']).size().items(): 
    dict_missing_nb2[k] = max(datacats.groupby(['cat1','cat2']).size()) - i
    


#%%
# one label case 
    
# RESAMPLING (UPSAMPLING) 

import pandas as pd
from scipy import stats
import sys
sys.path.append(r'D:\thesis\code')
from ColorVisualization import display_color
# RESAMPLING (UPSAMPLING) 
# Random gauss (not original distr) resample from given data list 

# sample from distribution only possible with at least 2 data points 

LABEL = 'cat1'

def intersection(lst1, lst2): 
    return set(lst1).intersection(lst2) 
# category = categories[4]
categories = sorted(intersection(list(set(datacat[LABEL])), list(dict_missing_nb.keys())))




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
    print(category, MORE)
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
        
    # plt.hist(VALUES[0])
    # plt.hist(VALUES[1])
    # plt.hist(VALUES[2])
    
    # normaltest 
    for i, value in enumerate(VALUES): 
        pts = 1000
        try: 
            k2, p = stats.normaltest(impute[value])            
            alpha = 1e-3
            # print("p = {:g}".format(p))
            
            if p < alpha:  # null hypothesis: x comes from a normal distribution
                #print("Other distribution: The null hypothesis can be rejected")
                if i == 0: 
                    normals.append('n')
                elif i ==1: 
                    normals2.append('n')
                else: 
                    normals3.append('n')
            else:
                #print("Normal distribution: The null hypothesis cannot be rejected")
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
    # plt.hist(impute['cielab'])
    # plt.show()
       
    import statistics as s 
    from statistics import stdev
    try: 
        lab_l = np.random.normal(s.mean(impute[VALUES[0]]), s.stdev(impute[VALUES[0]]), MORE)
        lab_a = np.random.normal(s.mean(impute[VALUES[1]]), s.stdev(impute[VALUES[1]]), MORE)
        lab_b = np.random.normal(s.mean(impute[VALUES[2]]), s.stdev(impute[VALUES[2]]), MORE)
    except: 
        print('Less than 2 data points')
    lab_l = np.clip(lab_l, 0, 100)
    lab_a = np.clip(lab_a, -128, 128) 
    lab_b = np.clip(lab_b, -128, 128)
    
    means.append(s.mean(impute[VALUES[0]])) 
    stdevs.append( s.stdev(impute[VALUES[0]]))
    means2.append(s.mean(impute[VALUES[1]]))
    stdevs2.append( s.stdev(impute[VALUES[1]]))
    means3.append(s.mean(impute[VALUES[2]]))
    stdevs3.append(s.stdev(impute[VALUES[2]]))

    
    
    df2 = pd.DataFrame({'cielab_L': list(lab_l)
                        , 'cielab_a': list(lab_a)
                        , 'cielab_b': list(lab_b)
                        , 'cielab': [list(i) for i in list(zip(lab_l, lab_a, lab_b))]
                        , 'cat1' : np.repeat(category, MORE)})
    
    df2 = df2.sort_index()
    impute2 = impute.append(df2)
    # frame with all new values for categories 
    newdata = newdata.append(df2)
    
    # plt.hist(impute2['cielab_L'])
    # plt.hist(impute2['cielab_a'])
    # plt.hist(impute2['cielab_b'])
    
    # # display colors of newly sampled values 
    # for i in range(len(df2['cielab'])): 
    #     display_color(df2['cielab'].iloc[i], origin='LAB')
    
    
    # # histogram probability density 
    # plt.hist(impute2['cielab_L'], density=True)
    # plt.hist(impute2['cielab_a'], density=True)
    # plt.hist(impute2['cielab_b'], density=True)
    
    # plt.hist(df2['cielab_L'], density=True)
    # plt.hist(df2['cielab_a'], density=True)
    # plt.hist(df2['cielab_b'], density=True)

   
    # # get discrete probability 
    # import scipy.stats
    # prob_cielab_L = []
    # prob_cielab_L_total = []
    # prob_cielab_a = []
    # prob_cielab_b = []
    # for i in df2['cielab_L']: 
    #     prob = scipy.stats.norm(s.mean(impute['cielab_L']), s.stdev(impute['cielab_L'])).pdf(i)
    #     prob_cielab_L.append(prob)
    #     # probc = scipy.stats.norm(s.mean(impute['cielab_L']), s.stdev(impute['cielab_L'])).cdf(i)
    #     # prob_cielab_L_total.append(probc)
    # for i in df2['cielab_a']: 
    #     prob2 = scipy.stats.norm(s.mean(impute['cielab_a']), s.stdev(impute['cielab_a'])).pdf(i)
    #     prob_cielab_a.append(prob2)
    # for i in df2['cielab_b']:
    #     prob3 = scipy.stats.norm(s.mean(impute['cielab_b']), s.stdev(impute['cielab_b'])).pdf(i)
    #     prob_cielab_b.append(prob3)
        
    # probability 
    # prob_cielab = [a*b*c for a,b,c in zip(prob_cielab_L,prob_cielab_a, prob_cielab_b)]    
    # df2['cielab_prob'] = prob_cielab
    
    # df2 = df2.sort_values(by='cielab_prob', ascending=False)
    
    # display colors in histogram sections 
    # tails = df2[df2['cielab_prob']>=.09]
    # tails = tails.sort_values(by='cielab_a', ascending=True)
    # for i in range(len(tails['cielab'])): 
    #     print(i, tails['cielab'].iloc[i], tails['cielab_prob'].iloc[i])
    #     display_color(tails['cielab'].iloc[i], origin='LAB', save=True)
    
    
    # for i in range(len(impute['cielab'])): 
    #     print(i, impute['cielab'].iloc[i])
    #     display_color(eval(impute['cielab'].iloc[i]), origin='LAB', save=True)

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
    
#%%
    
#IMPUTATION 


# SimpleImputer  
# LABEL = 'cat1'
# categories = sorted(list(set(df[LABEL])))
# category = 'apricot'

# impute = df[['cat1', 'cielab', 'cielab_L', 'cielab_a', 'cielab_b']][df['cat1']==category]
# imputelist = []
# for imp in range(len(impute)): 
#     row = impute.iloc[imp]
#     cielab = [row[2], row[3], row[4]]
#     imputelist.append(cielab)

# df2 = pd.DataFrame([[category], [category]], columns=['cat1'])
# impute2 = impute.append(df2)

# imputelist2 = []
# for imp in range(len(impute2)): 
#     row = impute2.iloc[imp]
#     cielab = [row[2], row[3], row[4]]
#     imputelist2.append(cielab)

# imp_mean = SimpleImputer(missing_values=np.nan, strategy='median')
# imp_mean.fit(imputelist)

# X = imputelist2
# print(imp_mean.transform(X))
# print(category)
# # results are true to the category
# # constant

# # IterativeImputer 
# LABEL = 'cat1'
# categories = sorted(list(set(df[LABEL])))
# category = categories[4]

# impute = df[['cat1', 'cielab', 'cielab_L', 'cielab_a', 'cielab_b']][df['cat1']==category]
# imputelist = []
# for imp in range(len(impute)): 
#     row = impute.iloc[imp]
#     cielab = [row[2], row[3], row[4]]
#     imputelist.append(cielab)

# df2 = pd.DataFrame([[category], [category]], columns=['cat1'])
# impute2 = impute.append(df2)

# imputelist2 = []
# for imp in range(len(impute2)): 
#     row = impute2.iloc[imp]
#     cielab = [row[2], row[3], row[4]]
#     imputelist2.append(cielab)

# imp_mean = IterativeImputer(missing_values=np.nan, sample_posterior=True, initial_strategy='median',  min_value=[4], max_value=[100], random_state=0)
# imp_mean.fit(imputelist)

# X = imputelist2
# print(imp_mean.transform(X))
# print(category)
# # results are false to the category
# # Gaussian distr


# # KNN Imputer
# LABEL = 'cat1'
# categories = sorted(list(set(df[LABEL])))
# category = categories[4]

# impute = df[['cat1', 'cielab', 'cielab_L', 'cielab_a', 'cielab_b']][df['cat1']==category]
# imputelist = []
# for imp in range(len(impute)): 
#     row = impute.iloc[imp]
#     cielab = [row[2], row[3], row[4]]
#     imputelist.append(cielab)


# df2 = pd.DataFrame([[category], [category]], columns=['cat1'])
# impute2 = impute.append(df2)

# imputelist2 = []
# for imp in range(len(impute2)): 
#     row = impute2.iloc[imp]
#     cielab = [row[2], row[3], row[4]]
#     imputelist2.append(cielab)
    
    
# imputer = KNNImputer(missing_values=np.nan, n_neighbors=2)
# X = imputelist2
# print(imputer.fit_transform(X))
# print(category)
# # results are true to the category
# # constant
#%%

# restructure
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


#%%

# multilabel case
from random import choice, sample

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
            a= [eval(i) for i in data['srgb'][(data['cat1']==one) & (data['cat2']==two)]]   
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

# restructure
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


 
#%%

# merge data and data2 

dataset = pd.concat([data, data2, data3]).reset_index(drop=True)
dataset['lang'] = 'eng' 
cols = data.columns.tolist()
dataset = dataset[cols]
dataset = dataset.sort_index()
dataset['cat1'].value_counts()
print('The imbalance is accounted for in the multilabels.')
dataset.groupby(['cat1','cat2']).size()


#%%

# save data (if adobe is first color! )
dataset.to_excel(OUTPUT_FILE)

