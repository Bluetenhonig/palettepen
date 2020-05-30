# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 19:42:03 2020

@author: lsamsi

=====================
Color Classification into 12 Color Palettes
=====================


For a given dataset of form (color value, color palette name),
predict for a test color value, it's color palette name 
using machine learning classification. 

For testing, part of the dataset's labeled color values can be used. 

Classifier: KNN 
Warning: two neighbors, neighbor k+1 and k, with identical distances 
but different labels, will have results depending on the ordering of the 
training data.
"""

# import modules
import os
import matplotlib.pyplot as plt #2.1.2
import numpy as np
import cv2
import pandas as pd 

# color picker: https://www.ginifab.com/feeds/pms/color_picker_from_image.php


#%%
### MACHINE LEARNING CLASSIFICATION ### 


# display lab_color 
def convert_color(col, conversion=cv2.COLOR_BGR2Lab): #default 
    color = cv2.cvtColor(np.array([[col]], dtype=np.float32), conversion)[0, 0]
    return color

def display_color(color):
    rgb_color = convert_color(color, cv2.COLOR_LAB2RGB)*255
    square = np.full((10, 10, 3), rgb_color, dtype=np.uint8) / 255.0
     #display RGB colors patch 
    plt.figure(figsize = (5,2))
    plt.imshow(square) 
    plt.axis('off')
    plt.show() 


#%%

# get X, y
def get_Xy(data): 
    X = []
    for i in range(len(data['lab'])): 
        x = list(map(float, data['lab'].iloc[i].strip('()').split(', ')))
        X.append(x)
    # encoding y
    classes, y, le = encoding(data)
    assert len(X) == len(y)
    
    return X, y

def encoding(data): 
    # preprocessing
    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    le.fit(data['season'].tolist()) # all classes
    classes = list(le.classes_) # all unique classes
    # encoding 
    le.transform(le.classes_) # all unique encoded classes
    y = list(le.transform(data['season'].tolist())) # all encoded classes 
    return classes, y, le

def get_mappings(classes, data): 
    # build s2S dict: 
    s2S = {}
    for c in classes: 
        s2S[c] = data['Season'][data['season']==c].iloc[0]
    
    Classes = []
    for c in classes: 
        Classes.append(s2S[c])
    
    return s2S, Classes

# train machine learning classifier 
def fit_knn(X, y, n_neighbors): 
    from sklearn.neighbors import KNeighborsClassifier
    neigh = KNeighborsClassifier(n_neighbors=n_neighbors)
    neigh.fit(X, y)
    return neigh 

def knn_classify_color(lab, n_neighbors, data):
    X, y = get_Xy(data)
    neigh = fit_knn(X, y, n_neighbors)
    enc = neigh.predict([lab])
    classes, y, le = encoding(data)
    season = data['Season'][data['season'] == le.inverse_transform(enc)[0]].iloc[0]
    return season 

def knn_classifiedprob_color(lab, n_neighbors, data): 
    X, y = get_Xy(data) 
    neigh = fit_knn(X, y, n_neighbors)
    prob_distr = list(neigh.predict_proba([lab])[0])
    classes, y,le = encoding(data)
    s2S, Classes = get_mappings(classes, data)
    d = {'Season':Classes,'Probability':prob_distr}
    prodist = pd.DataFrame(d)
    prodist = prodist.sort_values(by=['Probability'], ascending=False)
    return prodist 

#%%
    
if __name__ == '__main__':
    #TODO: RGB cs for classification not lab
    
    # set directory 
    os.getcwd()
    os.chdir(r'D:\thesis\code\ColorCode12\pd4cpInlab')
    
    # load dataframes 
    data = pd.read_csv('12seasons.csv')

    
    # SINGLE COLOR 
    ########### Pinterest Colors ###########
    LAB_COLOR = [64, 25, -14]
    # to specify
    N_NEIGHBORS = 5
#    LAB_COLOR = [100, 100, -128] # l, a, b
    
    # MANY COLORS 
    ########### VIAN Colors ###########
    PATH = r'D:\thesis\code\pd28vianhues'
    FILE = 'SRGBLABhsvhslLCHHEX_Eng_VIANHuesColorThesaurus.xlsx'    
    # set directory 
    os.chdir(PATH)    
    # load data 
    df = pd.read_excel(FILE, sep=" ", index_col=0)
    lst = df['VIAN_color_category'].unique()[1:]
    
    # get lab colors
    avg_lab_cols = []
    for i in range(len(lst)):
        avgcolcatrgb = np.mean([np.array(eval(n)) for n in df['cielab'][df['VIAN_color_category'] ==lst[i]]], axis=0).tolist()
        avg_lab_cols.append(avgcolcatrgb)
     # CP: Dark Autumn/Light Spring Palette #
     
    # list of lab colors 
    LAB_COLORS = [[64, 25, -14], [64, 5, -14]]
    
    #%%
    # [OPTIONAL]
#    import tkinter as tk
#    import tkinter.ttk as ttk
#    from tkcolorpicker import askcolor
#    # color picker: choose rgb color   
#    root = tk.Tk()
#    style = ttk.Style(root)
#    style.theme_use('default') # choose from  [ "clam", "alt", "default", "classic"]   
#    askcolor = askcolor((255, 255, 0), root)
#    root.destroy()
#
#    rgb = askcolor[0]
#    r,g,b = rgb
#    rgb = r/255, g/255, b/255
#    LAB_COLOR = list(convert_color(rgb, cv2.COLOR_RGB2LAB))
#    
    #%%
    # return results 
    pred_seasons = []
    for LAB_COLOR in avg_lab_cols: 
        # single color 
        print('---') 
        print('CLASSIFIER: KNN') 
        print(f'# neighbors: {N_NEIGHBORS}') 
        print('---') 
        print('Given Color: ')
        display_color(LAB_COLOR)
        season = knn_classify_color(LAB_COLOR, N_NEIGHBORS, data)
        print('Predicted Season: ', season) 
        pred_seasons.append(season)
        print('---') 
        probability_distribution = knn_classifiedprob_color(LAB_COLOR, N_NEIGHBORS, data)
        print('Top-3 highest probability: \n', probability_distribution.iloc[:3]) 
    
    # many colors 
    print('---') 
    print("Top Seasons for ALL lab colors in Palette:")
    print('---') 
    from collections import Counter
    counter = Counter(pred_seasons).most_common()
    #print(counter)
    topval = counter[0][1]
    winseasons =  []
    for i in counter: 
        if i[1] == topval: 
            winseasons.append(i[0])
    print('Predicted Season for Palette: ', winseasons)




