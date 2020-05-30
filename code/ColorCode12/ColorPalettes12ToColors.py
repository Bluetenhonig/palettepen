# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 19:42:03 2020

@author: lsamsi

=====================
Colors for 12 Color Palettes
=====================


For a given dataset of form (color value, color palette name),
predict for a test color value, it's color palette name 
using machine learning classification. 

For testing, part of the dataset's labeled color values can be used. 

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

def get_testcolors(steps): 
    import itertools
    l = np.linspace(0,100,steps)
    a = np.linspace(-128,128,steps)
    b = np.linspace(-128,128,steps)
    
    a = [l,a,b]

    lab_colors = list(itertools.product(*a))
    print('Number of colors:' , len(lab_colors))
    print('First test point: ', lab_colors[0])
    print('Last test point: ', lab_colors[-1])
    return lab_colors

#%%
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

def get_mappings(classes): 
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

def knn_classify_color(lab, n_neighbors):
    X, y = get_Xy(data)
    neigh = fit_knn(X, y, n_neighbors)
    enc = neigh.predict([lab])
    classes, y, le = encoding(data)
    season = data['Season'][data['season'] == le.inverse_transform(enc)[0]].iloc[0]
    return season 

def knn_classifiedprob_color(lab, n_neighbors): 
    X, y = get_Xy(data) 
    neigh = fit_knn(X, y, n_neighbors)
    prob_distr = list(neigh.predict_proba([lab])[0])
    classes, y,le = encoding(data)
    s2S, Classes = get_mappings(classes)
    d = {'Season':Classes,'Probability':prob_distr}
    prodist = pd.DataFrame(d)
    prodist = prodist.sort_values(by=['Probability'], ascending=False)
    return prodist 

#%%
    
if __name__ == '__main__':
    # set directory 
    os.getcwd()
    os.chdir(r'D:\thesis\code\pd4cpInlab')
    
    # load dataframes 
    data = pd.read_csv('12seasons.csv')

    # to specify
    # how many color patches (=STEPS*STEPS*STEPS meshgrid)
    STEPS = 51
    N_NEIGHBORS = 5    
    # given a season for colors patches
    SEASON = 'True Winter' 
    
    #%%
    # generate test colors
    lab_colors = get_testcolors(STEPS)
   
    #%%
    # return results 
    lab_colors_seasons = []
    for lab in lab_colors: 
        season = knn_classify_color(lab, N_NEIGHBORS)
        lab_colors_seasons.append(season)
        
    #%%
    # create DataFrame     
    seac = {'Season':lab_colors_seasons, 'lab': lab_colors}    
    df = pd.DataFrame(seac)

    #%%
    # plot season's colors 
    seascols = df['lab'][df['season'] ==SEASON]
    for lab_col in seascols: 
        display_color(lab_col)
    
    #%%
    # save DataFrame   
    os.chdir(r"D:\thesis\code\pd12seasons")
    df.to_csv("lab_cp1_knn5_steps51.csv",index=False)


# TODO: if predict_proba top 2 is equal to top 1, return both seasons for a given color

