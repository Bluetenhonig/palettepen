# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 08:49:56 2020

@author: Linda Samsinger

After last-word classification, some colors are not categorized yet into 
basic colors. This manual classification system helps getting all colors in a 
color name dictionary classified into a basic color. 

Method: Image-to-Label Classification

The goal is to help make a dataset with images and corresponding labels. 
Where an image is given, the user can key in the label when prompted. 

At the end all images have labels. 

Step Before: Last word classification
Goal: manual classification
Step AFter: visualization of color name dictionary in different color spaces 

"""


# import modules
import os
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

### Color-Thesaurus EPFL ###

PATH = r'D:\thesis\input_color_name_dictionaries\thesaurus\datasets'
FILE = 'ffcnd_thesaurus_lastword.xlsx'
OUTPUT_FILE = 'effcnd_thesaurus_vian.xlsx'


#%%
# functions

    
def plot_color(color, size): 
    # plot image
    image = np.full((size, size, 3), color, dtype=np.uint8) / 255
    plt.imshow(image) 
    plt.axis('off')
    plt.show()
    return image 

def load_data(path, file):  
    os.chdir(path) 
    data = pd.read_excel(file, sep=" ", index_col=0)
    return data

def manual_classification(data): 
    """ manual classification: add missing VIAN color categories to data set """
    subdata = data[data['name2cat1'].notnull() == False]  
    labels = []
    for f in range(len(subdata['name'])): 
        # VIAN color categories
        lst = data['name2cat1'].unique()[1:]    
        print("VIAN colors: ", lst)
        # plot test color 
        plot_color(eval(subdata['srgb'].iloc[f]), 10)
        print(subdata['name'].iloc[f])
        label = input("Which VIAN color category should this color have? ")  
        labels.append(label)
    return labels, subdata

def add_subset_to_whole(data, subdata, labels): 
    subdata['name2cat1'] = labels
    subdata[['name','name2cat1' ]]
    #data['VIAN_color_category_all'] = labels
    
    # add subset to whole dataset 
    data[data['name2cat1'].notnull() == False] = subdata
    data[['name','name2cat1']]
    data['name2cat1'].isnull().any()
    return data

#%% 

if __name__ == '__main__' : 

    data = load_data(PATH, FILE)
    labels, subdata = manual_classification(data)
    data = add_subset_to_whole(data, subdata, labels)

#%%

    # save dataframe
    os.chdir(PATH)
    data.to_excel(OUTPUT_FILE, index=False)