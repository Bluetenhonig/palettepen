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


# set directory 
os.chdir(PATH)

# load data 
data = pd.read_excel(FILE, sep=" ", index_col=0)

data[['name','srgb']]

#%%
### Processing ###

    
def plot_color(color, size): 
    # plot image
    image = np.full((size, size, 3), color, dtype=np.uint8) / 255
    plt.imshow(image) 
    plt.axis('off')
    plt.show()
    return image 


#%% 

### add missing VIAN color categories to data set 
subdata = data[data['VIAN_color_category'].notnull() == False]

# manual classification 
labels = []
for f in range(len(subdata['name'])): 
    # VIAN color categories
    lst = data['VIAN_color_category'].unique()[1:]    
    print("VIAN colors: ", lst)
    # plot test color 
    plot_color(eval(subdata['srgb'].iloc[f]), 10)
    print(subdata['name'].iloc[f])
    label = input("Which VIAN color category should this color have? ")  
    labels.append(label)

subdata['VIAN_color_category'] = labels
subdata[['name','VIAN_color_category' ]]
#data['VIAN_color_category_all'] = labels

# add subset to whole dataset 
data[data['VIAN_color_category'].notnull() == False] = subdata
data[['name','VIAN_color_category']]
data['VIAN_color_category'].isnull().any()

#%%
FILE = 'effcnd_thesaurus_vian.xlsx'

# set directory
os.chdir(PATH)

# save dataframe 
data.to_excel(FILE, index=False)