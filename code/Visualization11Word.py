# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 19:42:03 2020

@author: lsamsi

Color Patch Visualization

!!! WARNING: Visuazlizations only happen in RGB !!!

"""

# import modules
import os
import matplotlib.pyplot as plt #2.1.2
import numpy as np
import pandas as pd
import cv2

# color picker: https://www.ginifab.com/feeds/pms/color_picker_from_image.php

# to specify
PATH = r'D:\thesis\input_color_name_dictionaries'
FILE = r'color_naming_clusters.xlsx'
# change directory 
os.chdir(PATH)

#%%
df = pd.read_excel(FILE, sheet_name='Sheet2')

#%%
from matplotlib import colors 
import squarify    # pip install squarify (algorithm for treemap)

#Create our plot and resize it.
fig = plt.gcf()
ax = fig.add_subplot()
fig.set_size_inches(20, 5)

# create a color palette, mapped to these values
cmap = plt.cm.Blues
mini=min(df['count'])
maxi=max(df['count'])
norm = colors.Normalize(vmin=mini, vmax=maxi)
colors = [cmap(norm(value)) for value in df['count']]

# If you have 2 lists
squarify.plot(sizes=df['count'], label=df['theme'], alpha=.7, color=colors,text_kwargs={'fontsize':18})
plt.axis('off')
plt.title('Linguistic Color Names for Naming Colors', fontsize=24, loc="left", pad=10)
plt.show()
 

