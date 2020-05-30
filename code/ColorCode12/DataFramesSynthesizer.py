# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 18:44:38 2020

@author: Anonym
"""

#import modules
import os
import pandas as pd
import numpy as np

# to specify
PATH = r'D:\thesis\code\pd4cpInlab'
# define variables
SEASONS = ['lsp', 'tsp', 'bs',
          'ls', 'ts', 'ss',
          'sa', 'ta', 'da', 
          'cw', 'tw', 'dw' ]

seasons = {
           'lsp': 'Light Spring'
           , 'tsp': 'True Spring'
           , 'bs': 'Bright Spring'
           ,'ls' : 'Light Summer'
           , 'ts' : 'True Summer'
           , 'ss': 'Soft Summer'
           ,'sa' : 'Soft Autumn'
           , 'ta' : 'True Autumn'
           , 'da' : 'Dark Autumn'
           , 'cw' : 'Clear Winter'
           , 'tw' : 'True Winter'
           , 'dw' : 'Dark Winter' 
           }

index = 1

# set directory 
os.chdir(PATH)

# get filenames 
files = []
for season in SEASONS: 
    string = str(season) + str(index)
    files.append(string)

# load dataframes 
df = []
for i, file in enumerate(files): 
    data = pd.read_csv(f'{file}.csv')
    data['season'] = SEASONS[i]
    data['Season'] = seasons[SEASONS[i]]
    df.append(data)

# merge dataframes 
dataframe = pd.concat(df)

# save global dataframe 
dataframe.to_csv('12seasons.csv', index=False)