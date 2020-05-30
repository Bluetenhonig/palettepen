# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 19:05:46 2020

@author: lsamsi
"""
# RGB and HSV Viewer: https://www.rapidtables.com/web/color/RGB_Color.html
# TODO: click on color and have RGB (or other) values in clipboard 

# import modules
import os
import matplotlib.pyplot as plt #2.1.2
import numpy as np
import cv2

# define variables
seasons = ['lsp', 'tsp', 'bs',
          'ls', 'ts', 'ss',
          'sa', 'ta', 'da', 
          'cw', 'tw', 'dw' ]


# set directory 
os.getcwd()
os.chdir(r'D:\thesis\palettes\ss')


CP_INDEX = 'ss1'
twcp = cv2.imread(f'{CP_INDEX}.jpg')
print(twcp.shape)
#(1344, 1856, 3)
# plt.imshow(nemo)
# plt.show()

# it looks like the blue and red channels have been mixed up. 
# In fact, OpenCV by default reads images in BGR format.

twcp = cv2.cvtColor(twcp, cv2.COLOR_BGR2RGB)
plt.imshow(twcp)
plt.show()

#%%

# crop image 
if CP_INDEX == 'tw1': 
    crop_img = twcp[110:,]
elif CP_INDEX == 'tw2':
    crop_img = twcp[290:690,]
elif CP_INDEX == 'tw3' :
    crop_img = twcp[140:340,]    
elif CP_INDEX == 'ta1':
    crop_img = twcp[260:640,]  
elif CP_INDEX == 'da1' or CP_INDEX == 'ts1' or CP_INDEX == 'bs1' or CP_INDEX == 'ls1' or CP_INDEX == 'ss1':
    crop_img = twcp[210:540,]  
elif CP_INDEX == 'dw1' or CP_INDEX == 'cw1' or CP_INDEX == 'lsp1':
    crop_img = twcp[250:630,] 
elif CP_INDEX == 'sa1' or CP_INDEX == 'tsp1':
    crop_img = twcp[75:230,] 

    
plt.imshow(crop_img)
plt.show()
img = crop_img
# #pop-up window 
# cv2.imshow("cropped", crop_img)
# cv2.waitKey(0)


#%%

# build grid 
rows = 7
columns = 10

# hint: if the image is cropped to a size of another season, set grid size to the same season in the conditional
if CP_INDEX == 'tw1': 
    bordery = 15 # border around the color dataset matrix
    borderx = 35
    patchy = 150 # color patch size of each color in the color dataset
    patchx = 160
    delta = 20 # spacing in-between 
elif CP_INDEX == 'tw2':
    bordery = 10 # border around the color dataset matrix
    borderx = 39
    patchy = 49 # color patch size of each color in the color dataset
    patchx = 50
    delta = 5 # spacing in-between 
elif CP_INDEX == 'tw3':
    bordery = 9 # border around the color dataset matrix
    borderx = 20
    patchy = 24 # color patch size of each color in the color dataset
    patchx = 24
    delta = 2 # spacing in-between     
elif CP_INDEX == 'ta1':
    bordery = 15 # border around the color dataset matrix
    borderx = 36
    patchy = 46 # color patch size of each color in the color dataset
    patchx = 46
    delta = 5 # spacing in-between  
elif CP_INDEX == 'da1' or CP_INDEX == 'ts1' or CP_INDEX == 'bs1' or CP_INDEX == 'ls1' or CP_INDEX == 'ss1':
    bordery = 22 # border around the color dataset matrix
    borderx = 31
    patchy = 38 # color patch size of each color in the color dataset
    patchx = 38
    delta = 4 # spacing in-between  
elif CP_INDEX == 'dw1' or CP_INDEX == 'cw1'or CP_INDEX == 'lsp1' :
    bordery = 27 # border around the color dataset matrix
    borderx = 37
    patchy = 44 # color patch size of each color in the color dataset
    patchx = 44
    delta = 7 # spacing in-between  
elif CP_INDEX == 'sa1' or CP_INDEX == 'tsp1':
    bordery = 17 # border around the color dataset matrix
    borderx = 17
    patchy = 18 # color patch size of each color in the color dataset
    patchx = 16
    delta = 1 # spacing in-between   
    
#%%   
# test values to determine grid params 
img_grid = crop_img[:69,:80]
plt.imshow(img_grid)
plt.show()
# y - 28, 69, 76
# x - 38, 80, 87


#%%
ini = bordery+patchy/2   
nex = patchy/2+delta+patchy/2 

# recursive function to determine center coordinate points for each color patch 
def T(n):
    if n == 1:
        return int(ini)
    else:
        return int(T(n-1)+nex)

yy = [T(i) for i in range(1,rows+1)] # make sure rows is an int scalar, not a lst 
# [90.0, 260.0, 430.0, 600.0, 770.0, 940.0, 1110.0]

ini = borderx+patchx/2   
nex = patchx/2+delta+patchx/2 

xx = [T(i) for i in range(1,columns+1)]
# [115.0, 295.0, 475.0, 655.0, 835.0, 1015.0, 1195.0, 1375.0, 1555.0, 1735.0]


# modify pixel value at grid nodes (=center of color patch to black dots)
# for i in yy: 
#     for j in xx: 
#         img[i,j,:] = [0,0,0]

# plt.imshow(img)
# plt.show()

#%%

# access and save pixel value at grid nodes (=center of color patch)
import pandas as pd 

rows = []
cols = []
rgbs = []

for row, i in enumerate(yy): 
    for column, j in enumerate(xx): 
        color = img[i,j]
        color = tuple(list(color))
        #print('row:', row, 'column:', column, 'rgb:', color)
        rows.append(row)
        cols.append(column)
        rgbs.append(color)

df = pd.DataFrame()
df['row'] = rows
df['column'] = cols
df['rgb'] = rgbs


# round rgb vals: 1-3 precision, 0 scale 
df[['r', 'g', 'b']] = pd.DataFrame(df['rgb'].tolist(), index=df.index)
df['r'] = [int(round(x)) for x in df['r']]
df['g'] = [int(round(x*100)) for x in df['g']]
df['b'] = [int(round(x*100)) for x in df['b']]
df['rgb'] = tuple(zip(df['r'],df['g'],df['b'] ))
# df[['r', 'g', 'b']].head()


df.head()
#%%
# tweak dataframe: drop incorrect values 
df = df.iloc[:65]
#df = df.drop(49)
df = df.drop(59)
df = df.drop(58)

#%%
# save pd Dataframe 
os.chdir(r'D:\thesis\code\pd4cpInrgb')
df.to_csv(f'{CP_INDEX}.csv', index=False)
cv2.imwrite(f'{CP_INDEX}.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

