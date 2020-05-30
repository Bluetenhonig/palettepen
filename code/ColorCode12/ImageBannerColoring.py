# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 17:01:17 2020

@author: Linda Samsinger 
"""

# load modules 
import matplotlib.pyplot as plt
import os 
import pandas as pd 
import numpy as np
import matplotlib.patches as patches

# to specify 
PERSON_NAME = 'vanessa_hudgens'
EXTENSION = '.jpg'

# set directory 
os.getcwd()
os.chdir(r'D:\thesis\images\stars')

# get an example image
img = plt.imread(f'{PERSON_NAME}{EXTENSION}')
plt.imshow(img)
plt.show()
print(img.shape)


# get hex colors 
name = ['gold'
       , 'silver'
       , 'unclear'
       , 'clear'
       , 'dark'
       , 'light'
       ]

hexs = [ '#FFDF00'
       , '#C0C0C0'
       ,  '#aaa9ad'
       , '#FF0000'
       , '#100c08'
       , '#f5f5f5'
       ]


dfs = {}
dfs['color'] = name
dfs['HEX'] = hexs



#%%
def crop_image(height=None, width=None): 
    # crop image 
    crop_img = img[:height,]
    plt.imshow(crop_img)
    plt.show()
    return crop_img




def plot_image(points, width, height, color, name, axis=False, save_image=False): 
    if save_image: 
        # save the image 
        # set directory 
        os.getcwd()
        os.chdir(r'D:\thesis\code\12cps')
        try: 
            os.chdir(f'D:/thesis/code/12cps/{PERSON_NAME}')
        except: 
            os.mkdir(f'{PERSON_NAME}')
            os.chdir(f'D:/thesis/code/12cps/{PERSON_NAME}')
               
    # Create a figure. Equal aspect so circles look circular
    fig,ax = plt.subplots(1)
    ax.set_aspect('equal')
    
    # Show the image
    ax.imshow(img)
    if not axis: 
        ax.axis('off')
    
    # Now, loop through coord arrays, and create a circle at each x,y pair

    rect = patches.Rectangle((points),width,height,facecolor=color) # Rectangle(xy, width, height)
    # Add the patch to the Axes
    ax.add_patch(rect)
    
    if save_image: 
        # save image
        plt.savefig(f'{PERSON_NAME}_banner_{name}{EXTENSION}')
            
    # Show the image
    plt.show()


#%%
# to specify
X = 0
Y = 240 

CROP_IMAGE = False  # x,y  TO ADJUST with axis= True BEFORE SAVING THE IMAGES
CROP_IMAGE_HEIGHT = 400 

width = img.shape[1]
height = img.shape[0] - Y 
hex_colors = dfs['HEX']

# crop image
if CROP_IMAGE: 
    img = crop_image(height=CROP_IMAGE_HEIGHT)

# display color palette 
for i, color in enumerate(hex_colors):
    plot_image((X,Y), width, height, color, dfs['color'][i], axis=0, save_image=1)

               
