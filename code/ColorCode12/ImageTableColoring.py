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
CPs = ['lsp1'
       , 'tsp1'
       , 'bs1'
       , 'ls1'
       , 'ss1'
       , 'ts1'
       , 'sa1'
       , 'ta1'
       , 'da1'
       ,'tw1'
       ,'cw1'
       ,'dw1']
os.chdir(r'D:\thesis\code\pd4cpInhex')

dfs = {}
for CP in CPs: 
    df = pd.read_csv(f'{CP}.csv')
    dfs[CP] = df



#%%
def crop_image(height=None, width=None): 
    # crop image 
    crop_img = img[:height,]
    plt.imshow(crop_img)
    plt.show()
    return crop_img

def get_table_colors(hex_df, start, finish, sampling=False):
    assert finish - start == PATCH_COUNT, "Difference start and finish must be equal to PATCH_COUNT."
    if sampling: 
    # sampling colors 
        circle_color = HEX_COLORS.sample(n=PATCH_COUNT) # always same sample with random_state=1
        colors = [l for l in circle_color] # pd2list
    
    # sequencing colors
    circle_color = hex_df[start:finish] # always same sample with random_state=1
    colors = [l for l in circle_color] # pd2list
    return colors 

def get_table_coordinates(y_height_patch, patch_count):
    # Make coordinates for color palette
    yy = np.empty(10)
    yy.fill(y_height_patch)
    xx = np.linspace(0, img.shape[1], patch_count)
    width = xx[1]
    height = img.shape[0] - y_height_patch
    
    points = zip(xx,yy)
    return points, width, height  


def plot_image(points, width, height, colors, axis=False, save_image=False): 
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
    for i, (xx, yy) in enumerate(points): 
        rect = patches.Rectangle((xx,yy),width,height,facecolor=colors[i]) # Rectangle(xy, width, height)
        # Add the patch to the Axes
        ax.add_patch(rect)
    
    if save_image: 
        # save image
        plt.savefig(f'{PERSON_NAME}_{CP}_table{START}{FINISH}{EXTENSION}')
            
    # Show the image
    plt.show()


#%%
# to specify
PATCH_COUNT = 10 
PATCH_YLINE = 240
HEX_COLORS = df['HEX']

CROP_IMAGE = False  # x,y  TO ADJUST with axis= True BEFORE SAVING THE IMAGES
CROP_IMAGE_HEIGHT = 400 

START = 50
FINISH = 60

assert FINISH - START == PATCH_COUNT, "Difference start and finish must be equal to PATCH_COUNT."


# crop image
if CROP_IMAGE: 
    img = crop_image(height=CROP_IMAGE_HEIGHT)

# display color palette 
for CP in CPs:
    HEX_COLORS = dfs[CP]['HEX']
    colors = get_table_colors(HEX_COLORS, START, FINISH, sampling=False)
    points, width, height = get_table_coordinates(PATCH_YLINE, PATCH_COUNT)
    plot_image(points, width, height, colors, axis=0, save_image=1)

               
