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
from matplotlib.patches import Circle


# to specify 
PERSON_NAME = 'vanessa_hudgens'
EXTENSION = '.jpg'

# set directory 
os.getcwd()
os.chdir(r'D:\thesis\images\stars')

# get an example image
img = plt.imread(f'{PERSON_NAME}{EXTENSION}')
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
# functions for ring of circles

import math

# Two dimensional rotation
# returns coordinates in a tuple (x,y)
def rotate(x, y, r):
    rx = (x*math.cos(r)) - (y*math.sin(r))
    ry = (y*math.cos(r)) + (x*math.sin(r))
    return (rx, ry)

# create a ring of points centered on center (x,y) with a given radius
# using the specified number of points
# center should be a tuple or list of coordinates (x,y)
# returns a list of point coordinates in tuples
# ie. [(x1,y1),(x2,y2
def point_ring(center, num_points, radius):
    arc = (2 * math.pi) / num_points # what is the angle between two of the points
    points = []
    for p in range(num_points):
        (px,py) = rotate(0, radius, arc * p) 
        px += center[0]
        py += center[1]
        points.append((px,py))
    return points


#%%

def get_circle_colors(hex_df, start, finish, sampling=False):
    assert finish - start == CIRCLES_COUNT, "Difference start and finish must be equal to CIRCLES_COUNT."
    if sampling: 
    # sampling colors 
        circle_color = HEX_COLORS.sample(n=CIRCLES_COUNT) # always same sample with random_state=1
        colors = [l for l in circle_color] # pd2list
    
    # sequencing colors
    circle_color = hex_df[start:finish] # always same sample with random_state=1
    colors = [l for l in circle_color] # pd2list
    return colors 

def get_circle_coordinates(circles_center, circles_count, circles_radius):
    # Make coordinates for circles 
    points = []
    for point in point_ring(circles_center, circles_count, circles_radius):
    # loop through the points created from a ring centered at (150,150),
    # with a radius of 10, using 96 points
        (x,y) = point[0], point[1]
        points.append((x,y))
    return points

def plot_image(points, colors, circle_radius, axis=False, save_image=False): 
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
    for i, (xx,yy) in enumerate(points):
        circ = Circle((xx,yy), radius=circle_radius, color=colors[i]) # Circle(center, radius)
        ax.add_patch(circ)   
    
    if save_image: 
        # save image
        plt.savefig(f'{PERSON_NAME}_{CP}_ring{START}{FINISH}{EXTENSION}')
            
    # Show the image
    plt.show()


#%%
# to specify
CIRCLES_COUNT = 15 
CIRCLES_RADIUS = 200 
CIRCLES_CENTER = (240,220) # x,y  TO ADJUST with axis= True BEFORE SAVING THE IMAGES
CIRCLE_RADIUS = 20
#HEX_COLORS = df['HEX']
START = 0
FINISH = 15

assert FINISH - START == CIRCLES_COUNT, "Difference start and finish must be equal to CIRCLES_COUNT."
    

for CP in CPs: 
    HEX_COLORS = dfs[CP]['HEX']
    colors = get_circle_colors(HEX_COLORS, START, FINISH, sampling=False)
    points = get_circle_coordinates(CIRCLES_CENTER, CIRCLES_COUNT, CIRCLES_RADIUS)
    plot_image(points, colors, CIRCLE_RADIUS, axis=1, save_image=0)