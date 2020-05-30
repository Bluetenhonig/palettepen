# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 08:49:56 2020

@author: Linda Samsinger

Image-to-Label Classification

The goal is to help make a dataset with images and corresponding labels. 
Where an image is given, the user can key in the label when prompted. 

At the end all images have labels. 
"""

# import modules
import os
import matplotlib.pyplot as plt #2.1.2
import numpy as np
import pandas as pd 
import cv2

# color picker: https://www.ginifab.com/feeds/pms/color_picker_from_image.php
# https://htmlcolorcodes.com/

# to specify  
# body typing: 
#IMAGE_PATH = r'D:\thesis\images\stars'
#FILENAME = 'mischa_barton.jpg'
#SAVE_FILE = 'ImageLabels.csv'
# file: filename (extract person name), image: image as np array, label: natural/gamine/ etc.  for deep learning

# color analysis: 
IMAGE_PATH_CP12 = r'D:\thesis\code\ColorCode12\12cps\aishwarya_rai' # cp12 on person
SAVE_FILE = 'ImageLabels.csv'
# file: filename (extract person name + cp), image: image as np array, label: yes/no 

# new same image shape dimensions
WIDTH = 400
HEIGHT = 400


#%%
### Load Images ###

# get all images in folder
files = []
filenames = []
# r=root, d=directories, f = files
for r, d, f in os.walk(IMAGE_PATH_CP12):
    for file in f:
        filenames.append(file)
        print(file)
        if '.jpg' in file:
            files.append(os.path.join(r, file))

print(f"You have {len(files)} images loaded.")

#%%
# get single image in folder
file = []
for r, d, f in os.walk(IMAGE_PATH):
    file.append(os.path.join(r, FILENAME))

#%%
### Processing ###

def load_image(filepath): 
    # load image in BGR 
    image = cv2.imread(filepath) # BGR with numpy.uint8, 0-255 val  
    # convert image to RGB 
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image 
    
def plot_image(image): 
    # plot image
    plt.imshow(image) 
    plt.axis('off')
    plt.show()
    return image 

def make_same_shape(img, width, height): 
    #print('Original Dimensions : ',img.shape)
    # set shape dimensions 
    dim = (width, height)
    # resize image
    resized_img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)     
    #print('Resized Dimensions : ',resized_img.shape)
    return resized_img

#%% 

### add images to data
nimgs = []
labels = []
rimgs = []

for i, f in enumerate(files):  
    # load image
    image = load_image(f)
    # resize image
    resized_img = make_same_shape(image, WIDTH, HEIGHT)
    # manual classification questionnaire 
    plot_image(image)
    label = input("Which label should this image have? ")  
    nimgs.append(filenames[i])
    rimgs.append(resized_img)
    labels.append(label)

data = pd.DataFrame() 
data['name'] = nimgs
data['label'] = labels
data['image'] = rimgs
    
#%% 

### add single image to data

# load image
image = load_image(file[0])
# resize image 
resized_img = make_same_shape(image, WIDTH, HEIGHT)
# manual classification questionnaire 
plot_image(image)
label = input("Which label should this image have? ")  

# append new row to data
new_row = pd.DataFrame({'image' : [resized_img], 'label': [label]})
data = data.append(new_row, ignore_index = True)

#%%

# set directory
os.chdir(IMAGE_PATH_CP12)

# save dataframe 
data.to_csv(SAVE_FILE, index=False)