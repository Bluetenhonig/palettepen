# -*- coding: utf-8 -*-
"""
Created on Sat May  2 09:26:44 2020

@author: Linda Samsinger

Download Images form Google Image Search using an API 

"""



# to specify 
IMAGE = 'ultramarine'
FOLDER_PATH = r'D:\thesis\images\google\ultramarine'

#%%
### ALL IMAGES IN FOLDER ###

# openly 

## load modules
#from tkinter import *
#from tkinter.filedialog import askdirectory 
#from skimage.io import imread_collection
#
## pop up interface 
#root= Tk()    
#drcty = askdirectory(parent=root,title='Choose directory with image sequence stack files')
#print(drcty)
#path = str(drcty) + '/*.jpg'
#imgs = imread_collection(path) 
#root.destroy()
#
#print(f"Loaded {len(imgs)} images...")

#%%

# silently 

import os

files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(FOLDER_PATH):
    for file in f:
        if '.jpg' in file:
            files.append(os.path.join(r, file))

for f in files:
    print(f)
    

#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)


c = mcolors.ColorConverter().to_rgb
br = make_colormap(
    [c('black'), c('red')])
bg = make_colormap(
    [c('black'), c('green')])
bb = make_colormap(
    [c('black'), c('blue')])
gr = make_colormap(
    [c('green'), c('red')])
by = make_colormap(
    [c('blue'), c('yellow')])
N = 1000
array_dg = np.random.uniform(0, 10, size=(N, 2))
colors = np.random.uniform(-2, 2, size=(N,))
plt.scatter(array_dg[:, 0], array_dg[:, 1], c=colors, cmap=gr)
plt.colorbar()
plt.show()

#%%
# based on https://fairyonice.github.io/Color-space-defenitions-in-python-RGB-and-LAB.html

from keras.preprocessing.image import  img_to_array, load_img

target_size = (256,256)
# Get images
Ximg = []
for filename in files:
    Ximg.append(load_img(filename,target_size=target_size))
print(Ximg[0])

#%%
import numpy as np
Xsub_rgb = []
for img in Ximg:    
    Xsub_rgb.append(img_to_array(img))   
    
print(Xsub_rgb[0].shape)
print(Xsub_rgb[0])

## convert the entire list to numpy array
Xsub_rgb = np.array(Xsub_rgb)

#%%
Nsample = Xsub_rgb.shape[0]

def plotMinMax(Xsub_rgb,labels=["R","G","B"]):
    print("______________________________")
    for i, lab in enumerate(labels):
        mi = np.min(Xsub_rgb[:,:,:,i])
        ma = np.max(Xsub_rgb[:,:,:,i])
        print("{} : MIN={:8.4f}, MAX={:8.4f}".format(lab,mi,ma))
        
plotMinMax(Xsub_rgb,labels=["R","G","B"])   

#%%
#################
### Channeling ###
#################

# decompose into images of R,G,B channels 

from copy import copy
import matplotlib.pyplot as plt

count = 1
fig = plt.figure(figsize=(12,3*Nsample))
for rgb in Xsub_rgb:
    ## This section plot the original rgb
    ax = fig.add_subplot(Nsample,4,count)
    ax.imshow(rgb/255.0); ax.axis("off")
    ax.set_title("original RGB")
    count += 1
    
    for i, lab in enumerate(["R","G","B"]):
        crgb = np.zeros(rgb.shape)
        crgb[:,:,i] = rgb[:,:,0]
        ax = fig.add_subplot(Nsample,4,count)
        ax.imshow(crgb/255.0); ax.axis("off")
        ax.set_title(lab)
        count += 1
    
plt.show()

#%%
Xsub_rgb01 = Xsub_rgb/255.0
from skimage.color import rgb2lab, lab2rgb
Xsub_lab = rgb2lab(Xsub_rgb01)
plotMinMax(Xsub_lab,labels=["L","A","B"]) 

# lab2rgb has to have a dimension (-,-,3) 
Xsub_lab_rgb = np.zeros( Xsub_lab.shape)
for i in range(Xsub_lab.shape[0]):
    Xsub_lab_rgb[i] = lab2rgb(Xsub_lab[i])
plotMinMax(Xsub_lab_rgb.reshape((1,) + Xsub_lab_rgb.shape),labels=["R","G","B"])  

#%%
count = 1
fig = plt.figure(figsize=(6,3*Nsample))
for  irgb, irgb2 in zip(Xsub_rgb01, Xsub_lab_rgb):
    ax = fig.add_subplot(Nsample,2,count)
    ax.imshow(irgb); ax.axis("off")
    ax.set_title("original RGB")
    count += 1
    
    ax = fig.add_subplot(Nsample,2,count)
    ax.imshow(irgb2); ax.axis("off")
    ax.set_title("RGB -> LAB -> RGB")
    count += 1
    
plt.show()

#%%

# decompose into images of L,a,b channels 

def get1dim_from_LAB2RGB(image,idim):
    '''
    image is a single lab image of shape (None,None,3)
    '''
    z = np.zeros(image.shape)
    if idim != 0 :
        z[:,:,0]=80 ## I need brightness to plot the image along 1st or 2nd axis
    z[:,:,idim] = image[:,:,idim]
    z = lab2rgb(z)
    return(z)

count = 1
fig = plt.figure(figsize=(13,3*Nsample))
for lab in Xsub_lab:   
    ax = fig.add_subplot(Nsample,3,count)
    lab_rgb_gray = get1dim_from_LAB2RGB(lab,0) 
    ax.imshow(lab_rgb_gray); ax.axis("off")
    ax.set_title("L: lightness")
    count += 1
    
    ax = fig.add_subplot(Nsample,3,count)
    lab_rgb_a = get1dim_from_LAB2RGB(lab,1) 
    ax.imshow(lab_rgb_a); ax.axis("off")
    ax.set_title("A: color spectrums green to red")
    count += 1
    
    ax = fig.add_subplot(Nsample,3,count)
    lab_rgb_b = get1dim_from_LAB2RGB(lab,2) 
    ax.imshow(lab_rgb_b); ax.axis("off")
    ax.set_title("B: color spectrums blue to yellow")
    count += 1
plt.show()

#%%
#################
### HISTOGRAM ###
#################


# Image as 1D RGB - Histogram  

def get1dim_from_RGB(image,idim):
    '''
    image is a single rgb image of shape (None,None,3)
    '''
    z = np.zeros(image.shape)
    if idim != 0 :
        z[:,:,0]=80 ## I need brightness to plot the image along 1st or 2nd axis
    z[:,:,idim] = image[:,:,idim]
    return(z)

BINS = 32
count = 1
fig = plt.figure(figsize=(20,2*Nsample))

for rgb in Xsub_rgb:
    cm = br
    ax = fig.add_subplot(Nsample,3,count)
    rgb_gray = get1dim_from_RGB(rgb,0) 
    # calculate histogram
    _, bins, patches = ax.hist(rgb_gray.ravel(), bins=BINS, density=1, color='blue')
    bin_centers = 0.5*(bins[:-1]+bins[1:])
    col = bin_centers - min(bin_centers)
    col /= max(col)   
    for c, p in zip(col, patches):
        plt.setp(p, "facecolor", cm(c))
    ax.set_ylabel('Density')
    ax.set_xlabel('R')
    ax.set_title(f'RGB (R): {BINS} bins')
    count += 1
    
    cm = bg
    ax = fig.add_subplot(Nsample,3,count)
    rgb_a = get1dim_from_RGB(rgb,1) 
    _, bins, patches = ax.hist(rgb_a.ravel(), bins=BINS, density=1, color='blue')
    bin_centers = 0.5*(bins[:-1]+bins[1:])
    col = bin_centers - min(bin_centers)
    col /= max(col)   
    for c, p in zip(col, patches):
        plt.setp(p, "facecolor", cm(c))
    ax.set_ylabel('Density')
    ax.set_xlabel('G')
    ax.set_title(f'RGB (G): {BINS} bins')
    count += 1
    
    cm = bb
    ax = fig.add_subplot(Nsample,3,count)
    rgb_b = get1dim_from_RGB(rgb,2) 
    _, bins, patches = ax.hist(rgb_a.ravel(), bins=BINS, density=1, color='blue')
    bin_centers = 0.5*(bins[:-1]+bins[1:])
    col = bin_centers - min(bin_centers)
    col /= max(col)   
    for c, p in zip(col, patches):
        plt.setp(p, "facecolor", cm(c))
    ax.set_ylabel('Density')
    ax.set_xlabel('B')
    ax.set_title(f'RGB (B): {BINS} bins')
    count += 1

    plt.subplots_adjust(top=2.5)
plt.show()



#%%

# Image as 1D LAB- Histogram  

def get1dim_from_LAB2RGB(image,idim):
    '''
    image is a single lab image of shape (None,None,3)
    '''
    z = np.zeros(image.shape)
    if idim != 0 :
        z[:,:,0]=80 ## I need brightness to plot the image along 1st or 2nd axis
    z[:,:,idim] = image[:,:,idim]
    z = lab2rgb(z)
    return(z)

BINS = 32
count = 1
fig = plt.figure(figsize=(15,2*Nsample))

for lab in Xsub_lab:
    cm = plt.cm.get_cmap("gray")
    ax = fig.add_subplot(Nsample,3,count)
    lab_rgb_gray = get1dim_from_LAB2RGB(lab,0) 
    # calculate histogram
    _, bins, patches = ax.hist(lab_rgb_gray.ravel(), bins=BINS, density=1, color='blue')
    bin_centers = 0.5*(bins[:-1]+bins[1:])
    col = bin_centers - min(bin_centers)
    col /= max(col)   
    for c, p in zip(col, patches):
        plt.setp(p, "facecolor", cm(c))
    ax.set_ylabel('Density')
    ax.set_xlabel('L')
    ax.set_xlim([0, 1])
    ax.set_title(f'Lab (L): {BINS} bins')
    count += 1
    
    cm = gr
    ax = fig.add_subplot(Nsample,3,count)
    lab_rgb_a = get1dim_from_LAB2RGB(lab,1) 
    _, bins, patches = ax.hist(lab_rgb_a.ravel(), bins=BINS, density=1, color='blue')
    bin_centers = 0.5*(bins[:-1]+bins[1:])
    col = bin_centers - min(bin_centers)
    col /= max(col)   
    for c, p in zip(col, patches):
        plt.setp(p, "facecolor", cm(c))
    ax.set_ylabel('Density')
    ax.set_xlabel('a')
    ax.set_xlim([0, 1])
    ax.set_title(f'Lab (a): {BINS} bins')
    count += 1
    
    cm = by
    ax = fig.add_subplot(Nsample,3,count)
    lab_rgb_b = get1dim_from_LAB2RGB(lab,2) 
    _, bins, patches = ax.hist(lab_rgb_a.ravel(), bins=BINS, density=1, color='blue')
    bin_centers = 0.5*(bins[:-1]+bins[1:])
    col = bin_centers - min(bin_centers)
    col /= max(col)   
    for c, p in zip(col, patches):
        plt.setp(p, "facecolor", cm(c))
    ax.set_ylabel('Density')
    ax.set_xlabel('b')
    ax.set_xlim([0, 1])
    ax.set_title(f'Lab (b): {BINS} bins')
    count += 1

    plt.subplots_adjust(top=2.5)
plt.show()
 




