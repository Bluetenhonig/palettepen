# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 08:57:41 2020

@author: Linda Samsinger

uploading image using windows dialog
 
"""


# import modules
import os
from tkinter import *
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import askdirectory 
from skimage.io import imread_collection


#declare variables
IMAGE_PATH = r'D:\thesis\videos'
EXTENSION = '.jpg'

#%%

### SINGLE IMAGE FROM FOLDER ###

root=Tk()
 
path = askopenfilename(filetypes=[('PNG Files','*.png'), ('JPG Files','*.jpg')])
print(path)

root.destroy()


#%%

### ALL IMAGES IN FOLDER ###

# openly 
root= Tk()    
drcty = askdirectory(parent=root,title='Choose directory with image sequence stack files')
print(drcty)
path = str(drcty) + '/*.jpg'
imgs = imread_collection(path) 
root.destroy()


#%%

# silently 
files = []
for r, d, f in os.walk(IMAGE_PATH):
    for file in f:
        if EXTENSION in file:
            files.append(os.path.join(r, file))

for f in files:
    print(f)

#%%
# folder for saving
FOLDER_URL=filedialog.askdirectory(title="Select the folder where you want to save the images: ")
if FOLDER_URL=='':
    print('Folder not selected')
    exit(0)
