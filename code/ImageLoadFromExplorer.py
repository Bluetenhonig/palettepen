# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 08:57:41 2020

@author: Anonym
"""

# upload image using windows dialog 


### SINGLE IMAGE FROM FOLDER ###

# load modules
from tkinter import *
from tkinter.filedialog import askopenfilename
  
root=Tk()
 
path = askopenfilename(filetypes=[('PNG Files','*.png'), ('JPG Files','*.jpg')])
print(path)

root.destroy()


#%%

### ALL IMAGES IN FOLDER ###

# openly 

# load modules
from tkinter import *
from tkinter.filedialog import askdirectory 
from skimage.io import imread_collection

# pop up interface 
root= Tk()    
drcty = askdirectory(parent=root,title='Choose directory with image sequence stack files')
print(drcty)
path = str(drcty) + '/*.jpg'
imgs = imread_collection(path) 
root.destroy()


#%%

# silently 

import os

IMAGE_PATH = r'D:\thesis\videos'

files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(IMAGE_PATH):
    for file in f:
        if '.jpg' in file:
            files.append(os.path.join(r, file))

for f in files:
    print(f)

#%%
# folder for saving
from tkinter import filedialog
FOLDER_URL=filedialog.askdirectory(title="Select the folder where you want to save the images: ")
if FOLDER_URL=='':
    print('Folder not selected')
    exit(0)
