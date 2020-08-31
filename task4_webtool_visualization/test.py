# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 16:38:37 2020

@author: Anonym
"""

# warning: when saving image files all need to have unique filename otherwise same filenames will duplicate with display

import sys
import os

sys.path.append('D:\thesis\webtool_visualization')
os.chdir(r'D:\thesis\webtool_visualization')

from hiercp import make_hiercp

IMAGE_PATH = r"D:\thesis\webtool_visualization\static\img" #save image-extracted CP
 
EXTENSION = '.jpg'
IMAGE_FILE = [] #['45445.jpg', '45446.jpg', '45447.jpg', '45448.jpg']
for r, d, f in os.walk(IMAGE_PATH): # r=root, d=directories, f = files
    for file in f:
        if EXTENSION in file:
            IMAGE_FILE.append(file)  
CP_PATH =  r"D:\thesis\webtool_visualization\static\hiercp" #save image-extracted CP   

make_hiercp(IMAGE_PATH, IMAGE_FILE, CP_PATH, EXTENSION)

#%%

from flatcp import make_flatcp

os.chdir(r'D:\thesis\webtool_visualization')

IMAGE_PATH = r"D:\thesis\webtool_visualization\static\img" #save image-extracted CP
 
EXTENSION = '.jpg'
IMAGE_FILE = [] #['45445.jpg', '45446.jpg', '45447.jpg', '45448.jpg']
for r, d, f in os.walk(IMAGE_PATH): # r=root, d=directories, f = files
    for file in f:
        if EXTENSION in file:
            IMAGE_FILE.append(file) 
CP_PATH =  r"D:\thesis\webtool_visualization\static\flatcp" #save image-extracted CP   

make_flatcp(IMAGE_PATH, IMAGE_FILE, CP_PATH, EXTENSION)