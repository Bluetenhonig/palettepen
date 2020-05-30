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




#%%
### Load Dataframe ###

# get all images in folder
os.chdir(IMAGE_PATH_CP12)
df = pd.read_csv(SAVE_FILE)


#%%
# analyze dataframe

# derived variables 
shorts = []
cps = []
for i in range(len(df['name'])): 
    short = df['name'].iloc[i][33:46]
    shorts.append(short)
    cp = df['name'].iloc[i][61:65]
    cps.append(cp)

df['shortname'] = shorts
df['cp'] = cps


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



#%% 
df.groupby(['cp'])['label'].count()

df2 = df[['cp']][~df['label'].isin(['n'])]

rank = df2['cp'].value_counts()

# solution here
rank = pd.DataFrame(rank)
rank = rank.reset_index()
rank.columns = ['cp', 'counts'] # change column names

maxval = rank['counts'].max()
wins = rank['cp'][rank['counts']==maxval]
wins = wins.tolist()

sent = f"{df['shortname'].iloc[0]} is a {', '.join(wins)}."
print(sent)


#%%

# set directory
os.chdir(IMAGE_PATH_CP12)

# save result 
with open('result.txt', 'w') as f:
    f.write(sent)