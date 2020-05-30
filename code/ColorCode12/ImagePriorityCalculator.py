# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 11:08:31 2020

@author: Anonym
"""
#TODO: ties 
#TODO: shuffle combinations for display 
#TODO: do not rank all, but determine only the top-1 fast 
#TODO: for each image (without labeling), yes or no or unsure, fast refining

# load modules 
import os
import cv2
import matplotlib.pyplot as plt 
from tkinter import *
from tkinter.filedialog import askopenfilename 
from tkinter.filedialog import askdirectory 
from skimage.io import imread_collection

### Get answers: specify number of images

print('-----------------------')
print('Welcome to PRICA - the PRIORITY CALCULATOR. \nThe priority calculator helps in the decision-making process by using pairwise comparisons to rank images of your choosing. \nFollow these steps to determine your best choice of images.')
print('-----------------------')


LOAD_BULK = bool(input("Do you want to bulk load all images in a folder? (1/0)"))
if not LOAD_BULK: 
    print('Write down the number of unique images you want to compare.')
    images_len = int(input("How many images would you like to compare?"))
    
    print('-----------------------')     
    print(f"Load {images_len} images you want to compare.")
else:       
    print('-----------------------')     
    print(f"Load images you want to compare.")



def bulkload_images():     
    root= Tk()    
    drcty = askdirectory(parent=root,title='Choose directory with image sequence stack files')
    print(drcty)
    # try: 
    #     path = str(drcty) + '/*.PNG'
    # except: 
    path = str(drcty) + '/*.JPG'
    imgs = imread_collection(path) 
    root.destroy()
    return imgs 


def upload_image():     
    root= Tk()    
    path = askopenfilename(filetypes=[('Image Files','*.png *.jpg')])
    print(path)
    root.destroy()
    return path 
   

def load_image(path): 
    # load image in BGR + convert to RGB 
    image_np = cv2.imread(path) # BGR with numpy.uint8, 0-255 val 
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    return image_np

def show_image(image_np): 
    plt.axis('off')
    plt.imshow(image_np) #now it is in RGB  
    return plt.show()


if not LOAD_BULK: 
    # get paths to all images
    paths = []
    for i in range(images_len): 
        paths.append(upload_image())
else: 
    images = bulkload_images()
    images_len = len(images)
    print(images_len, ' images loaded.')
    print('First image: ')
    show_image(images[0])


#%%
 
import string
# all images to compare
# to specify 
if not LOAD_BULK: 
    enc_str = [] # images with code val (no need to type in full name in answers)
    images_str = [] # images as names 
    images = [] # images as numpy arrays 
    for i, path in enumerate(paths): 
        show_image(load_image(path))
        name_image = input('Name the image: ')
        print('-----------------------')
        images.append(load_image(path))
        images_str.append(name_image)
        enc_str.append(list(string.ascii_uppercase)[i])
        
    
    str2enc = {}
    for i in range(len(images_str)): 
        str2enc[images_str[i]] = enc_str[i]
    
    enc2img = {}
    for i in range(len(images_str)): 
        enc2img[enc_str[i]] = images[i]

else:
    enc_str = []
    for i in range(len(images)):
        enc_str.append(i)
    
    enc2img = {}
    for i in range(len(images)): 
        enc2img[enc_str[i]] = images[i]
    
#%%
### Get answers: compare pairwise images  

import itertools
image_comb = list(itertools.combinations(images, 2))
if not LOAD_BULK:
    images_comb_str = list(itertools.combinations(images_str, 2))
enc_comb_str = list(itertools.combinations(enc_str, 2))

import numpy as np
def img2code(image_np, dic):
    for k, v in dic.items(): 
        if np.array_equal(v, image_np): #compare np arrays
            return k 


def show_image_display(image_np1, image_np2): 
    plt.figure(figsize = (10,4))
    plt.subplot(1,2,1)    
    plt.imshow(image_np1) #now it is in RGB  
    plt.axis('off')   
    plt.title(img2code(image_np1, enc2img))
    plt.subplot(1,2,2)
    plt.imshow(image_np2) #now it is in RGB 
    plt.axis('off')
    plt.title(img2code(image_np2, enc2img))
    return plt.show()


answers = []
for i in image_comb:
    print("")
    show_image_display(i[0], i[1]) 
    answer = input(f"Your preferred choice ({img2code(i[0], enc2img)}/{img2code(i[1], enc2img)}): ")
    if answer.islower(): 
        answer = answer.upper()
    try: 
        if answer in images_str: 
            answer = str2enc[answer]
    except: 
        pass
    answers.append(answer)

#%%

### Processing: assign scores to images 

def score4pair(question, answer):
    el1 = question[0]
    el2 = question[1]
    if not scores: #empty dict
        #print('no vals in scores')
        if answer == el1: 
            scores[el1] = 1
            scores[el2] = 2
        else: 
            scores[el1] = 2
            scores[el2] = 1
    else: # append vals to dict
        #print('scores has values')
        if answer == el1: 
            scores[el1] += 1
            try: 
                scores[el2] += 2
            except: 
                scores[el2] = 2
        else: 
            try: 
                scores[el2] += 1
            except: 
                scores[el2] = 1
            try: 
                scores[el1] += 2
            except: 
                scores[el1] = 2
            
    return scores


scores = dict()      
for i in range(len(image_comb)):
    if i == 0: 
        assert scores == {}
    score4pair(enc_comb_str[i], answers[i])

#print(scores)


#%%

### Post results: rank images 

# get key with minimum score  
winner = min(scores, key=scores.get)

# convert bool to name 
for key, value in str2enc.items():  # for name, age in dictionary.iteritems():  (for Python 2.x)
    if value == winner:
        winner_name = key

print('######################################################################')
print('')
print('-----------------------')
print('THE WINNER: ', winner_name)
print('-----------------------')

# from scores to ranks 
rank = {key: rank for rank, key in enumerate(sorted(scores, key=scores.get, reverse=False), 1)}

# convert bool to name 
rank_name={}
for i, (k,v) in enumerate(str2enc.items()):
    rank_name[k]=rank[v]

# sort dict by value 
rank_name={k: v for k, v in sorted(rank_name.items(), key=lambda item: item[1])}

# show ranking  
for x, y in rank_name.items():     
    #print ('Rank', y, ':', x)
    print (y, '-', x)
    show_image(enc2img[str2enc[x]])
    print('-----------------------')


#%%

### Save results: save images to dataframe
import os
import pandas as pd 
  
save = input("Do you want to save your results? (yes/no)")  
if save == 'yes' or save == 'y': 
    # save ranking 
    topic = input("Choose topic of images: ") 
    df = pd.DataFrame.from_dict(rank_name, orient='index',columns=['rank'])
    df[topic] = df.index
    df.reset_index(level=0, inplace=True)
    df = df.drop(columns=['index'])
    print(df)
    print('WINNER IMAGE: ')
    show_image(enc2img[str2enc[winner_name]])
    path = input('Path to where you want to save it:')
    os.chdir(path)
    # save winner    
    winner_img = enc2img[str2enc[winner_name]]
    cv2.imwrite(f'prica_IMG_{topic}-winner-{images_len}.png', winner_img)
    
    print('-----------------------')
    print('>>> Your results were successfully saved to your computer. ')
    
else: 
    pass 

print('-----------------------')
print('Thank you for using PRICA the priority calculator. \nSee you next time!')
print('-----------------------')