# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 18:37:10 2020

@author: lsamsi
"""



# import modules
import os
import matplotlib.pyplot as plt 
import numpy as np
import cv2
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE

import os
import cv2 
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np 
import sys
sys.path.append(r'C:\Users\Anonym\Desktop\thesis_backup\code')
from ColorConversion00000 import convert_color, floatify
from ColorConversion00000 import *

#%%

#one image only

# path 
#PATH = r'C:\Users\Anonym\Desktop\thesis_backup\film_colors_project\sample-dataset\screenshots\7\7\45609.jpg'
#PATH = r'C:\Users\Anonym\Desktop\thesis_backup\film_colors_project\sample-dataset\screenshots\7\7\45533.jpg'

#%%

# silently 

#multiple images 
FOLDER_PATH = r'D:\thesis\film_colors_project\sample-dataset\screenshots\7\images'
files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(FOLDER_PATH):
    for file in f:
        if '.jpg' in file:
            files.append(os.path.join(r, file))


print('First 10 files: ', files[:10])
    
#%%

def load_image(path): 
    # get image
    img = cv2.imread(path)
    return img 

def show_image():
    # show image 
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #print(img2.shape) 
    plt.imshow(img2)
    plt.axis('off')
    plt.show()

def convert_img_to_LAB(img):
    # convert to LAB 
    img= floatify(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    return img 
    

def calc_hist(img, histSize1, histSize2): 
    combinations = histSize1 * histSize2**2
#    print(combinations) # should be as low as possible 
    
    hist_lab = cv2.calcHist([img], # image
                          [0, 1, 2], #channels
                          None, # no mask
                          [histSize1, histSize2, histSize2], # size of histogram
                          [0, 100, -128, 127, -128, 127]) # channel values 
    return hist_lab, combinations

# there are many bins that are empty because they are not in the visible light
# because no image can fill up the corner of the lab-values in the lab-space 
# -128, -128 is not visible light anymore 

def make_channelfoci(ranged, divisor):
    num_centers = divisor #- 1
    init = len(range(ranged[0], ranged[1]))/divisor/2 
    dist = 2* init
    foci_list = []
    for i in range(num_centers): 
        el = ranged[0] + init + i * dist 
        el = np.round(el, 2)
        foci_list.append(el)
    return foci_list 
    
    


# from 3D to 1D
def get_lab_freq(lfoci, afoci, bfoci): 
    lab = []
    frequency = []
    for l in range(len(l_foci)):
        for a in range(len(a_foci)): 
            for b in range(len(b_foci)): #needs to be int        
                cielab = (l_foci[l],a_foci[a],b_foci[b])
                freq = hist_lab[l][a][b]
#                print(l,a,b, ':', freq)
                lab.append(cielab)
                frequency.append(freq)
    return lab, frequency 



def print_hist_colors(lab, frequency): 
    labfreq = zip(lab, frequency)
    labfreqslim = [el for el in list(labfreq) if el[-1] != 0]
    for el in labfreqslim: 
        rgb_el = convert_color(el[0], "LAB", "RGB", lab2rgb)
        square = np.full((10, 10, 3), rgb_el, dtype=np.uint8) / 255.0
         #display RGB colors patch 
        plt.figure(figsize = (5,2))
        plt.imshow(square) 
        plt.axis('off')
        plt.show()    
#        print(el[1])


def get_rgb_hcl(lab):  
    lch_hue = [] 
    lch = []
    hcl = []
    rgb = []    
    for el in lab: 
        lch_el = convert_color((el), "LAB", "LCH")
        hcl.append(lch_el[::-1])
        lch.append(lch_el)
        lch_hue.append((lch_el[2]))
        rgb_el = convert_color((el), "LAB", "RGB", lab2rgb)
        r, g, b = rgb_el
        rgb_el = r/255, g/255, b/255
        rgb.append(rgb_el)
    return rgb, hcl 


def zip_and_sort(lab, hcl, rgb, frequency): 
    # zip function of 3 lists: frequency, rgb, lch 
    myzip = zip(lab, hcl, rgb, frequency)
    #myzip = zip(lab, lch_hue, rgb, frequency)
    # to sort by lch
    my_sorted_zip= sorted(list(myzip), key=lambda x:x[1])
#    max_height = int(max([el[-1] for el in my_sorted_zip]))
    sum_height = int(sum([el[-1] for el in my_sorted_zip]))
    return my_sorted_zip, sum_height 

def remove_zero_from_zip(my_sorted_zip): 
    zeroremoved = [el for el in my_sorted_zip if el[-1] != 0]
    #zeroremoved = my_sorted_zip
    return zeroremoved


##plot colors as patch
#for i, rgb in enumerate([el[2] for el in zeroremoved]):
#    print(i, rgb)
#    r, g, b = rgb
#    rgb = r*255, g*255, b*255 
#    square = np.full((10, 10, 3), rgb, dtype=np.uint8) / 255.0
#     #display RGB colors patch 
#    plt.figure(figsize = (5,2))
#    plt.imshow(square) 
#    plt.axis('off')
#    plt.show()    
#    print(rgb)


def plot_and_save_figure(zeroremoved, sum_height, path, save=True): 
    fig = plt.figure(dpi=100)
    #plot histogram    
    for el in range(len(zeroremoved)): 
        width_id = el 
        height = np.round(zeroremoved[el][-1] / sum_height,3) 
        color = list(zeroremoved[el][2])
        color.append(1)
#        print(el, height, color)       
        plt.bar(width_id, height, color=color)
    plt.ylabel('Frequency (in %)')
    plt.xlabel('LCH')
    if save: 
        os.chdir(SAVE_PATH)
        plt.savefig(f'{f[-9:]}')
    plt.show()



#%%
 
HISTSIZE1 = 5
HISTSIZE2 = 8
SAVE_PATH = r'D:\thesis\film_colors_project\sample-dataset\screenshots\7\image_histogram_lab_img'

for f in files: 
    print(f[-9:])
    img_loaded = load_image(f)
    img = convert_img_to_LAB(img_loaded)
    hist_lab, combinations = calc_hist(img, HISTSIZE1, HISTSIZE2)
    # l, a, b - facecolor foci 
    l_foci = make_channelfoci((0, 100), HISTSIZE1)
    a_foci = make_channelfoci((-128, 127), HISTSIZE2)
    b_foci = make_channelfoci((-128, 127), HISTSIZE2)
    lab, frequency = get_lab_freq(l_foci, a_foci, b_foci)
#    print(max(frequency))
#    print_hist_colors(lab, frequency)
    rgb, hcl = get_rgb_hcl(lab)
    my_sorted_zip, sum_height = zip_and_sort(lab, hcl, rgb, frequency)
    zeroremoved = remove_zero_from_zip(my_sorted_zip)
    plot_and_save_figure(zeroremoved, sum_height, SAVE_PATH)