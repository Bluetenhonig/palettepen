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
import sys
sys.path.append(r'D:\thesis\code\04_Final')
from ColorConversion00 import convert_color, floatify
from ColorConversion00 import *


#%%

HISTSIZE1 = 5
HISTSIZE2 = 8
SAVE_PATH = r'D:\thesis\film_colors_project\sample-dataset\screenshots\7\image_histogram_lab_img'

IMAGE = r'C:\Users\Anonym\Desktop\thesis_backup\film_colors_project\sample-dataset\screenshots\7\images\45609.jpg'
IMAGES_PATH = r'D:\thesis\film_colors_project\sample-dataset\screenshots\7\images'


    
#%%

def load_image(path): 
    img = cv2.imread(path)
    return img 

def get_all_files_in_folder(path): 
    files = []
    for r, d, f in os.walk(path):
        for file in f:
            if '.jpg' in file:
                files.append(os.path.join(r, file))
    files = sorted(files)
    print('First 10 files: ', files[:10])
    return files 

def show_image(img): 
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img2)
    plt.axis('off')
    plt.show()

def convert_img_to_LAB(img):
    img= floatify(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    return img 
    

def calc_hist(img, histSize1, histSize2): 
    combinations = histSize1 * histSize2**2
    hist_lab = cv2.calcHist([img],
                          [0, 1, 2],
                          None,
                          [histSize1, histSize2, histSize2], 
                          [0, 100, -128, 127, -128, 127]) 
    return hist_lab, combinations


def make_channelfoci(ranged, divisor):
    num_centers = divisor
    init = len(range(ranged[0], ranged[1]))/divisor/2 
    dist = 2* init
    foci_list = []
    for i in range(num_centers): 
        el = ranged[0] + init + i * dist 
        el = np.round(el, 2)
        foci_list.append(el)
    return foci_list 
    
 
def get_lab_freq(lfoci, afoci, bfoci, hist_lab): 
    lab = []
    frequency = []
    for l in range(len(lfoci)):
        for a in range(len(afoci)): 
            for b in range(len(bfoci)):       
                cielab = (lfoci[l],afoci[a],bfoci[b])
                freq = hist_lab[l][a][b]
                lab.append(cielab)
                frequency.append(freq)
    return lab, frequency 



def print_hist_colors(lab, frequency): 
    labfreq = zip(lab, frequency)
    labfreqslim = [el for el in list(labfreq) if el[-1] != 0]
    for el in labfreqslim: 
        rgb_el = convert_color(el[0], "LAB", "RGB", lab2rgb)
        square = np.full((10, 10, 3), rgb_el, dtype=np.uint8) / 255.0
        plt.figure(figsize = (5,2))
        plt.imshow(square) 
        plt.axis('off')
        plt.show()    


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
    myzip = zip(lab, hcl, rgb, frequency)
    my_sorted_zip= sorted(list(myzip), key=lambda x:x[1])
    sum_height = int(sum([el[-1] for el in my_sorted_zip]))
    return my_sorted_zip, sum_height 

def remove_zero_from_zip(my_sorted_zip): 
    zeroremoved = [el for el in my_sorted_zip if el[-1] != 0]
    return zeroremoved


def plot_histogram_colors_as_patch(zeroremoved): 
    for i, rgb in enumerate([el[2] for el in zeroremoved]):
        print(i, rgb)
        r, g, b = rgb
        rgb = r*255, g*255, b*255 
        square = np.full((10, 10, 3), rgb, dtype=np.uint8) / 255.0
        plt.figure(figsize = (5,2))
        plt.imshow(square) 
        plt.axis('off')
        plt.show()    


def plot_and_save_figure(zeroremoved, sum_height, path, save=True): 
    fig = plt.figure(dpi=100)    
    for el in range(len(zeroremoved)): 
        width_id = el 
        height = np.round(zeroremoved[el][-1] / sum_height,3) 
        color = list(zeroremoved[el][2])
        color.append(1)      
        plt.bar(width_id, height, color=color)
    plt.ylabel('Frequency (in %)')
    plt.xlabel('LCH')
    if save: 
        os.chdir(SAVE_PATH)
        plt.savefig(f'{f[-9:]}')
    plt.show()

def get_color_histogram(files, show_histogram_colors=False): 
    histograms = []
    for f in files: 
        img_loaded = load_image(f)
        img = convert_img_to_LAB(img_loaded)
        hist_lab, combinations = calc_hist(img, HISTSIZE1, HISTSIZE2)
        l_foci = make_channelfoci((0, 100), HISTSIZE1)
        a_foci = make_channelfoci((-128, 127), HISTSIZE2)
        b_foci = make_channelfoci((-128, 127), HISTSIZE2)
        lab, frequency = get_lab_freq(l_foci, a_foci, b_foci, hist_lab)
        if show_histogram_colors: 
            print_hist_colors(lab, frequency)
        rgb, hcl = get_rgb_hcl(lab)
        my_sorted_zip, sum_height = zip_and_sort(lab, hcl, rgb, frequency)
        zeroremoved = remove_zero_from_zip(my_sorted_zip)
        histogram = (zeroremoved, sum_height)
        histograms.append(histogram)
    return histograms
    
def show_color_histogram(histograms, path, save=False):
    for histogram, sum_height in histograms:     
        plot_and_save_figure(histogram, sum_height, path, save)



#%%

if __name__ == '__main__': 
    

    files = get_all_files_in_folder(IMAGES_PATH)
    histograms = get_color_histogram(files, show_histogram_colors=False)
    show_color_histogram(histograms, SAVE_PATH, save=False)
    plot_histogram_colors_as_patch(histograms[0][0])
