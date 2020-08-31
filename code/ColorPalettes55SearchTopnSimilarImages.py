# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 10:40:15 2020

@author: Linda Samsinger

=====================
Color Palettes with Same Color 
=====================

For a given color, find all color palettes with the same color in them. 
Filter color palettes which contain the same color. 

search only in lowest row, thus cannot filter by palette depth, 
threshold ratio and colorbar count because of ratio width information 
only there in whole palette image filters
"""


# import modules
import os
import random 
import matplotlib.pyplot as plt 
import numpy as np
import cv2
import pandas as pd
import itertools
import sys
sys.path.append(r"D:\thesis\code")
from ColorConversion00 import convert_color
from timeit import default_timer as timer

# USER SPECIFICATION 
#SEARCH_PALETTE = "45737" 
#SEARCH_PALETTE = "45829" 
SEARCH_PALETTE = "45923"
#SEARCH_PALETTE = None
TOPN = 10
CS = "lab"


# declare variables 
PALETTE_PATH = r'D:\thesis\film_colors_project\sample-dataset\screenshots\7\D0_100_lab_palette_img'
PALETTE_EXTENSION = '_lab_palette.jpg' 

IMAGE_PATH = r'D:\thesis\film_colors_project\sample-dataset\screenshots\7\images'
IMG_EXTENSION = '.jpg'

performance = {}
kombinatorik = ['cartesian_product', 'combis', 'permuts']
correlation = cv2.HISTCMP_CORREL 
chi_square = cv2.HISTCMP_CHISQR 
intersection = cv2.HISTCMP_INTERSECT 
bhattacharyya = cv2.HISTCMP_BHATTACHARYYA 
# metrics 
metrics = [correlation, chi_square, intersection, bhattacharyya]
metrics_name = ['correlation', 'chi_square', 'intersection', 'bhattacharyya']
metrix = dict(zip(metrics_name, metrics))
METRIC = metrics_name[0]

SAVE_PATH = r'D:\thesis\film_colors_project\sample-dataset\screenshots\7\task2_sample_search_queries_topn_similar'
SAVE_PATH2 = r'D:\thesis\film_colors_project\sample-dataset\screenshots\7\similarity_matrix'
MATRIX_PATH = r'D:\thesis\film_colors_project\sample-dataset\screenshots\7\similarity_matrix'
MATRIX_FILE = 'palette_pair_pbonds_pool569_top11_metriccorrelation_cslab.xlsx'

#%%

def full_short_filename(filename, short=True): 
    if short: 
        if len(filename) == 5: 
            pass
        else: 
            filename = filename[:-23]
    if not short: 
        if len(filename) == 28: 
            filename 
        else: 
            filename = filename + '_D0_100_lab_palette.jpg'  
    return filename 
    
def random_search_palette(cp_names): 
    search_palette = random.choice(cp_names)
    return search_palette

def load_all_palettes_in_folder(path, file_extension): 
    """ load all palettes contained in a folder """
    files = []
    for r, d, f in os.walk(path): 
        for file in f:
            if file_extension in file:
                files.append(file) 
    return files

def get_img_palettes_array(img_files): 
    """ get images and palettes as array """
    cp_images = [] 
    cp_names = []
    cp_palettes = [] 
    
    for img_file in img_files: 
        os.chdir(IMAGE_PATH)
        img = cv2.imread(img_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cp_images.append(img)
        
    for palette_file in palette_files: 
        cp_names.append(palette_file)
        
    for img_file in palette_files: 
        os.chdir(PALETTE_PATH)
        img = cv2.imread(img_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cp_palettes.append(img)
        
    return cp_images, cp_names, cp_palettes


def show_img_or_palette_array(plate_as_array):  
    plt.imshow(plate_as_array) 
    plt.show()
    


def show_hist(file, color_space, path): 
    os.chdir(path)
    img = cv2.imread(file) 
    if color_space == "lab": 
        img = cv2.cvtColor(img.astype(np.float32) / 255, cv2.COLOR_BGR2LAB) 
        hist = cv2.calcHist([img],[0],None,[100],[0, 100])       
        plt.plot(hist,color = 'k', label='l: luminance 0-100') 
        hist = cv2.calcHist([img],[1],None,[128],[-128, 127])        
        plt.plot(hist,color = 'g', label='a: green-,red+')
        hist = cv2.calcHist([img],[2],None,[128],[-128, 127])        
        plt.plot(hist,color = 'b', label='b: blue-,yellow+') 
        plt.axvline(x=128/2, label='0 for a,b axis -128,127', c='r')
        plt.legend()
        plt.show()
    if color_space == "rgb": 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        color = ('r','g','b')
        for i,col in enumerate(color):
            hist = cv2.calcHist([img],[i],None,[256],[0,256])
            plt.plot(hist,color = col)
            plt.xlim([0,256])
        plt.show()
        
def get_hist(file, color_space): 
    img = cv2.imread(file) 
    if color_space == "bgr": 
        hist = cv2.calcHist([img],[0, 1, 2],None,[256, 256, 256],[0, 255, 0, 255, 0, 255])
    if color_space == "lab": 
        img = cv2.cvtColor(img.astype(np.float32) / 255 , cv2.COLOR_BGR2LAB) 
        hist = cv2.calcHist([img],[0, 1, 2],None,[100, 2*128, 2*128],[0, 100, -128, 127, -128, 127])
    if color_space == "rgb": 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        hist = cv2.calcHist([img],[0, 1, 2],None,[256, 256, 256],[0, 255, 0, 255, 0, 255]) 

    return hist 

def make_folder(folder): 
    cwd = os.getcwd()
    try: 
        os.mkdir(folder)
        new_path = os.path.join(cwd, folder)
        os.chdir(new_path)
    except: 
        new_path = os.path.join(cwd, folder)
        os.chdir(new_path)
        
def save_hist(file, color_space, cwd=None, folder=None): 
    if color_space == "lab": 
        os.chdir(cwd)
        img = cv2.imread(file)
        img = cv2.cvtColor(img.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)
        hist = cv2.calcHist([img],[0],None,[100],[0, 100])       
        plt.plot(hist,color = 'k', label='l: luminance 0-100')
        hist = cv2.calcHist([img],[1],None,[128],[-128, 127])        
        plt.plot(hist,color = 'g', label='a: green-,red+')
        hist = cv2.calcHist([img],[2],None,[128],[-128, 127])       
        plt.plot(hist,color = 'b', label='b: blue-,yellow+') 
        plt.axvline(x=128/2, label='0 for a,b axis -128,127', c='r')
        plt.legend()
        new_path = os.path.join(cwd, folder)
        os.chdir(new_path)
        plt.savefig(f"{file[:-4]}_histogram_{color_space}.jpg")
        plt.show()
    if color_space == "rgb": 
        os.chdir(cwd)
        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # in rgb 
        color = ('r','g','b')
        for i,col in enumerate(color):
            hist = cv2.calcHist([img],[i],None,[256],[0,256])
            plt.plot(hist,color = col)
            plt.xlim([0,256])
        new_path = os.path.join(cwd, folder)
        os.chdir(new_path)
        plt.savefig(f"{file[:-4]}_histogram_{color_space}.jpg")
        plt.show()
        

def get_histogram_of_images(img_files, image_path, cs): 
    """ get histogram of images """
    cp_hists_images = []  
    os.chdir(image_path)
    for file in img_files:  
        hist = get_hist(file, cs)
        cp_hists_images.append(hist)
    return cp_hists_images

 
def get_histogram_of_palettes(palette_files, palette_path, cs): 
    """ get histogram of palettes """
    cp_hists_palettes = []
    os.chdir(PALETTE_PATH)  
    for file in palette_files:             
        cv2.imread(file)     
        hist = get_hist(file, cs)        
        cp_hists_palettes.append(hist)
    return cp_hists_palettes



def show_histograms(files, cs, path): 
    """ show histogram of color palettes of images """
    for file in files: 
        show_hist(file, cs, path)


def save_histograms(path, files, cs): 
    """ save histogram of color palettes of images to directory """
    folder_name = f'histogram_{cs}'
    make_folder(folder_name)
    for file in files: 
        save_hist(file, cs, PALETTE_PATH, folder_name)

def three_image_comparison(image1, image2, image3, files, cp_hists):  
    image1 = files.index(image1)
    image2 = files.index(image2)
    image3 = files.index(image3)
    
    corr1 = cv2.compareHist(cp_hists[image1], cp_hists[image1], correlation)
    corr2 = cv2.compareHist(cp_hists[image1], cp_hists[image2], correlation)
    corr3 = cv2.compareHist(cp_hists[image1], cp_hists[image3], correlation)
      
    chi1 = cv2.compareHist(cp_hists[image1], cp_hists[image1], chi_square)
    chi2 = cv2.compareHist(cp_hists[image1], cp_hists[image2], chi_square)
    chi3 = cv2.compareHist(cp_hists[image1], cp_hists[image3], chi_square)
      
    intsec1 = cv2.compareHist(cp_hists[image1], cp_hists[image1], intersection)  
    intsec2 = cv2.compareHist(cp_hists[image1], cp_hists[image2], intersection)  
    intsec3 = cv2.compareHist(cp_hists[image1], cp_hists[image3], intersection)  
    
    bhacha1 = cv2.compareHist(cp_hists[image1], cp_hists[image1], bhattacharyya) 
    bhacha2 = cv2.compareHist(cp_hists[image1], cp_hists[image2], bhattacharyya) 
    bhacha3 = cv2.compareHist(cp_hists[image1], cp_hists[image3], bhattacharyya) 

    print(f"Corr1: {corr1}, Corr2: {corr2}, Corr3: {corr3}")
    print(f"Chi1: {chi1}, Chi2: {chi2}, Chi3: {chi3}")
    print(f"Intsec1: {intsec1}, Intsec2: {intsec2}, Intsec3: {intsec3}")
    print(f"Bhacha1: {bhacha1}, Bhacha2: {bhacha2}, Bhacha3: {bhacha3}")
    
  
def all_pairwise(cp_hists, way = 'combis'):     
    if way == 'cartesian_product':
        print(f"Processing {len(cp_hists)*len(cp_hists)}")
        pairit = [p for p in itertools.product(cp_hists, repeat=2)]
        ids_pairs =  [p for p in itertools.product(range(len(cp_hists)), repeat=2)]
    if way == 'combis': 
        print(f"Processing {(len(cp_hists)*len(cp_hists)-len(cp_hists))/2}")
        pairit = list(itertools.combinations(cp_hists, 2))
        ids_pairs =  list(itertools.combinations(range(len(cp_hists)), 2))
    if way == 'permuts': 
        print(f"Processing {len(cp_hists)*len(cp_hists)-len(cp_hists)}")
        pairit = list(itertools.permutations(cp_hists, 2))
        ids_pairs =  list(itertools.permutations(range(len(cp_hists)), 2))
    return ids_pairs, pairit

 
def get_metrics_of_pairs(pairit, metric): 
    """ metrics of pairwise pairs (cartesian product) """
    mtrs = []
    pairs = []
    for i, p in enumerate(pairit): 
        print(f"{i} processed.")
        corr = cv2.compareHist(p[0], p[1], metrix[metric])
        pair = ids_pairs[i]
        mtrs.append(corr)
        pairs.append(pair)
    return mtrs, pairs

def pairwise_in_dataframe(way, cp_hists, mtrs, cp_names): 
    """ reshape: list to nested list """
    if way == 'cartesian_product':
        nested_list = [mtrs[i:i+len(cp_hists)] for i in range(0, len(mtrs), len(cp_hists))]
        assert len(nested_list[0]) == len(nested_list), "Matrix is not symmetrical."
        df = pd.DataFrame(nested_list, cp_names, cp_names)
    if way == 'combis':
        df = pd.DataFrame([[0 for i in range(len(cp_names))] for j in range(len(cp_names))], cp_names, cp_names)
        for i, p in enumerate(pairs): 
            df.iloc[p[0], p[1]] = mtrs[i]
            df.iloc[p[1], p[0]] = mtrs[i]
        for i in range(len(cp_names)): 
            df.iloc[i, i] = 1
    if way == 'permuts': 
        df = pd.DataFrame([[0 for i in range(len(cp_names))] for j in range(len(cp_names))], cp_names, cp_names)
        for i, p in enumerate(pairs): 
            df.iloc[p[0], p[1]] = mtrs[i]
        for i in range(len(cp_names)): 
            df.iloc[i, i] = 1       
    return df 



def get_topn_similar(df, search_palette, topn, short=True):  
    search_palette = full_short_filename(search_palette, short)
    pbonds = df.loc[search_palette].sort_values(ascending=False)[:topn]
    result_palettes = pbonds.index.tolist()

    print("-------------------------")
    print(f"Task: Find most similar color palettes")
    print(f"Searching color palette id: {search_palette}")
    print(f"Total number of result palettes: {len(pbonds)}")
    print(f"Top-{TOPN} result palettes: \n{result_palettes}")
    print("-------------------------")
    return result_palettes, pbonds

def make_save_folder(path, folder):      
    os.chdir(path)
    try: 
        os.mkdir(folder)
        new_path = os.path.join(path, folder)
        os.chdir(new_path)
    except: 
        new_path = os.path.join(path, folder)
        os.chdir(new_path)

def save_plots(search_palette, dicn): 
    plt.imshow(dicn[search_palette])  
    plt.axis('off')
    plt.savefig(f"0_{search_palette}_palette.jpg")

def show_plots(search_palette, dicn): 
    plt.imshow(dicn[search_palette]) 
    plt.axis('off')
    plt.show()
 

def save_topn_similar(path, folder, search_palette, dicn, topn, result_palettes): 
    make_save_folder(path, folder)
    save_plots(search_palette, dicn)
    new_path = os.path.join(path, folder)
    os.chdir(new_path)
    print("-------------------------")
    print(f"Display top {topn} most similar palettes (output):")
    for i, filename in enumerate(result_palettes):
        print(f"{i+1}: {filename}")    
        plt.imshow(dicn[filename]) 
        plt.axis('off')
        if dicn == name2img: 
            plt.savefig(f"{i+1}_{filename[:-4]}_histimage.jpg")
        else: 
            plt.savefig(f"{i+1}_{filename[:-4]}_histpalette.jpg") 
        plt.show()  
        
def show_topn_similar(topn, name2palette, name2img, result_palettes): 
    if not any(result_palettes): 
        print(f"No palettes found.")
    else: 
        print("-------------------------")   
        for search_palette in result_palettes[1:]: 
            print(f"{search_palette}") 
            show_plots(search_palette, name2palette)
            show_plots(search_palette, name2img)

def show_all_topn_similar(topn, name2palette, name2img, all_result_palettes): 
    for result_palettes in all_result_palettes: 
        show_topn_similar(topn, name2palette, name2img, result_palettes)
        
           
def save_pbonds(pbdons, path, filename): 
    os.chdir(path)
    pbdons.to_excel(filename, index=True)    

def save_all_pbonds(all_pbonds, path, filename): 
    os.chdir(path)
    for pbonds in all_pbonds: 
        pbonds.to_excel(filename, index=True) 
        
    
def load_pbonds(path, filename):
    os.chdir(path)
    df = pd.read_excel(filename, index_col=0)    
    return df

def get_all_topn_similar(df, cp_names, topn): 
    all_pbonds = []
    all_result_palettes = []
    for i, filename in enumerate(cp_names): 
        search_palette = full_short_filename(filename, short=True) 
        print(f"{i} processed.")

        pbonds = df.loc[search_palette].sort_values(ascending=False)[:TOPN]
        result_palettes = pbonds.index.tolist()
        
        print("-------------------------")
        print(f"Task: Find most similar color palettes")
        print(f"Searching color palette id: {SEARCH_PALETTE}")
        print(f"Total number of result palettes: {len(pbonds)}")
        print(f"Top-{topn} result palettes: \n{result_palettes}")
        print("-------------------------")
        all_pbonds.append(pbonds)
        all_result_palettes.append(result_palettes)
    return all_pbonds, all_result_palettes
     

def save_all_topn_similar(path, folder, cp_names, topn, dicn, all_result_palettes):
    make_save_folder(path, folder)
    new_path = os.path.join(path, folder)
    os.chdir(new_path)
    for i, search_palette in enumerate(cp_names): 
        print("-------------------------")
        print(f"Display top {topn} most similar palettes (output):")
        for i, filename in enumerate(all_result_palettes[i]):
            print(f"{i+1}: {filename}")     
            plt.imshow(dicn[filename]) 
            plt.axis('off')
            plt.savefig(f"{i+1}_{filename}_palette.jpg")
            plt.show() 

    
#%%
if __name__ == '__main__':
      
    start = timer()
     
    # load images
    img_files = load_all_palettes_in_folder(IMAGE_PATH, IMG_EXTENSION)
    print('Number of images: ', len(sorted(img_files)))  
    print(f"First five images: {img_files[:5]}")
    
    # load palettes
    palette_files = load_all_palettes_in_folder(PALETTE_PATH, PALETTE_EXTENSION)
    print('Number of color palettes: ', len(sorted(palette_files)))
    print(f"First five palettes in palette_files: {palette_files[:5]}")
    palette_files = sorted(palette_files)

    assert len(palette_files) == len(img_files), 'Number of color palettes not equal to number of images.'
    assert TOPN <= len(palette_files), "Not as many files to evaluate."

    cp_images, cp_names, cp_palettes = get_img_palettes_array(img_files)  
    name2palette= dict(zip(cp_names, cp_palettes))
    name2img = dict(zip(cp_names, cp_images))

#%% 
    
    # make matrix 
    cp_hists_palettes = get_histogram_of_palettes(palette_files, PALETTE_PATH, CS)
    cp_hists_images = get_histogram_of_images(img_files[:100], IMAGE_PATH, CS)
    print(f"Total of {len(cp_hists_images)} histograms processed.")
  
    show_histograms(palette_files[:1], CS, PALETTE_PATH)
    show_histograms(img_files[:1], CS, IMAGE_PATH)

    ids_pairs, pairit = all_pairwise(cp_hists_palettes, kombinatorik[1])
    mtrs, pairs = get_metrics_of_pairs(pairit, METRIC)
    print(mtrs[:5], pairs[:5], len(mtrs), len(pairs))
 
    ids_pairs, pairit = all_pairwise(cp_hists_images, kombinatorik[1])
    mtrs, pairs = get_metrics_of_pairs(pairit, METRIC)
    print(mtrs[:5], pairs[:5], len(mtrs), len(pairs))
    
    
    df_palettes = pairwise_in_dataframe(kombinatorik[1], cp_hists_palettes, mtrs, cp_names)
    df_images = pairwise_in_dataframe(kombinatorik[1], cp_hists_images, mtrs, cp_names)
    
    # save matrix 
    SAVE_PALETTE_FILE = f"palette_pair_pbonds_pool{len(palette_files)}_top{TOPN}_metric{METRIC}_cs{CS}.xlsx"
    SAVE_IMAGE_FILE = f"image_pair_pbonds_pool{len(img_files)}_top{TOPN}_metric{METRIC}_cs{CS}.xlsx"
    
    save_pbonds(df_palettes, MATRIX_PATH, SAVE_PALETTE_FILE)
    save_pbonds(df_images, MATRIX_PATH, SAVE_IMAGE_FILE)
    
    end = timer()
    duration = round(end - start, 2)
    print(f"{kombinatorik[1]} took this amount of time:")
    print(f"{duration} sec") 
    performance[kombinatorik[1]] = duration 

#%% 
    df_palettes = load_pbonds(MATRIX_PATH, MATRIX_FILE)
#    df_images = load_pbonds(MATRIX_PATH, MATRIX_FILE)

#%%
# search query

    # single file
    
    # input 
    if SEARCH_PALETTE == None: 
        SEARCH_PALETTE = random_search_palette(cp_names)
    else: 
        SEARCH_PALETTE = full_short_filename(SEARCH_PALETTE, short=False)
    show_plots(SEARCH_PALETTE, name2img)
    show_plots(SEARCH_PALETTE, name2palette)
    
    # output
    result_palettes, pbonds = get_topn_similar(df_palettes, SEARCH_PALETTE, TOPN, short=False)
    show_topn_similar(TOPN, name2palette, name2img, result_palettes)

    result_images, pbonds = get_topn_similar(df_images, SEARCH_PALETTE, TOPN, short=False)
    show_topn_similar(TOPN, name2palette, name2img, result_images)

    # save output 
    SAVE_FOLDER = f"search_query_histpalette_{SEARCH_PALETTE[:-4]}_pool{len(palette_files)}_top{TOPN}"    
    save_topn_similar(SAVE_PATH, SAVE_FOLDER, SEARCH_PALETTE, name2img, TOPN, result_palettes)
    save_topn_similar(SAVE_PATH, SAVE_FOLDER, SEARCH_PALETTE, name2palette, TOPN, result_palettes)
   
    SAVE_FOLDER = f"search_query_histimage_{SEARCH_PALETTE[:-4]}_pool{len(img_files)}_top{TOPN}"    
    save_topn_similar(SAVE_PATH, SAVE_FOLDER, SEARCH_PALETTE, name2img, TOPN, result_images)
    save_topn_similar(SAVE_PATH, SAVE_FOLDER, SEARCH_PALETTE, name2palette, TOPN, result_images)
      
#
##%%
#    # all files
#    all_pbonds, all_result_palettes = get_all_topn_similar(df_palettes, cp_names, TOPN)
#    


  
