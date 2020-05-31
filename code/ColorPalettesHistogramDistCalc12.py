# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 10:40:15 2020

@author: Linda Samsinger

=====================
Color Palettes with Same Color 
=====================

For a given color, find all color palettes with the same color in them. 
Filter color palettes which contain the same color. 
"""


########### ColorPalette Search ###########

# import modules
import os
import matplotlib.pyplot as plt 
import numpy as np
import cv2
import pandas as pd

       

#%%
# load color palettes 
# images
PALETTE_PATH = r'D:\thesis\film_colors_project\sample-dataset\screenshots\7'
EXTENSION = '_lab_palette.jpg' #  load lab_palette = a rgb image where calc was done in lab
#IMAGE_FILE = 'frame125.jpg'
#FILES = ['frame250.jpg', 'frame375.jpg']     # for a list of images to process 
# load files from directory 
#FILES = ['frame12625_bgr_palette.csv', 'frame125_bgr_palette.csv']
FILES = []
for r, d, f in os.walk(PALETTE_PATH): # r=root, d=directories, f = files
    for file in f:
        if EXTENSION in file:
            FILES.append(file) 

print(f"Number of files: {len(FILES)}")
print(f"First five files in FILES: {FILES[:5]}")

# subset (out of memory error)
FILES = FILES[:50]
performance = {}

# set directory 
os.chdir(PALETTE_PATH)

#%% get images and palettes as array
cp_images = [] # in rgb
cp_names = []
cp_palettes = [] # in rgb


# get images
for FILE in FILES: 
    print(f"Image filename: {FILE[:-21]+'.jpg'}")
    img = cv2.imread(FILE[:-21]+'.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cp_images.append(img)
    
# get image file names
for FILE in FILES: 
    cp_names.append(FILE[:-21])
    
# get color palettes of images  
for FILE in FILES: 
    img = cv2.imread(FILE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cp_palettes.append(img)



# show image 
plt.imshow(cp_images[0]) #now it is in RGB 
plt.show()

# show image 
plt.imshow(cp_palettes[0]) #now it is in RGB 
plt.show() 
   
#%%            
import cv2
import numpy as np
from matplotlib import pyplot as plt


def show_hist(file): 
    pass 
    
def get_hist(file, color_space): 
    img = cv2.imread(file) # in bgr
    # function signature: images, channels, mask, number of bins, array of the dims arrays of the histogram bin boundaries in each dimension.
    if color_space == "bgr": 
        hist = cv2.calcHist([img],[0, 1, 2],None,[256, 256, 256],[0, 255, 0, 255, 0, 255]) # in bgr 
    if color_space == "lab": 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB) # in lab 
        hist = cv2.calcHist([img],[0, 1, 2],None,[100, 2*128, 2*128],[0, 100, -128, 127, -128, 127]) # function signature: images, channels, mask, number of bins, array of the dims arrays of the histogram bin boundaries in each dimension.
        # normalize 
#    hist = cv2.normalize(hist,hist,0,255,cv2.NORM_MINMAX)
    if color_space == "rgb": 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # in rgb
        hist = cv2.calcHist([img],[0, 1, 2],None,[256, 256, 256],[0, 255, 0, 255, 0, 255]) # in rgb 

    return hist 
    
def save_hist(file): 
    img = cv2.imread(file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    hist = cv2.calcHist([img],[0],None,[256],[0,256])
    # TODO convert back LAB2BGR for display
    plt.plot(hist)
    color = ('b','g','r')
    for i,col in enumerate(color):
        histr = cv2.calcHist([img],[i],None,[256],[0,256])
        plt.plot(histr,color = col)
        plt.xlim([0,256])
    plt.savefig(f"{file[:-4]}_histogram.jpg")
    plt.show()



#%%
# all color palettes    
cp_hists = []

# show histogram of color palettes of images
#for FILE in FILES: 
#    show_hist(FILE)
# 
# get histogram of color palettes of images  
CS = "lab"
for FILE in FILES: # out-of-memory error    
    hist = get_hist(FILE, CS)
    cp_hists.append(hist)
    
# save histogram of color palettes of images to directory
#for FILE in FILES: 
#    save_hist(FILE)

print(f"Total of {len(cp_hists)} color palettes processed.")

#%%
# compare histograms 
# based on : https://docs.opencv.org/2.4/doc/tutorials/imgproc/histograms/histogram_comparison/histogram_comparison.html
# metrics to compare histograms
correlation = cv2.HISTCMP_CORREL 
chi_square = cv2.HISTCMP_CHISQR 
intersection = cv2.HISTCMP_INTERSECT 
bhattacharyya = cv2.HISTCMP_BHATTACHARYYA 

#%%
# single images

# images
image1 = FILES.index('frame2125_lab_palette.jpg') #why corr=1?
image2 = FILES.index('frame500_lab_palette.jpg')
image3 = FILES.index('frame1625_lab_palette.jpg')

# compareHist: how well two histograms match with each other.
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
 
# dense or sparse histogram 
print(f"Corr1: {corr1}, Corr2: {corr1}, Corr3: {corr3}")
print(f"Chi1: {chi1}, Chi2: {chi2}, Chi3: {chi3}")
print(f"Intsec1: {intsec1}, Intsec2: {intsec2}, Intsec3: {intsec3}")
print(f"Bhacha1: {bhacha1}, Bhacha2: {bhacha2}, Bhacha3: {bhacha3}")


#%%
# all image combinations

# all pairwise pairs (cartesian product) 
import time 
import itertools

start = time.time()


kombinatorik = ['cartesian_product', 'combis', 'permuts']
# performance in seconds: {'cartesian_product': 3.14, 'permuts': 2.82, 'combis': 1.48}

WAY = kombinatorik[1]

if WAY == 'cartesian_product':
    print(f"Processing {len(cp_hists)*len(cp_hists)}")
    pairit = [p for p in itertools.product(cp_hists, repeat=2)]
    ids_pairs =  [p for p in itertools.product(range(len(cp_hists)), repeat=2)]
if WAY == 'combis': 
    print(f"Processing {(len(cp_hists)*len(cp_hists)-len(cp_hists))/2}")
    pairit = list(itertools.combinations(cp_hists, 2))
    ids_pairs =  list(itertools.combinations(range(len(cp_hists)), 2))
if WAY == 'permuts': 
    print(f"Processing {len(cp_hists)*len(cp_hists)-len(cp_hists)}")
    pairit = list(itertools.permutations(cp_hists, 2))
    ids_pairs =  list(itertools.permutations(range(len(cp_hists)), 2))

# metrics 
metrics = [correlation, chi_square, intersection, bhattacharyya]
metrics_name = ['correlation', 'chi_square', 'intersection', 'bhattacharyya']
metrix = dict(zip(metrics_name, metrics))

# to specify (metric)
METRIC = metrics_name[0]

# metrics of pairwise pairs (cartesian product) 
mtrs = []
pairs = []

# loop through cartesian product 
for i, p in enumerate(pairit): 
    print(f"{i} processed.")
    corr = cv2.compareHist(p[0], p[1], metrix[METRIC])
    pair = ids_pairs[i]
    mtrs.append(corr)
    pairs.append(pair)

print(mtrs[:5])
print(pairs[:5])
print(len(mtrs))
print(len(pairs))


# reshape: list to nested list 
if WAY == 'cartesian_product':
    nested_list = [mtrs[i:i+len(cp_hists)] for i in range(0, len(mtrs), len(cp_hists))]
    assert len(nested_list[0]) == len(nested_list), "Matrix is not symmetrical."
    # build dataframe: hist1, hist2... x hist1, hist2,.... cell= correlation
    df = pd.DataFrame(nested_list, cp_names, cp_names)
if WAY == 'combis':
    # initialize dataframe 
    df = pd.DataFrame([[0 for i in range(len(cp_names))] for j in range(len(cp_names))], cp_names, cp_names)
    for i, p in enumerate(pairs): 
        df.iloc[p[0], p[1]] = mtrs[i]
        df.iloc[p[1], p[0]] = mtrs[i]
    for i in range(len(cp_names)): 
        df.iloc[i, i] = 1
if WAY == 'permuts': 
    df = pd.DataFrame([[0 for i in range(len(cp_names))] for j in range(len(cp_names))], cp_names, cp_names)
    for i, p in enumerate(pairs): 
        df.iloc[p[0], p[1]] = mtrs[i]
    for i in range(len(cp_names)): 
        df.iloc[i, i] = 1       



end = time.time()
duration = round(end - start, 2)
print(f"{WAY} took this amount of time:")
print(f"{duration} sec")
performance[WAY] = duration 

#%%
# search query: single file 

# get top-n closest palettes for given palette

# Search request - Finding result 
# to specify - USER SPECIFICATION (VIAN)
SEARCHKEY_PALETTE = "45443" #choose from: cp_names

TOPN = 20 
# search only in lowest row, thus cannot filter by palette depth, threshold ratio and colorbar count because of ratio width information only there in whole palette image filters
assert TOPN <= len(FILES), "Not as many files to evaluate."

# get pair-partners for given palette and sort pbonds 
gold_pbonds = df.loc[SEARCHKEY_PALETTE].sort_values(ascending=False)[:TOPN]
gold_palettes = gold_pbonds.index.tolist()


        
# show top-n closest pbonds for a given palette 
print("-------------------------")
print(f"Task: Find most similar color palettes")
print(f"Searching color palette id: {SEARCHKEY_PALETTE}")
print(f"Total number of gold palettes: {len(gold_pbonds)}")
print(f"Top-{TOPN} gold palettes: \n{gold_palettes}")
print("-------------------------")


 
#%%

# show search results (golden waterfalls): single file 
name2palette= dict(zip(cp_names, cp_palettes))
name2img = dict(zip(cp_names, cp_images))

if not any(gold_palettes): 
    print(f"No palettes found.")
else: 
    print("-------------------------")
    print("Search query (input): ")
    print(f"{SEARCHKEY_PALETTE}")
    # show palette 
    plt.imshow(name2palette[SEARCHKEY_PALETTE]) #now it is in RGB 
    plt.show() 
    # show image 
    plt.imshow(name2img[SEARCHKEY_PALETTE]) #now it is in RGB 
    plt.show() 
    # save search query
    FOLDER_NAME = f"search_query_{SEARCHKEY_PALETTE}_pool{len(FILES)}_top{TOPN}"
    os.chdir(PALETTE_PATH)
    try: 
        os.mkdir(FOLDER_NAME)
        new_path = os.path.join(PALETTE_PATH, FOLDER_NAME)
        os.chdir(new_path)
    except: 
        os.path.join(PALETTE_PATH, FOLDER_NAME)
        os.chdir(new_path)
    # save palette
    plt.imshow(name2palette[SEARCHKEY_PALETTE]) #now it is in RGB 
    plt.axis('off')
    plt.savefig(f"0_{SEARCHKEY_PALETTE}_palette.jpg")
    # save image
    plt.imshow(name2img[SEARCHKEY_PALETTE])
    plt.axis('off')
    plt.savefig(f"0_{SEARCHKEY_PALETTE}_image.jpg")
    
    # save results
    SAVE = True 
    print("-------------------------")
    print(f"Display top {TOPN} most similar palettes (output):")
    for i, filename in enumerate(gold_palettes):
        print(f"{i+1}: {filename}")    
        # show palette 
        plt.imshow(name2palette[filename]) #now it is in RGB 
        plt.axis('off')
        if SAVE: 
            plt.savefig(f"{i+1}_{filename}_palette.jpg")
        plt.show() 
        # show image 
        plt.imshow(name2img[filename]) #now it is in RGB 
        plt.axis('off')
        if SAVE: 
            plt.savefig(f"{i+1}_{filename}_image.jpg")
        plt.show() 

#%%
# search query: for all loaded files = search files
# get top-n closest palettes for given palette and save it to folder of file

# foldername = filename of search file
# folder contents: search file as 0, result files with top-n index 

# Search request - Finding result 
# to specify - USER SPECIFICATION (VIAN)
for i, filename in enumerate(cp_names): 
    SEARCHKEY_PALETTE = filename
    print(f"{i} processed.")
    
    TOPN = 10 
    # search only in lowest row, thus cannot filter by palette depth, threshold ratio and colorbar count because of ratio width information only there in whole palette image filters
    assert TOPN <= len(FILES), "Not as many files to evaluate."
    
    # get pair-partners for given palette and sort pbonds 
    gold_pbonds = df.loc[SEARCHKEY_PALETTE].sort_values(ascending=False)[:TOPN]
    gold_palettes = gold_pbonds.index.tolist()
    
            
    # show top-n closest pbonds for a given palette 
    print("-------------------------")
    print(f"Task: Find most similar color palettes")
    print(f"Searching color palette id: {SEARCHKEY_PALETTE}")
    print(f"Total number of gold palettes: {len(gold_pbonds)}")
    print(f"Top-{TOPN} gold palettes: \n{gold_palettes}")
    print("-------------------------")
    
    
    # show search results (golden waterfalls)
    name2palette= dict(zip(cp_names, cp_palettes))
    name2img = dict(zip(cp_names, cp_images))
    
    if not any(gold_palettes): 
        print(f"No palettes found.")
    else: 
        print("-------------------------")
        print("Search query (input): ")
        print(f"{SEARCHKEY_PALETTE}")
        # show palette 
        plt.imshow(name2palette[SEARCHKEY_PALETTE]) #now it is in RGB 
        plt.show() 
        # show image 
        plt.imshow(name2img[SEARCHKEY_PALETTE]) #now it is in RGB 
        plt.show() 
        # save search query
        FOLDER_NAME = f"search_query_{SEARCHKEY_PALETTE}_pool{len(FILES)}_top{TOPN}_metric{METRIC}_cs{CS}"
        os.chdir(PALETTE_PATH)
        try: 
            os.mkdir(FOLDER_NAME)
            new_path = os.path.join(PALETTE_PATH, FOLDER_NAME)
            os.chdir(new_path)
        except: 
            os.path.join(PALETTE_PATH, FOLDER_NAME)
            os.chdir(new_path)
        # save palette
        plt.imshow(name2palette[SEARCHKEY_PALETTE]) #now it is in RGB 
        plt.axis('off')
        plt.savefig(f"0_{SEARCHKEY_PALETTE}_palette.jpg")
        # save image
        plt.imshow(name2img[SEARCHKEY_PALETTE])
        plt.axis('off')
        plt.savefig(f"0_{SEARCHKEY_PALETTE}_image.jpg")
        
        # save results
        SAVE = True 
        print("-------------------------")
        print(f"Display top {TOPN} most similar palettes (output):")
        for i, filename in enumerate(gold_palettes):
            print(f"{i+1}: {filename}")    
            # show palette 
            plt.imshow(name2palette[filename]) #now it is in RGB 
            plt.axis('off')
            if SAVE: 
                plt.savefig(f"{i+1}_{filename}_palette.jpg")
            plt.show() 
            # show image 
            plt.imshow(name2img[filename]) #now it is in RGB 
            plt.axis('off')
            if SAVE: 
                plt.savefig(f"{i+1}_{filename}_image.jpg")
            plt.show() 

  

#%%
# save pair bonds dataframe
os.chdir(r'D:\thesis\code\pd4cpbonds')
df.to_excel(f"palette_pair_pbonds_pool{len(FILES)}_top{TOPN}_metric{METRIC}_cs{CS}.xlsx", index=True)    

#%%

# load pair bonds dataframe
os.chdir(r'D:\thesis\code\pd4cpbonds')
df = pd.read_excel(f"palette_pair_pbonds_pool{len(FILES)}_top{TOPN}_metric{METRIC}_cs{CS}.xlsx", index_col=0)    

