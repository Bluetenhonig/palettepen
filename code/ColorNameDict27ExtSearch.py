# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 09:11:42 2020

@author: lsamsi



The extention will extend the EFFCND to an Extended-EFFCND (EEFCND) dataframe.  

The extentions steps: 
    
1. EFFCND: remove "cat1" and "cat2" and copy-past column "name" to column "cat"
2. download image for all color names and get at least two more different averages for same color name
 by averaging over different set lengths of google images
3. fill out remaining cells if possible (not possible for color name) using duplication or color conversion methods


columns =  [id, lang, name, image, srgb, srgb_r, srgb_g, srgb_b, hsv, hsv_h, hsv_s, hsv_v, lab, lab_l, lab_a, lab_b, hex, cat]
filename = "eeffcnd_"+source+"_"+method+".xlsx" where method: search  


Goal: extend all unique color names with color values that can be categroized into them 


Step Before: EFFCND (dict colors classified into basic colors)
Goal: EEFCND (no basic colors ie basic colors equal to dict colors, new color values for each dict color)
Step AFter: visualization (?), train machine learning model 
    
"""


# load modules
import os
import pandas as pd
import numpy as np


#%%

### Color-Thesaurus EPFL ###

PATH = r'D:\thesis\input_color_name_dictionaries\thesaurus\datasets'
FILE = 'effcnd_thesaurus_basicvian.xlsx'
OUTPUT_FILE = 'eeffcnd_thesaurus_search.xlsx' 

# set directory 
os.chdir(PATH)

# load data 
data = pd.read_excel(FILE, sep=" ", index_col=0)
data.info()
data.columns

data = data.drop(['cat1', 'cat2'], axis=1)
data['cat'] = data['name']
data[data['name'] == 'ultramarine']


#%%

# PART 1: web scraping  

searchkeys = ['apricot color', 'ultramarine color']
searchkeyurls = {}

# to specify: here only first 5 
colorn = data['name'].tolist()[:5]

# for color in colorn: 
#     colorn = color + ' color' 
#     searchkeys.append(colorn)

# to specify 
SEARCHKEY = None 
MAXLINKS = 100 
MARGIN = 10 # margin for corrupt images
TOFETCH = MAXLINKS + MARGIN 
FOLDER_PATH = r'D:\thesis\input_images\google'


#%%

# get image urls from browser 

import time 
import io
from selenium import webdriver
import sys 
sys.path.append(r'D:\thesis\input_color_name_dictionaries\thesaurus\datasets')

def fetch_image_urls(query, max_links_to_fetch, wd, results_start=0, sleep_between_interactions=1):
    
    def scroll_to_end(wd):
        wd.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(sleep_between_interactions)    
    
    # build the google query
    search_url = "https://www.google.com/search?safe=off&site=&tbm=isch&source=hp&q={q}&oq={q}&gs_l=img"

    # load the page
    wd.get(search_url.format(q=query))

    image_urls = set()
    image_count = 0
    results_start = results_start
    while image_count < max_links_to_fetch:
        scroll_to_end(wd)

        # get all image thumbnail results
        thumbnail_results = wd.find_elements_by_css_selector("img.Q4LuWd")
        number_results = len(thumbnail_results)
        
        print(f"Found: {number_results} search results. Extracting links from {results_start}:{number_results}")
        
        for img in thumbnail_results[results_start:number_results]:
            # try to click every thumbnail such that we can get the real image behind it
            try:
                img.click()
                time.sleep(sleep_between_interactions)
            except Exception:
                continue

            # extract image urls    
            actual_images = wd.find_elements_by_css_selector('img.n3VNCb')
            for actual_image in actual_images:
                if actual_image.get_attribute('src') and 'http' in actual_image.get_attribute('src'):
                    image_urls.add(actual_image.get_attribute('src'))

            image_count = len(image_urls)

            if len(image_urls) >= max_links_to_fetch:
                print(f"Found: {len(image_urls)} image links, done!")
                break
        else:
            print("Found:", len(image_urls), "image links, looking for more ...")
            time.sleep(30)
            return
            load_more_button = wd.find_element_by_css_selector(".mye4qd")
            if load_more_button:
                wd.execute_script("document.querySelector('.mye4qd').click();")

        # move the result startpoint further down
        results_start = len(thumbnail_results)
        
    wd.close()
    image_urls = list(image_urls)
#    if len(image_urls) != max_links_to_fetch:       
#        image_urls = image_urls[:max_links_to_fetch]
    return image_urls

for searchkey in searchkeys: 
    # supports only chrome version 81 
    DRIVER = webdriver.Chrome('D:/thesis/input_color_name_dictionaries/thesaurus/datasets/chromedriver.exe')
    urls = fetch_image_urls(searchkey, TOFETCH, DRIVER,1)
    DRIVER.quit()
    searchkeyurls[searchkey] = urls


#%% 

# persist image

import os
import requests
from PIL import Image
import hashlib

def persist_image(folder_path:str,url:str, index):
    try:
        image_content = requests.get(url).content

    except Exception as e:
        print(f"ERROR - Could not download {url} - {e}")

    try:
        image_file = io.BytesIO(image_content)
        image = Image.open(image_file).convert('RGB')
        file_path = os.path.join(folder_path,f'{index}_'+hashlib.sha1(image_content).hexdigest()[:10] + f'.jpg')
        #file_path = os.path.join(folder_path,hashlib.sha1(image_content).hexdigest()[:10] + '.jpg')
        with open(file_path, 'wb') as f:
            image.save(f, "JPEG", quality=85)
        print(f"SUCCESS - saved {url}")
        return url 
    except Exception as e:
        print(f"ERROR - Could not save {url} - {e}")
        return 0

# saves images to folder (unsorted)
for key, urls in searchkeyurls.items(): 
    color = key.replace(' color', '')
    try: 
        searchkey_folder = os.path.join(FOLDER_PATH, color)
        os.mkdir(searchkey_folder)  
    except: 
        searchkey_folder = os.path.join(FOLDER_PATH, color)
    savedimages = []
    for i in range(len(urls)): 
        print(i+1)
        url = persist_image(searchkey_folder, urls[i],i)
        if url != 0: 
            savedimages.append(url)
        if len(savedimages) == MAXLINKS: 
            break
    
    print("---")
    print(f"SUCCESS - {len(savedimages)} images saved into {searchkey_folder}")
    print("---")



#%%
    
# PART 2: calculation 
    
# silently 

import os


colorfiles = {}

for searchkey in searchkeys:
    color = searchkey.replace(' color', '')
    searchkey_folder = os.path.join(FOLDER_PATH, color)
    
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(searchkey_folder):
        for file in f:
            if '.jpg' in file:
                files.append(os.path.join(r, file))
    print("First 5 files: ", files[:5])
    print(len(files))
    colorfiles[color] = files 
    



#%%
# find average color of all images in LAB 
import cv2 
import matplotlib.pyplot as plt 
import sys 
sys.path.append(r'D:\thesis\code')
from ColorConversion00000 import convert_color
from ColorConversion00000  import * 

coloravgs = {}

# at least two more colors 
for key, files in colorfiles.items():       
    filelen1 = len(files)
    filelen2 = len(files) // 2
    filelengths = {}
    for lengths in (filelen1, filelen2):         
        avgs = []
        for i in range(lengths): 
            # load image in BGR with numpy.uint8, 0-255 val 
            image = cv2.imread(files[i])  
            flt_image = image.astype(np.float32) / 255
            # convert image to LAB 
            lab_image = cv2.cvtColor(flt_image, cv2.COLOR_BGR2Lab)
            # convert image to RGB   
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # plot image 
            # plt.imshow(rgb_image)
            # plt.show()
            # make mask 
            hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV) #make a copy in HSV
            lowerb = (0, 40, 80) # all values smaller than hsv
            upperb = (255, 255, 190) # all values bigger, yields only positive difference
            mask = cv2.inRange(hsv_image, lowerb, upperb) # inRange(first input array src, lowerb, upperb, [dst])  
            # show mask 
            plt.imshow(mask, cmap="gray")
            masked_data = cv2.bitwise_and(lab_image, lab_image, mask=mask) # in RGB 
            plt.imshow(masked_data)
            lab_image = masked_data
            # replace zeroes with nan
            lab_image = np.where(lab_image!=0,lab_image,np.nan) 
        #     calculate average LAB
            average = np.nanmean(np.nanmean(lab_image,0),0)  #image.mean(axis=0).mean(axis=0)
        #    if np.isnan(np.array(average[0])) == 0: # if MARGIN too big 
            avgs.append(list(average))
        
    

        # calculate average of averages LAB
        avgs = np.array(avgs)
        avgavg = np.nanmean(avgs, 0) 
        # transform to RGB for display
        avgavg = convert_color(avgavg, "LAB", "RGB", lab2rgb)
        avgcolor = list(avgavg)
        avgcolor = round(avgcolor[0]), round(avgcolor[1]), round(avgcolor[2])
        print(f"Average RGB color across all images: {avgcolor}")
        filelengths[lengths] = avgcolor
    coloravgs[key] = filelengths 

#%% 
# show average of averages color
for key, value in coloravgs.items():
    for length, avgavg  in value.items(): 
        print(str(key), ': ',  str(length))
        print(avgavg)
        a = np.full((20, 20, 3), avgavg, dtype=np.uint8)
        plt.imshow(a) #now it is in RGB 
        plt.axis('off')
        plt.show()



#%%

# transform dataframe 

name = data['name'].tolist()
srgb = data['srgb'].tolist()

new_values = []
new_colors = []
for col, val in coloravgs.items(): 
    for val, avgrgb in val.items(): 
        color = str(col) + ' ' + str(val) 
        new_colors.append(color)
        avgrgb = str(list(avgrgb))
        new_values.append(avgrgb)

lst_len = len(name) + len(new_colors)
indices = list(range(0, lst_len, 3))
cindices = list(range(0, lst_len, 2))
final_name = [None] * lst_len

for i, na in enumerate(name[:5]):
    final_name[indices[i]] = na
    final_name[indices[i]+1] = new_colors[cindices[i]]
    final_name[indices[i]+2] = new_colors[cindices[i]+1]

final_name[15:]=name[5:]

final_name[:20]
final_name[-5:]

#%%

final_lang = ['eng'] * lst_len


final_cat = [None] * lst_len
final_cat[:15] = list(np.repeat(name[:5], 3))
final_cat[15:]=name[5:]

final_srgb = [None] * lst_len

final_srgb[:20]
type(data['name'][0])

for i, rgb in enumerate(srgb[:5]):
    final_srgb[indices[i]] = rgb
    final_srgb[indices[i]+1] = new_values[cindices[i]]
    final_srgb[indices[i]+2] = new_values[cindices[i]+1]

final_srgb[15:]=srgb[5:]   


final_rgb_R = [eval(r)[0] for r in final_srgb]
final_rgb_G = [eval(r)[1] for r in final_srgb]
final_rgb_B = [eval(r)[2] for r in final_srgb]


#%%

# restructure
data2 = pd.DataFrame({
        'id': range(lst_len),
        'lang': final_lang,
        'name': final_name,
        'srgb': final_srgb,
        'srgb_R': final_rgb_R,
        'srgb_G': final_rgb_G,
        'srgb_B': final_rgb_B,
        'hsv': None,
       'hsv_H': None
       , 'hsv_S': None
       , 'hsv_V': None
       , 'cielab': None
       , 'cielab_L': None
       , 'cielab_a': None
       , 'cielab_b': None
       ,'hsl': None
       , 'hsl_H': None
       , 'hsl_S': None
       , 'hsl_L': None
       , 'LCH': None
       , 'LCH_L': None
       , 'LCH_C': None
       , 'LCH_H': None
       , 'hex': None
        ,'cat': final_cat
        })





#%%

# save data (if adobe is first color! )
data2.to_excel(OUTPUT_FILE)

