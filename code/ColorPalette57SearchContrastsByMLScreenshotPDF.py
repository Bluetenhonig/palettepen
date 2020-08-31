# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 16:25:48 2020

@author: Linda Samsinger

=====================
Classification Visualization
=====================

TOP-DOWN APPROACH:
The results of the ML classifier's prediction for each screenshot
of a movie will be handed in to Prof. Flückiger for prediction 
evaluation. 

"""

#####################################
### Load Data 
 
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2

#%%
# load color palettes dataframe 
# palette/s
PALETTE_PATH = r'D:\thesis\film_colors_project\sample-dataset\screenshots\7\bgr_palette'
EXTENSION = 'bgr_palette.csv'
PALETTE_FILE = 'frame125_bgr_palette.csv'
#FILES = ['frame250.jpg', 'frame375.jpg']     # for a list of images to process 
# load files from directory 
#FILES = ['frame12625_bgr_palette.csv', 'frame125_bgr_palette.csv']
FILES = []
for r, d, f in os.walk(PALETTE_PATH): # r=root, d=directories, f = files
    for file in f:
        if EXTENSION in file:
            FILES.append(file) 

FILES = sorted(FILES)
print('Number of color palettes: ', len(FILES))

# palette/s
CP_PATH = r'D:\thesis\film_colors_project\sample-dataset\screenshots\7\D100_lab_palette'
EXTENSION = '_D100_lab_palette.jpg'
PALETTE_FILE = '45452_D100_lab_palette.jpg'
#FILES = ['frame250.jpg', 'frame375.jpg']     # for a list of images to process 
# load files from directory 
PALETTES = []
for r, d, f in os.walk(CP_PATH): # r=root, d=directories, f = files
    for file in f:
        if EXTENSION in file:
            PALETTES.append(file) 

PALETTES = sorted(PALETTES)
print('Number of color palette images: ', len(PALETTES))


IMG_PATH = r'D:\thesis\film_colors_project\sample-dataset\screenshots\7\7\images'
EXTENSION = '.jpg'
IMAGES = []
for r, d, f in os.walk(IMG_PATH): # r=root, d=directories, f = files
    for file in f:
        if EXTENSION in file:
            IMAGES.append(file) 

IMAGES = sorted(IMAGES)          
print('Number of images: ', len(IMAGES))

assert len(FILES) == len(IMAGES), 'Number of color palettes not equal to number of images.'
assert len(FILES) == len(PALETTES), 'Number of color palettes not equal to number of images.'


#%%
# to specify
DATASET_PATH = r'D:\thesis\film_colors_project\sample-dataset'

# load data (requirements: LAB, (RGB,) HEX)
os.chdir(DATASET_PATH)
data = pd.read_excel('dataset.xlsx', index_col=[0])
jigokumon = data[data.project_id==7]
jigokumon.columns[-10:-4]
cols = jigokumon.columns

predicts = pd.read_excel('KNN21_dataset.xlsx', index_col=[0])    
predicts.info()



#%%
# analysis
no_of_segm = jigokumon['segment_id'].nunique()  
no_of_scrn = jigokumon['screenshot_id'].nunique() 
range_of_segm = [jigokumon['segment_id'].min(), jigokumon['segment_id'].max()]
range_of_scrn = [jigokumon['screenshot_id'].min(), jigokumon['screenshot_id'].max()]  
         
print('Number of segments: ', no_of_segm)
# 54, range = [312, 365]
print('Number of screenshots: ',no_of_scrn)
# 569, range =  [45442, 46010]

# processing 
IMAGES = sorted(IMAGES)
IMAGES[:5]
SEGMENTS= jigokumon['segment_id'].unique().tolist()
SEGMENTS[:5]
jigokumon = jigokumon.sort_values(by=['segment_id'], ascending=True)
jigokumon[['screenshot_id', 'segment_id']].head()
groups = jigokumon['screenshot_id'].groupby(jigokumon['segment_id']).agg('count').to_frame()


#%%
IMG_len = len(IMAGES)  

print('------------------------') 
print(cols[0], ':', jigokumon[cols[0]][jigokumon['screenshot_id'] == np.array(IMAGES[0][:-4], dtype='uint64')].iloc[0])
print(cols[3], ':', jigokumon[cols[3]][jigokumon['screenshot_id'] == np.array(IMAGES[0][:-4], dtype='uint64')].iloc[0])
print(cols[4], ':', jigokumon[cols[4]][jigokumon['screenshot_id'] == np.array(IMAGES[0][:-4], dtype='uint64')].iloc[0])
print(cols[5], ':', jigokumon[cols[5]][jigokumon['screenshot_id'] == np.array(IMAGES[0][:-4], dtype='uint64')].iloc[0])
print(cols[6], ':', jigokumon[cols[6]][jigokumon['screenshot_id'] == np.array(IMAGES[0][:-4], dtype='uint64')].iloc[0])
   
# for all information
for img in IMAGES[:5]: 
    print('------------------------')   
    # show image information         
    print(cols[1], ':', jigokumon[cols[1]][jigokumon['screenshot_id'] == np.array(img[:-4], dtype='uint64')].iloc[0])
    print(cols[2], ':', jigokumon[cols[2]][jigokumon['screenshot_id'] == np.array(img[:-4], dtype='uint64')].iloc[0])
    
    # show image
    os.chdir(IMG_PATH)
    image = cv2.imread(img)           
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image) #now it is in RGB 
    plt.axis('off')
    plt.show()
    image.shape
    
    # all info about contrasts
    inst = predicts[predicts['Palette_id']==eval(img[:-4])]
    for i in range(1, 20):
        col = predicts.columns[i]
        print(col, ':', inst[col].iloc[0])

#%%
IMG_len = len(IMAGES) 

# to specify 
IMG_number = '45447'
IMG_index = IMAGES.index('45547.jpg')

print('------------------------') 
print(cols[0], ':', jigokumon[cols[0]][jigokumon['screenshot_id'] == np.array(IMAGES[0][:-4], dtype='uint64')].iloc[0])
print(cols[3], ':', jigokumon[cols[3]][jigokumon['screenshot_id'] == np.array(IMAGES[0][:-4], dtype='uint64')].iloc[0])
print(cols[4], ':', jigokumon[cols[4]][jigokumon['screenshot_id'] == np.array(IMAGES[0][:-4], dtype='uint64')].iloc[0])
print(cols[5], ':', jigokumon[cols[5]][jigokumon['screenshot_id'] == np.array(IMAGES[0][:-4], dtype='uint64')].iloc[0])
print(cols[6], ':', jigokumon[cols[6]][jigokumon['screenshot_id'] == np.array(IMAGES[0][:-4], dtype='uint64')].iloc[0])
    
before = [0]    
# only true information
for img in IMAGES: 
    print('------------------------')   
    # show image info 
    if jigokumon[cols[1]][jigokumon['screenshot_id'] == np.array(img[:-4], dtype='uint64')].iloc[0] != before[-1]:         
        print('------------------------')
        print('------NEW SEGMENT-------')
        print('------------------------')
    
    print(str(cols[1]), ':', str(jigokumon[cols[1]][jigokumon['screenshot_id'] == np.array(img[:-4], dtype='uint64')].iloc[0]))
    before.append(jigokumon[cols[1]][jigokumon['screenshot_id'] == np.array(img[:-4], dtype='uint64')].iloc[0])
    print(cols[2], ':', jigokumon[cols[2]][jigokumon['screenshot_id'] == np.array(img[:-4], dtype='uint64')].iloc[0])
   # show image
    image = cv2.imread(img)           
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image) #now it is in RGB 
    plt.axis('off')
    plt.show()
    image.shape
    
    # info about contrasts if there 
    inst = predicts[predicts['Palette_id']==eval(img[:-4])]
    eln = 0 
    for i in range(1, 20):      
        col = predicts.columns[i]
        print(col, ':', inst[col].iloc[0])
        eln += 1
        

        
    
    
#%%


# save images to pdf with contrast classification for only true information
import fpdf

# create pdf instance 
pdf = fpdf.FPDF(format='A4')
# Document Properties in Adobe Reader 
title = 'FilmColors Project: \nFilm Screenshots and Information about Color Contrasts'
pdf.set_title(title)
pdf.set_author('Linda Samsinger')
pdf.accept_page_break()

# add title page 
pdf.add_page()
pdf.set_font("Arial", 'B', size=20)
pdf.multi_cell(0, 10, title, align = 'L')
pdf.ln()
pdf.set_font("Arial", size=18)
a = f'{cols[3]} : '+ str(jigokumon[cols[3]][jigokumon['screenshot_id'] == np.array(IMAGES[0][:-4], dtype='uint64')].iloc[0])
b = f'{cols[0]} : ' + str(jigokumon[cols[0]][jigokumon['screenshot_id'] == np.array(IMAGES[0][:-4], dtype='uint64')].iloc[0])
c = f'{cols[4]} : ' + str(jigokumon[cols[4]][jigokumon['screenshot_id'] == np.array(IMAGES[0][:-4], dtype='uint64')].iloc[0])
d = f'{cols[5]} : ' + str(jigokumon[cols[5]][jigokumon['screenshot_id'] == np.array(IMAGES[0][:-4], dtype='uint64')].iloc[0])
e = f'{cols[6]} : ' + str(jigokumon[cols[6]][jigokumon['screenshot_id'] == np.array(IMAGES[0][:-4], dtype='uint64')].iloc[0])
pdf.multi_cell(0, 10, a, align = 'L')
pdf.multi_cell(0, 10, b, align = 'L')
pdf.multi_cell(0, 10, c, align = 'L')
pdf.multi_cell(0, 10, d, align = 'L')
pdf.multi_cell(0, 10, e, align = 'L')
pdf.multi_cell(0, 10, '---------------------------------------', align = 'L')
pdf.multi_cell(0, 10, 'Number of segments: '+ str(no_of_segm) + ' ' + str(range_of_segm), align = 'L')
pdf.multi_cell(0, 10, 'Number of screenshots: '+ str(no_of_scrn) + ' ' + str(range_of_scrn), align = 'L')
pdf.multi_cell(0, 10, '---------------------------------------', align = 'L')
pdf.set_font("Arial", size=12)
pdf.multi_cell(0, 10, 'Color Contrast Classification: \n1. Contrast of hue: At least 3 different colors exist. \n2. Light-dark contrast: Luminance values (LCH color space) of more than 75 and less than 25 are simultaneously present. \n3. Cold-warm contrast: At least one cold and one warm color is present at the same time. \n4. Complementary contrast: At least green-red or blue-orange or violet-yellow are simultaneously present. \n 5. Contrast of saturation: Saturation values (LCH color space) of more than 50 and less than 25 are simultaneously present. \n6. Contrast of extension: Sub-category of contrast of saturation where a saturated area of less than 2%, but bigger than .5% pops out against a desaturated area of more than 98%. \nDISCLAIMER: Medium ranges for lumens and tone are not indicated (dark<25, light>75) as they are not contributing factors for contrast classification.', align = 'L')


# loop out other pages 
pdf.set_font("Arial", size=12)
i = 0  
before = [0]

for idx, img in enumerate(IMAGES):
    pdf.add_page()
    # index page
    pdf.set_font("Arial", 'B', size=12)
    pdf.multi_cell(0, 20, "Screenshot: " + str(idx+1))
    pdf.set_font("Arial", size=12)
    str1 = f'{cols[1]} : ' + str(jigokumon[cols[1]][jigokumon["screenshot_id"] == np.array(img[:-4], dtype="uint64")].iloc[0])
    pdf.multi_cell(0, 5,str1)
    pdf.ln()
    str2 = f'{cols[2]} : ' + str(jigokumon[cols[2]][jigokumon['screenshot_id'] == np.array(img[:-4], dtype='uint64')].iloc[0])
    pdf.multi_cell(0, 0,str2)
    # show next segment 
    if jigokumon[cols[1]][jigokumon['screenshot_id'] == np.array(img[:-4], dtype='uint64')].iloc[0] != before[-1]:         
        pdf.text(165, 20, 'Start of Segment')
    before.append(jigokumon[cols[1]][jigokumon['screenshot_id'] == np.array(img[:-4], dtype='uint64')].iloc[0])
    # show images
    os.chdir(IMG_PATH)
    pdf.image(img, x = 10, y = 45, w = 50, h = 50, type = 'JPG')   
    os.chdir(CP_PATH)
    pdf.image(PALETTES[idx], x = 70, y = 45, w = 50, h = 50, type = 'JPG')
    # show color contrast categories
    pdf.set_font("Arial", 'B', size=12)
    pdf.multi_cell(0, 130, 'Color Contrasts: ')
    pdf.set_font("Arial", size=12)
    inst = predicts[predicts['Palette_id']==eval(img[:-4])]
    compl = []
    for i in range(5,8): 
        el = predicts.columns[i]
        if inst[el].iloc[0] == 1:
            compl.append(el)
#    compl = ', '.join(compl).lower()
    lgth = len(compl)
    if lgth == 1:         
        if 'Green-red' in compl: 
            gred = str(inst['Green'].iloc[0]) +': ' + str(inst['Red'].iloc[0])     
            compl = ', '.join(compl).lower() + ' (' +  gred + ')' 
        elif 'Blue-orange' in compl: 
            blorange = str(inst['Blue'].iloc[0]) +': ' + str(inst['Orange'].iloc[0]) 
            compl = ', '.join(compl).lower() + ' (' +  blorange + ')'
        else: 
            vyellow = str(inst['Violet'].iloc[0]) +': ' + str(inst['Yellow'].iloc[0]) 
            compl = ', '.join(compl).lower() + ' (' + vyellow + ')' 
    elif lgth == 2:       
        if 'Green-red' in compl and 'Blue-orange' in compl:
            gred =  str(inst['Green'].iloc[0]) +': ' + str(inst['Red'].iloc[0])     
            blorange =  str(inst['Blue'].iloc[0]) +': ' + str(inst['Orange'].iloc[0]) 
            compl = ', '.join(compl).lower() + ' (' +  gred + '; ' + blorange + ')' 
        elif 'Green-red' in compl and 'Violet-yellow' in compl:
            gred =  str(inst['Green'].iloc[0]) +': ' + str(inst['Red'].iloc[0])     
            vyellow =  str(inst['Violet'].iloc[0]) +': ' + str(inst['Yellow'].iloc[0]) 
            compl = ', '.join(compl).lower() + ' (' +  gred + '; ' + vyellow + ')' 
        else: 
            blorange = str(inst['Blue'].iloc[0]) +': ' + str(inst['Orange'].iloc[0]) 
            vyellow = str(inst['Violet'].iloc[0]) +': ' + str(inst['Yellow'].iloc[0]) 
            compl = ', '.join(compl).lower() + ' (' +  blorange + '; ' + vyellow + ')'           
    else: 
        gred = str(inst['Green'].iloc[0]) +':' + str(inst['Red'].iloc[0])     
        blorange = str(inst['Blue'].iloc[0]) +':' + str(inst['Orange'].iloc[0]) 
        vyellow = str(inst['Violet'].iloc[0]) +':' + str(inst['Yellow'].iloc[0]) 
        compl = ', '.join(compl).lower() + ' (' + gred +  ';' + blorange + ';' + vyellow + ')'
        
    len_colors = inst['colors'].iloc[0]
    coldr = inst['cold'].iloc[0]
    warmr = inst['warm'].iloc[0]
    sat = inst['Saturated'].iloc[0]
    desat = inst['Desaturated'].iloc[0]
    satr = inst['sat-ratio'].iloc[0]
    desatr = inst['desat-ratio'].iloc[0]
    eln = 1 
    empty = []
    for i in [1,2,3,4,8,9]:
        contr = predicts.columns[i]
        if inst[contr].iloc[0] == 1:
            if eln%2 == 1: 
                if i == 1: 
                    pdf.multi_cell(0, -118, f'- {contr} ({len_colors} colors)')
                elif i == 3: 
                    pdf.multi_cell(0, -118, f'- {contr} (cold colors: {coldr}, warm colors: {warmr})')
                elif i == 4: 
                    pdf.multi_cell(0, -118, f'- {contr}: {compl}')
                elif i == 8: 
                    pdf.multi_cell(0, -118, f'- {contr} (sat: {sat}, desat: {desat})')
                elif i == 9: 
                    pdf.multi_cell(0, -118, f'- {contr} (sat/desat-ratio: {int(satr*100)}: {int(desatr*100)})')
                else: 
                    pdf.multi_cell(0, -118, f'- {contr}')
            if eln%2 == 0:
                if i == 1: 
                    pdf.multi_cell(0, 130, f'- {contr} ({len_colors} colors)')
                elif i == 3: 
                    pdf.multi_cell(0, 130, f'- {contr} (cold colors: {coldr}, warm colors: {warmr})')
                elif i == 4: 
                    pdf.multi_cell(0, 130, f'- {contr}: {compl}')
                elif i == 8: 
                    pdf.multi_cell(0, 130, f'- {contr} (sat: {sat}, desat: {desat})')
                elif i == 9: 
                    pdf.multi_cell(0, 130, f'- {contr} (sat-desat-ratio: {int(satr*100)}: {int(desatr*100)})')
                else: 
                    pdf.multi_cell(0, 130, f'- {contr}')
            eln += 1
            empty.append(contr) 
            
    if empty == []: 
        if eln%2 == 1:
            pdf.multi_cell(0, -118, f'None')
        if eln%2 == 0:
            pdf.multi_cell(0, 130, f'None')
        eln += 1
    
    if eln%2 == 0:
        pdf.set_font("Arial", 'B', size=12)
        pdf.multi_cell(0, 130, 'Colors: ') 
        pdf.set_font("Arial", size=12)
    if eln%2 == 1:
        pdf.set_font("Arial", 'B', size=12)
        pdf.multi_cell(0, -115, 'Colors: ')   
        pdf.set_font("Arial", size=12)
    eln += 1
    for j in range(10, 16):
        contr = predicts.columns[j]
        area = inst[contr].iloc[0] 
        if area != 0 : 
            if eln%2 == 1:
                pdf.multi_cell(0, -118, f'- {contr}: {area}')
            if eln%2 == 0:
                pdf.multi_cell(0, 128, f'- {contr}: {area}')
            eln += 1
            
    if eln%2 == 0:
        pdf.set_font("Arial", 'B', size=12)
        pdf.multi_cell(0, 130, 'Lumens: ') 
        pdf.set_font("Arial", size=12)
    if eln%2 == 1:
        pdf.set_font("Arial", 'B', size=12)
        pdf.multi_cell(0, -118, 'Lumens: ') 
        pdf.set_font("Arial", size=12)
    eln += 1           
    for k in range(16, 18):
        contr = predicts.columns[k]
        area = inst[contr].iloc[0] 
        if area != 0 : 
            if eln%2 == 1:
                pdf.multi_cell(0, -118, f'- {contr}: {area}')
            if eln%2 == 0:
                pdf.multi_cell(0, 128, f'- {contr}: {area}')
            eln += 1

    if eln%2 == 0:
        pdf.set_font("Arial", 'B', size=12)
        pdf.multi_cell(0, 130, 'Tone: ')  
        pdf.set_font("Arial", size=12)
    if eln%2 == 1:
        pdf.set_font("Arial", 'B', size=12)
        pdf.multi_cell(0, -118, 'Tone: ')  
        pdf.set_font("Arial", size=12)
    eln += 1               
    for l in range(18, 20):
        contr = predicts.columns[l]
        area = inst[contr].iloc[0] 
        if area != 0 : 
            if eln%2 == 1:
                pdf.multi_cell(0, -118, f'- {contr}: {area}')
            if eln%2 == 0:
                pdf.multi_cell(0, 128, f'- {contr}: {area}')
            eln += 1

os.chdir(IMG_PATH)            
pdf.output("screenshots_contrasts_knn21.pdf")

