# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 19:05:46 2020

@author: lsamsi
"""
# RGB and HSV Viewer: https://www.rapidtables.com/web/color/RGB_Color.html
# TODO: click on color and have RGB (or other) values in clipboard 

# import modules
import os
import matplotlib.pyplot as plt #2.1.2
import numpy as np
import cv2
import pandas as pd 


#%%

def show_palette(palette): 
    # convert default BGR to RGB 
    palette = cv2.cvtColor(palette, cv2.COLOR_BGR2RGB)
    plt.imshow(palette)
    plt.show()
    
    
#%%

def crop_image(CP_INDEX, palette): 
    # crop image 
    if CP_INDEX == 'tw1': 
        crop_img = palette[110:,]
    elif CP_INDEX == 'tw2':
        crop_img = palette[290:690,]
    elif CP_INDEX == 'tw3' :
        crop_img = palette[140:340,]    
    elif CP_INDEX == 'ta1'   :
        crop_img = palette[260:640,]  
    elif CP_INDEX == 'da1' or CP_INDEX == 'ts1' or CP_INDEX == 'bs1' or CP_INDEX == 'ls1' or CP_INDEX == 'ss1':
        crop_img = palette[210:540,]  
    elif CP_INDEX == 'dw1' or CP_INDEX == 'cw1'  or CP_INDEX == 'lsp1':
        crop_img = palette[250:630,] 
    elif CP_INDEX == 'sa1' or CP_INDEX == 'tsp1':
        crop_img = palette[75:230,] 
    
    crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)     
    plt.imshow(crop_img)
    plt.show()
    # #pop-up window 
    # cv2.imshow("cropped", crop_img)
    # cv2.waitKey(0)
    
    return crop_img

#%%

def get_raster(CP_INDEX):
    # build grid 
    rows = 7
    columns = 10
    
    # hint: if the image is cropped to a size of another season, set grid size to the same season in the conditional
    if CP_INDEX == 'tw1': 
        bordery = 15 # border around the color dataset matrix
        borderx = 35
        patchy = 150 # color patch size of each color in the color dataset
        patchx = 160
        delta = 20 # spacing in-between 
    elif CP_INDEX == 'tw2':
        bordery = 10 # border around the color dataset matrix
        borderx = 39
        patchy = 49 # color patch size of each color in the color dataset
        patchx = 50
        delta = 5 # spacing in-between 
    elif CP_INDEX == 'tw3':
        bordery = 9 # border around the color dataset matrix
        borderx = 20
        patchy = 24 # color patch size of each color in the color dataset
        patchx = 24
        delta = 2 # spacing in-between     
    elif CP_INDEX == 'ta1':
        bordery = 15 # border around the color dataset matrix
        borderx = 36
        patchy = 46 # color patch size of each color in the color dataset
        patchx = 46
        delta = 5 # spacing in-between  
    elif CP_INDEX == 'da1' or CP_INDEX == 'ts1' or CP_INDEX == 'bs1' or CP_INDEX == 'ls1' or CP_INDEX == 'ss1':
        bordery = 22 # border around the color dataset matrix
        borderx = 31
        patchy = 38 # color patch size of each color in the color dataset
        patchx = 38
        delta = 4 # spacing in-between  
    elif CP_INDEX == 'dw1' or CP_INDEX == 'cw1'  or CP_INDEX == 'lsp1':
        bordery = 27 # border around the color dataset matrix
        borderx = 37
        patchy = 44 # color patch size of each color in the color dataset
        patchx = 44
        delta = 7 # spacing in-between     
    elif CP_INDEX == 'sa1' or CP_INDEX == 'tsp1':
        bordery = 17 # border around the color dataset matrix
        borderx = 17
        patchy = 18 # color patch size of each color in the color dataset
        patchx = 19
        delta = 1 # spacing in-between  
        
    return bordery, borderx, patchy, patchx, delta

#%%   
# test values to determine grid params 
# img_grid = crop_img[:69,:80]
# plt.imshow(img_grid)
# plt.show()
# y - 28, 69, 76
# x - 38, 80, 87



#%%

def palette_grid(CP_INDEX, img =None, show_gridnodes = False):
    
    rows = 7
    columns = 10
    
    bordery, borderx, patchy, patchx, delta = get_raster(CP_INDEX)
    
    ini = bordery+patchy/2   
    nex = patchy/2+delta+patchy/2 
     # recursive function to determine center coordinate points for each color patch
    def T(n):
        if n == 1:
            return int(ini)
        else:
            return int(T(n-1)+nex)
    yy = [T(i) for i in range(1,rows+1)] # make sure rows is an int scalar, not a lst 
    # [90.0, 260.0, 430.0, 600.0, 770.0, 940.0, 1110.0]
    
    ini = borderx+patchx/2   
    nex = patchx/2+delta+patchx/2   
    xx = [T(i) for i in range(1,columns+1)]
    # [115.0, 295.0, 475.0, 655.0, 835.0, 1015.0, 1195.0, 1375.0, 1555.0, 1735.0]
    
    # blacken pixel value at grid nodes (=center of color patch to black dots)
    if show_gridnodes: 
        for i in yy: 
            for j in xx: 
                img[i,j,:] = [0,0,0]
        
        plt.imshow(img)
        plt.show()
    
    try: 
        print(img.shape)
    except: 
        pass
    return xx, yy 

#%%

def convert_color(col, conversion=cv2.COLOR_BGR2Lab): #default 
    color = cv2.cvtColor(np.array([[col]], dtype=np.float32), conversion)[0, 0]
    return color

def get_dataframe(CP_INDEX, img): 
    rows = []
    cols = []
    labs = []
       
    # access and save pixel value at grid nodes (=center of color patch)
    xx, yy = palette_grid(CP_INDEX)
    
    for row, i in enumerate(yy): 
        for column, j in enumerate(xx):  
            RGB = img[i,j]  
            R,G,B = RGB[0], RGB[1], RGB[2] 
            rgb = R/255, G/255, B/255 
            lab = convert_color(rgb, conversion=cv2.COLOR_RGB2Lab)
            #print('row:', row, 'column:', column, 'rgb:', color)
            rows.append(row)
            cols.append(column)
            labs.append(lab)
    
    df = pd.DataFrame()
    df['row'] = rows
    df['column'] = cols
    df['lab'] = labs
       
    # round rgb vals: 1-3 precision, 0 scale 
    df[['l', 'a', 'b']] = pd.DataFrame(df['lab'].tolist(), index=df.index)
    df['l'] = [round((x),2) for x in df['l']]
    df['a'] = [round((x),2) for x in df['a']]
    df['b'] = [round((x),2) for x in df['b']]
    df['lab'] = tuple(zip(df['l'],df['a'],df['b'] ))
    # df[['r', 'g', 'b']].head()
    
    #print(df.head())
    
    return df



#%%

def tweak_df(df, balken=False, silver=False, gold=False, goldup=False):
    # tweak dataframe: drop incorrect values 
    if balken: 
        df = df.iloc[:65]
    #df = df.drop(49)
    if gold: 
        df = df.drop(59)
    if goldup: 
        df = df.drop(49)
    if silver: 
        df = df.drop(58)
    return df

#%%

if __name__ == '__main__':

    # to specify
    PATH = r'D:\thesis\palettes'
    # define variables
    SEASONS = ['lsp', 'tsp', 'bs',
              'ls', 'ts', 'ss',
              'sa', 'ta', 'da', 
              'cw', 'tw', 'dw' ]
    BALKEN = True 
    SILVER = True
    GOLD = True
    GOLDUP = False
    index = 11 # iterates through all CPs 
    
    ################### Settings ########################
    CP_PATHS = []
    for season in SEASONS: 
        path = os.path.join(PATH, season)
        CP_PATHS.append(path)
    
    FILENAME = []
    for i in range(1):
        for season in SEASONS: 
            string = str(season) + str(i+1)
            FILENAME.append(string)

    
    # set directory 
    os.chdir(CP_PATHS[index])
    
    ################### Loading ########################
    # read image (palette)
    palette = cv2.imread(f'{FILENAME[index]}.jpg')
    # plot image
    show_palette(palette)

    
    ################### Processing ########################
    # check and adjust BALKEN, SILVER, GOLD, GOLDUP if necessary
    crop_img = crop_image(FILENAME[index], palette)
#%%    
    # check palette grid should be smaller than image shape 
    print(crop_img.shape) # image shape 
    print(palette_grid(FILENAME[index])) # center points on grid 
    # fit grid to image 
    crop_img_dots = crop_image(FILENAME[index], palette)
    palette_grid(FILENAME[index], img= crop_img_dots, show_gridnodes=True) 
#%%
    # get colors for grid on image 
    df = get_dataframe(FILENAME[index], crop_img)    
    # to inspect 
    df = tweak_df(df, balken=BALKEN, silver=SILVER, gold=GOLD, goldup=GOLDUP) 
    
    ################### Saving ########################
    # save pd Dataframe 
    os.chdir(r'D:\thesis\code\pd4cpInlab')
    df.to_csv(f'{FILENAME[index]}.csv', index=False)
    cv2.imwrite(f'{FILENAME[index]}.jpg', cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR))
    print('Color Palette Colors saved successfully.')


