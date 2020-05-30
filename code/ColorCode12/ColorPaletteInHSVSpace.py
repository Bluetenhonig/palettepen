# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 19:48:20 2020

@author: lsamsi
"""
# Requirements: RGB colors of hues (see ColorPaletteToRGB.py)

# Visualizing Color Palette's Hue Values of Color Palette in HSV Color Space

# RGB and HSV Viewer: https://www.rapidtables.com/web/color/RGB_Color.html
# TODO: click on color and have RGB (or other) values in clipboard 
# TODO: compart TW palette with other TW palettes and all other season's palettes   

# import modules
import os
import matplotlib.pyplot as plt #2.1.2
import numpy as np
import pandas as pd
#import cv2

# define variable
SEASONS = ['cw1', 'tw1', 'dw1', 
           'bs1', 'tsp1', 'lsp1', 
           'sa1', 'ta1', 'da1', 
           'ss1', 'ts1', 'ls1']


# define function
def get_colors(space, cp): 
    if space == 'HSV': 
        # get HSV colors for hues 
        os.chdir(r'D:\thesis\code\pd4cpInhsv')
        hsvdf = pd.read_csv(f'{cp}.csv')
        return hsvdf
    else: 
        # get RGB colors for hues 
        os.chdir(r'D:\thesis\code\pd4cpInrgb')
        rgbdf = pd.read_csv(f'{cp}.csv')
        return rgbdf 

def merge_spaces(df1, df2): 
    # merge data sets 
    df3 = pd.merge(df1, df2, left_on=['row', 'column'], right_on=['row', 'column'])
    df = df3.sort_values(by=['h'])
    # df[['row', 'column','hsv', 'rgb']].iloc[12:15]
    return df 

def get_pixelcolors(df, *lst): 
    # get RGB colors for dot face colorization 
    rgb = df['rgb'] # df[['h']]
    rgb = rgb.reset_index(drop=True)
    
    collst = []
    for i in range(len(rgb)): 
        test_str = rgb[i][1:-1]
        res = list(map(int, test_str.split(', ')))
        collst.append(res)
     
    from matplotlib import colors
    pixel_colors = np.asarray(collst, dtype=np.uint8)
    norm = colors.Normalize(vmin=-1.,vmax=1.)
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()
    if lst: 
        return pixel_colors
    else: 
        df['pixel'] = pixel_colors
        return df 


#%%
# build dictionary where key is the color palette and the value is the dataframe of all color values
cp_dfs = dict()

for season in SEASONS: 
    hsvdf = get_colors('HSV', season)
    rgbdf = get_colors('RGB', season)
    df = merge_spaces(hsvdf, rgbdf)
    cp_dfs[season] = df 

#%%
# build pandas dataframe 
cp_pd = []

for season in SEASONS: 
    hsvdf = get_colors('HSV', season)
    rgbdf = get_colors('RGB', season)
    df = merge_spaces(hsvdf, rgbdf)
    df['season'] = season
    get_pixelcolors(df)
    cp_pd.append(df)

cp_pd = pd.concat(cp_pd)


#%%
# test
#cp_dfs['tw1']['h']
#cp_dfs['cw1']['h']
#cp_dfs['dw1']['h']
#cp_dfs['ls1']['h']


#%%
############################
### Exploratory Analysis ###
############################

cp_pd.info()
cp_pd.describe()

# ranking seasons 

# saturation: 0 - soft/desat; 100 - bright/saturated  
cp_pd.groupby(['season'])['s'].max().sort_values(ascending=False)
cp_pd.groupby(['season'])['s'].min().sort_values(ascending=False)

# value: 0 - dark; 100 - light
cp_pd.groupby(['season'])['v'].max().sort_values(ascending=False)
cp_pd.groupby(['season'])['v'].min().sort_values(ascending=False)



cp_pd.groupby(['season'])['s'].value_counts().sort_values(ascending=False)
cp_pd.groupby(['season'])['v'].value_counts().sort_values(ascending=False)



#%%
######################
### VISUALIZATIONS ###
######################


### HUE ###


# saving directory
os.chdir(r'D:\thesis\code\pd4cpInhsv\analysis')

# bar plot 
# df['h'].sort_values().plot(kind='bar')


# scatter plot 
fig = plt.figure(figsize=(15,20))
for i, cp in enumerate(cp_dfs):     
    fig = plt.subplot(4, 3, i+1)
    # get Hue-channel from HSV colors 
    hue = cp_dfs[cp]['h'].sort_values()
    hue = hue.reset_index(drop=True)
    hue = hue.reset_index()
    
    pixel_colors = get_pixelcolors(cp_dfs[cp], 'y')
    plt.scatter(hue.index, hue.h, facecolors=pixel_colors, marker=".")
    plt.ylabel('Hue  [0-360°]')
    plt.xlabel(f'CP color')
    # TODO: diagonal axis 
    #ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
    plt.tight_layout()
    plt.title(f'CP: {cp} \nRGB: facecolors \nHSV: hue-channel ', loc='left')

plt.savefig('seasons_hue_scatterplot.png')
plt.show()

# line plot 
#plt.plot(hue.index, hue.h)
#plt.show()

# color bar 
# TODO: install once you have internet 
# get HSV colormap  on axis colorbar basics 
# import matplotlib as mpl
# from matplotlib import cm 
# from colorspacious import cspace_converter  
# from collections import OrderedDict
#
# cmaps = OrderedDict()
#
# cmaps['Cyclic'] = ['hsv']
#
# nrows = max(len(cmap_list) for cmap_category, cmap_list in cmaps.items())
# gradient = np.linspace(0, 1, 256)
# gradient = np.vstack((gradient, gradient))
#
#
# def plot_color_gradients(cmap_category, cmap_list, nrows):
#     fig, axes = plt.subplots(nrows=nrows)
#     fig.subplots_adjust(top=0.95, bottom=0.01, left=0.2, right=0.99)
#     axes[0].set_title(cmap_category + ' colormaps', fontsize=14)
#
#     for ax, name in zip(axes, cmap_list):
#         ax.imshow(gradient, aspect='auto', cmap=plt.get_cmap(name))
#         pos = list(ax.get_position().bounds)
#         x_text = pos[0] - 0.01
#         y_text = pos[1] + pos[3]/2.
#         fig.text(x_text, y_text, name, va='center', ha='right', fontsize=10)
#
#     # Turn off *all* ticks & spines, not just the ones with colormaps.
#     for ax in axes:
#         ax.set_axis_off()
#
#
# for cmap_category, cmap_list in cmaps.items():
#     plot_color_gradients(cmap_category, cmap_list, nrows)
#
# plt.show()

#%%

## polar plot 
#import numpy as np
#import pandas as pd
#import seaborn as sns
#
#sns.set()
#
## Generate an example radial datast
#r = np.linspace(0, 10, num=100)
#df = pd.DataFrame({'r': r, 'slow': r, 'medium': 2 * r, 'fast': 4 * r})
#
## Convert the dataframe to long-form or "tidy" format
#df = pd.melt(df, id_vars=['r'], var_name='speed', value_name='hue')
#
## Set up a grid of axes with a polar projection
#g = sns.FacetGrid(df, col="speed", hue="speed",
#                  subplot_kws=dict(projection='polar'), height=4.5,
#                  sharex=False, sharey=False, despine=False)
#
## Draw a scatterplot onto each axes in the grid
#g.map(sns.scatterplot, "theta", "r")



#%%

# plot swarmplot 

import seaborn as sns
import matplotlib.pyplot as plt

os.chdir(r'D:\thesis\code\pd4cpInhsv\analysis')

sns.set(style="ticks")

# Initialize the figure with a logarithmic x axis
f, ax = plt.subplots(figsize=(25, 9))

# Add in points to show each observation
sns.swarmplot(x="h", y="season", data=cp_pd,
              size=6, hue="h", 
              #color=".5",  #transparency value 
              linewidth=0)

# Tweak the visual presentation
ax.xaxis.grid(True)
ax.set(ylabel="Season", xlabel="Hue (HSV) in Angles")
ax.get_legend().set_visible(False)
sns.despine(trim=False, left=True)

f.savefig("seasons_hue_swarmplot.png")


#%%

### Saturation ###


# plot swarmplot 

import seaborn as sns
import matplotlib.pyplot as plt

os.chdir(r'D:\thesis\code\pd4cpInhsv\analysis')

sns.set(style="ticks")

# Initialize the figure with a logarithmic x axis
f, ax = plt.subplots(figsize=(25, 9))

# Add in points to show each observation
sns.swarmplot(x="s", y="season", data=cp_pd,
              size=6, 
              #hue="black", 
              color=".5",  #transparency value 
              linewidth=0)

# Tweak the visual presentation
ax.xaxis.grid(True)
ax.set(ylabel="Season", xlabel="Saturation (HSV)")
ax.get_legend().set_visible(False)
sns.despine(trim=False, left=True)

f.savefig("seasons_saturation_swarmplot.png")


#%%

### Value ###

# plot swarmplot 

import seaborn as sns
import matplotlib.pyplot as plt

os.chdir(r'D:\thesis\code\pd4cpInhsv\analysis')

sns.set(style="ticks")

# Initialize the figure with a logarithmic x axis
f, ax = plt.subplots(figsize=(25, 9))

# Add in points to show each observation
sns.swarmplot(x="v", y="season", data=cp_pd,
              size=6, 
              #hue="black", 
              color=".5",  #transparency value 
              linewidth=0)

# Tweak the visual presentation
ax.xaxis.grid(True)
ax.set(ylabel="Season", xlabel="Value (HSV)")
ax.get_legend().set_visible(False)
sns.despine(trim=False, left=True)

f.savefig("seasons_value_swarmplot.png")