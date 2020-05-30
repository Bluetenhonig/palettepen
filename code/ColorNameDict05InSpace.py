# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 15:52:37 2020

@author: Linda Samsinger

All 28 VIAN colors were looked up in the Color Thesaurus dictionary of 
color name-rgb/lab value mappings of Lindner (EPFL). The lookup of colors
is discretized. 
The dataset was extended from rgb/lab values to include also hsv/hsl and 
lch values. All 28 VIAN color values were plotted in each color space. 

"""

# import modules
import os
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

### Color-Thesaurus EPFL ###

PATH = r'D:\thesis\code\pd28vianhues'
FILE = 'SRGBLABhsvhslLCHHEX_EngAll_VIANHuesColorThesaurus.xlsx'

# set directory 
os.chdir(PATH)

# load data 
data = pd.read_excel(FILE, sep=" ", index_col=0)

#%%

# analyze data
data.head()
data.info()

# counts per color 
data['english name'].nunique()
data['VIAN_color_category'].value_counts()

#%%

#################
### RGB SPACE ###
#################

# show all the RGB values in 3D 

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors

#original CS: RGB 
r = np.array(data['srgb_R'])
g = np.array(data['srgb_G'])
b = np.array(data['srgb_B'])
g.shape
#  (298, 198)

p = [eval(l) for l in data['srgb'].tolist()]
p = np.array(p)


pixel_colors = p
norm = colors.Normalize(vmin=-1.,vmax=1.)  #quench it into -1, 1 interval
norm.autoscale(pixel_colors)
pixel_colors = norm(pixel_colors).tolist()

fig = plt.figure(figsize=(8,8))
axis = fig.add_subplot(1, 1, 1, projection="3d")


axis.scatter(r, g, b, facecolors=pixel_colors, marker=".")
axis.set_xlabel("Red") 
axis.set_ylabel("Green") 
axis.set_zlabel("Blue")  

plt.title(f"Color Thesaurus VIAN colors in RGB Space",fontsize=20, y=1.05)

os.chdir(r'D:\thesis\images')
plt.savefig('RGB_Space_VIAN_Color_Thesaurus.jpg')

plt.show()


#%%

# filter the data
vian_hues = [
        'blue'
        , 'cyan'
        , 'green'
        , 'magenta'
        , 'orange'
        , 'pink'
        , 'red'
        , 'yellow'
        , 'beige'
        , 'black'
        , 'brown'
        , 'copper'
        , 'cream'
        , 'gold'
        , 'grey'
        , 'purple'
        , 'rust'
        , 'silver'
        , 'white'
        , 'amber'
        , 'lavender'
        , 'sepia'
        , 'apricot'
        , 'bronze'
        , 'coral'
        , 'peach'
        , 'ultramarine'
        , 'mustard'
        ]

# to specify 
COLOR_FILTER = 'lavender' 

# filter the dataset based on color
subdata = data[data['english name'] == COLOR_FILTER]


#%%

# Visualizing a subset of RGB Colors in RGB-Space

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors

#original CS: RGB 
r = np.array(subdata['srgb_R'])
g = np.array(subdata['srgb_G'])
b = np.array(subdata['srgb_B'])
g.shape
#  (298, 198)

p = [eval(l) for l in subdata['srgb'].tolist()]
p = np.array(p)


pixel_colors = p
norm = colors.Normalize(vmin=-1.,vmax=1.)  #quench it into -1, 1 interval
norm.autoscale(pixel_colors)
pixel_colors = norm(pixel_colors).tolist()


fig = plt.figure(figsize=(8,8))
axis = fig.add_subplot(1, 1, 1, projection="3d")

for i in range(len(r)): #plot each point + it's index as text above
    axis.scatter(r[i],g[i],b[i], facecolors=pixel_colors[i], marker=".", s=100) 
    axis.text(r[i],g[i],b[i],  '%s' % (str(i)), size=10, zorder=1, color='k') 
 
#axis.scatter(r, g, b, facecolors=pixel_colors, marker=".")
axis.set_xlabel("Red") 
axis.set_ylabel("Green") 
axis.set_zlabel("Blue")  
 
plt.title(f"Color Thesaurus VIAN colors in RGB Space: {COLOR_FILTER.upper()}",fontsize=20, y=1.05)
plt.show()


#%%
#################
### LAB SPACE ###
#################

# Visualizing all LAB Colors in LAB-Space

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors

#original CS: LAB 
l = np.array(data['cielab_L'])
a = np.array(data['cielab_a'])
b = np.array(data['cielab_b'])
l.shape
#  (298, 198)
b.flatten().max()

# facecolors in RGB 
p = [eval(l) for l in data['srgb'].tolist()]
p = np.array(p)

pixel_colors = p
norm = colors.Normalize(vmin=-1.,vmax=1.)  #quench it into -1, 1 interval
norm.autoscale(pixel_colors)
pixel_colors = norm(pixel_colors).tolist()


# plot figure
fig = plt.figure(figsize=(8,8))
axis = fig.add_subplot(1, 1, 1, projection="3d")

# plot dots 
axis.scatter(a, b, l, facecolors=pixel_colors, marker=".")
axis.set_xlabel("a*: green-red") 
axis.set_ylabel("b*: blue-yellow") 
axis.set_zlabel("Luminance")  

axis.plot([0,0], [b.flatten().min(), b.flatten().max()], zs=0, color='red', linestyle='dashed', linewidth=2, markersize=12)
axis.plot([a.flatten().min(), a.flatten().max()], [0, 0], zs=0, color='red', linestyle='dashed', linewidth=2, markersize=12)
axis.plot([0, 0], [0, 0], zs=[0,l.flatten().max()], color='red', linestyle='dashed', linewidth=2, markersize=12)

plt.title(f"Color Thesaurus VIAN colors in L*ab Space",fontsize=20, y=1.05)

os.chdir(r'D:\thesis\images')
plt.savefig('LAB_Space_VIAN_Color_Thesaurus.jpg')

plt.show()


#Why is 0,0 not on coordinate axis? because z's 0 is not at the bottom

#%%

# Calculate 28 averages for lab values for each VIAN color category
 
import cv2
def convert_color(col, conversion=cv2.COLOR_BGR2Lab): #default 
    """" converts BGR to LAB by default supports all color spaces except lch 
    """
    color = cv2.cvtColor(np.array([[col]], dtype=np.float32), conversion)[0, 0]
    return color


pdf = data['VIAN_color_category'].value_counts()
cats = pdf.index

catdict = {}
mean_labs = []
for cat in cats:
    color = data['cielab'][data['VIAN_color_category'] == cat]
    lst_ar = []
    for c in range(len(color)): 
        ar = np.array(eval(color.iloc[c]))
        lst_ar.append(ar)
    mean_lab = np.mean(lst_ar, axis=0).tolist()
    mean_labs.append(mean_lab)
    catdict[cat] = mean_lab

# plot lab averages

#original CS: LAB 
l = np.array([val[0] for val in catdict.values()])
a = np.array([val[1] for val in catdict.values()])
b = np.array([val[2] for val in catdict.values()])
l.shape
#  (298, 198)
b.flatten().max()

# facecolors in RGB 
p = []
for col in catdict.values():
    rgb = convert_color(col, cv2.COLOR_Lab2RGB).tolist()
    p.append(rgb)

p = np.array(p)

pixel_colors = p
norm = colors.Normalize(vmin=-1.,vmax=1.)  #quench it into -1, 1 interval
norm.autoscale(pixel_colors)
pixel_colors = norm(pixel_colors).tolist()

# plot figure
fig = plt.figure(figsize=(10,10))
axis = fig.add_subplot(1, 1, 1, projection="3d")

# plot dots 
axis.scatter(a, b, l, facecolors=pixel_colors, marker=".", s=150)
axis.set_xlabel("a*: green-red") 
axis.set_ylabel("b*: blue-yellow") 
axis.set_zlabel("Luminance")  

TEXT_SPACE = 1
labels = [] 
handles = []
# make labels 
for i in range(len(l)): #plot each point + it's index as text above
     handle = axis.scatter(a[i], b[i], l[i], facecolors=pixel_colors[i], marker=".", s=150)
     # annotate dots 
     axis.text(a[i]+TEXT_SPACE,b[i]+TEXT_SPACE,l[i]+TEXT_SPACE,  '%s' % (str(cats[i])), size=10, zorder=1, color='k') 
     labels.append(str(cats[i]))
     handles.append(handle)

# plot red lines
axis.plot([0,0], [b.flatten().min(), b.flatten().max()], zs=0, color='red', linestyle='dashed', linewidth=2, markersize=12)
axis.plot([a.flatten().min(), a.flatten().max()], [0, 0], zs=0, color='red', linestyle='dashed', linewidth=2, markersize=12)
axis.plot([0, 0], [0, 0], zs=[0,l.flatten().max()], color='red', linestyle='dashed', linewidth=2, markersize=12)
# set axis limits 
#axis.set_zlim([0,1])
 # make legend
axis.legend(handles=tuple(handles), labels=tuple(labels), loc='best', title="VIAN colors \n(by frequency)", bbox_to_anchor=(1.3,.95))
# make title 
plt.suptitle(f"Color Thesaurus Class Center Averages VIAN colors in L*ab Space",fontsize=20, y=.9) #, title="title"
# save plot
os.chdir(r'D:\thesis\images')
plt.savefig('LAB_Space_AVG_VIAN_Color_Thesaurus.jpg')
axis.view_init(60, 150)

plt.show()

#%%

# filter the data
vian_hues = [
        'blue'
        , 'cyan'
        , 'green'
        , 'magenta'
        , 'orange'
        , 'pink'
        , 'red'
        , 'yellow'
        , 'beige'
        , 'black'
        , 'brown'
        , 'copper'
        , 'cream'
        , 'gold'
        , 'grey'
        , 'purple'
        , 'rust'
        , 'silver'
        , 'white'
        , 'amber'
        , 'lavender'
        , 'sepia'
        , 'apricot'
        , 'bronze'
        , 'coral'
        , 'peach'
        , 'ultramarine'
        , 'mustard'
        ]

# to specify 
COLOR_FILTER = 'blue' 

# filter the dataset based on color
subdata = data[data['VIAN_color_category'] == COLOR_FILTER]


#%%


# Visualizing filtered LAB Colors in LAB-Space

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors

#original CS: LAB 
l = np.array(subdata['cielab_L'])
a = np.array(subdata['cielab_a'])
b = np.array(subdata['cielab_b'])
l.shape
#  (298, 198)

# facecolors in RGB 
p = [eval(l) for l in subdata['srgb'].tolist()]
p = np.array(p)

pixel_colors = p
norm = colors.Normalize(vmin=-1.,vmax=1.)  #quench it into -1, 1 interval
norm.autoscale(pixel_colors)
pixel_colors = norm(pixel_colors).tolist()


# plot figure
fig = plt.figure(figsize=(8,8))
axis = fig.add_subplot(1, 1, 1, projection="3d")
# plot points
axis.scatter(a, b, l, facecolors=pixel_colors, marker=".")
# name axis 
axis.set_xlabel("a*: green-red") 
axis.set_ylabel("b*: blue-yellow") 
axis.set_zlabel("Luminance")  

axis.plot([0,0], [b.flatten().min(), b.flatten().max()], zs=0, color='red', linestyle='dashed', linewidth=2, markersize=12)
axis.plot([a.flatten().min(), a.flatten().max()], [0, 0], zs=0, color='red', linestyle='dashed', linewidth=2, markersize=12)
axis.plot([0, 0], [0, 0], zs=[0,l.flatten().max()], color='red', linestyle='dashed', linewidth=2, markersize=12)

plt.title(f"Color Thesaurus VIAN colors in L*ab Space: {COLOR_FILTER}",fontsize=20, y=1.05)

plt.show()


#%%
#################
### LCH SPACE ###
#################

# will be the same plot as in the LAB space, just with different parameter names 
# lab und lch plots are the same

#%%

#################
### HSV SPACE ###
#################

# polar plot in 2d 

import math
import numpy as np
import matplotlib.pyplot as plt
# polar scatter plot all hsv values in 2D  

# 
catdict = {}
mean_hsvs = []
for cat in cats:
    color = data['hsv'][data['VIAN_color_category'] == cat]
    lst_ar = []
    for c in range(len(color)): 
        ar = np.array(eval(color.iloc[c]))
        lst_ar.append(ar)
    mean_hsv = np.mean(lst_ar, axis=0).tolist()
    mean_hsvs.append(mean_hsv)
    catdict[cat] = mean_hsv
 
# Compute areas and colors
# make radius = saturation
r = np.array([hsv[1] for hsv in mean_hsvs] )  # radius 
# make angles: degrees to radians = hue
theta = np.array([math.radians(l) for l in [hsv[0] for hsv in mean_hsvs] ]) # angle 
# make size of the data points
area = np.full(len(data), 10) 
colors = theta

# plot figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='polar')
c = ax.scatter(theta, r, c=colors, s=area, cmap='hsv', alpha=0.75)
plt.title(f"Color Thesaurus AVG VIAN colors in HSV 2D",fontsize=20, y=1.1)

# save figure 
os.chdir(r'D:\thesis\images')
plt.savefig('HSV_PolarPlot_AVG_VIAN_Color_Thesaurus.jpg')

#%%
# polar plot in 3d 

# mapping from radians to cartesian (based on function lch to lab) 
def hsvdeg2hsvcart(hsv, h_as_degree = True):
    """
    convert hsv in polar view to cartesian view
    """
    if not isinstance(hsv, np.ndarray):
        hsv = np.array(hsv, dtype=np.float32)
    hsv_cart = np.zeros_like(hsv, dtype=np.float32)
    hsv_cart[..., 2] = hsv[..., 2]
    if h_as_degree:
        hsv[..., 0] = hsv[..., 0] *np.pi / 180
    hsv_cart[..., 0] = hsv[..., 1]*np.cos(hsv[..., 0])
    hsv_cart[..., 1] = hsv[..., 1]*np.sin(hsv[..., 0])
    return hsv_cart  

hsv_cart = hsvdeg2hsvcart([eval(l) for l in data['hsv']]).tolist()
data['hsv_cart'] = hsv_cart
data['hsv_cart_H'] = [i[0] for i in hsv_cart]
data['hsv_cart_S'] = [i[1] for i in hsv_cart]
data['hsv_cart_V'] = [i[2] for i in hsv_cart]

#%%
# Visualizing all HSV Colors in HSV-Space

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors

#original CS: HSV 
h = np.array(data['hsv_cart_H'])
s = np.array(data['hsv_cart_S'])
v = np.array(data['hsv_cart_V'])
h.shape
#  (298, 198)
s.flatten().min()

# facecolors in RGB 
p = [eval(l) for l in data['srgb'].tolist()]
p = np.array(p)

pixel_colors = p
norm = colors.Normalize(vmin=-1.,vmax=1.)  #quench it into -1, 1 interval
norm.autoscale(pixel_colors)
pixel_colors = norm(pixel_colors).tolist()


# plot figure
fig = plt.figure(figsize=(8,8))
axis = fig.add_subplot(1, 1, 1, projection="3d")
# plot dots 
axis.scatter(h, s, v, facecolors=pixel_colors, marker=".")
# label axis 
axis.set_xlabel("Hue") 
axis.set_ylabel("Saturation") 
axis.set_zlabel("Value")  
# set axis limits 
axis.set_zlim([0,1])
# draw in red lines
axis.plot([0,0], [s.flatten().min(), s.flatten().max()], zs=0, color='red', linestyle='dashed', linewidth=2, markersize=12)
axis.plot([h.flatten().min(), h.flatten().max()], [0, 0], zs=0, color='red', linestyle='dashed', linewidth=2, markersize=12)
axis.plot([0, 0], [0, 0], zs=[0,v.flatten().max()], color='red', linestyle='dashed', linewidth=2, markersize=12)
# make title
plt.title(f"Color Thesaurus VIAN colors in HSV Space",fontsize=20, y=1.05)

# save figure 
os.chdir(r'D:\thesis\images')
plt.savefig('HSV_Space_VIAN_Color_Thesaurus.jpg')

plt.show()

#%%

# filter the data
vian_hues = [
        'blue'
        , 'cyan'
        , 'green'
        , 'magenta'
        , 'orange'
        , 'pink'
        , 'red'
        , 'yellow'
        , 'beige'
        , 'black'
        , 'brown'
        , 'copper'
        , 'cream'
        , 'gold'
        , 'grey'
        , 'purple'
        , 'rust'
        , 'silver'
        , 'white'
        , 'amber'
        , 'lavender'
        , 'sepia'
        , 'apricot'
        , 'bronze'
        , 'coral'
        , 'peach'
        , 'ultramarine'
        , 'mustard'
        ]

# to specify 
COLOR_FILTER = 'blue' 

# filter the dataset based on color
subdata = data[data['VIAN_color_category'] == COLOR_FILTER]

#%%
# Visualizing a subset HSV Colors in HSV-Space

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors

#original CS: HSV 
h = np.array(subdata['hsv_cart_H'])
s = np.array(subdata['hsv_cart_S'])
v = np.array(subdata['hsv_cart_V'])
h.shape
#  (298, 198)
s.flatten().min()

# facecolors in RGB 
p = [eval(l) for l in subdata['srgb'].tolist()]
p = np.array(p)

pixel_colors = p
norm = colors.Normalize(vmin=-1.,vmax=1.)  #quench it into -1, 1 interval
norm.autoscale(pixel_colors)
pixel_colors = norm(pixel_colors).tolist()


# plot figure
fig = plt.figure(figsize=(8,8))
axis = fig.add_subplot(1, 1, 1, projection="3d")
# plot dots 
axis.scatter(h, s, v, facecolors=pixel_colors, marker=".")
# label axis 
axis.set_xlabel("Hue") 
axis.set_ylabel("Saturation") 
axis.set_zlabel("Value")  
# set axis limits 
axis.set_zlim([0,1])
# draw in red lines
axis.plot([0,0], [s.flatten().min(), s.flatten().max()], zs=0, color='red', linestyle='dashed', linewidth=2, markersize=12)
axis.plot([h.flatten().min(), h.flatten().max()], [0, 0], zs=0, color='red', linestyle='dashed', linewidth=2, markersize=12)
axis.plot([0, 0], [0, 0], zs=[0,v.flatten().max()], color='red', linestyle='dashed', linewidth=2, markersize=12)
# make title
plt.title(f"Color Thesaurus VIAN colors in HSV Space: {COLOR_FILTER}",fontsize=20, y=1.05)

plt.show()

#%%

#for angle in np.linspace(0, 360, 60).tolist():
#    # Plot stuff.
#    fig = plt.figure()
#    #...
#    
#    # rotate around the plot 
#    ax.view_init(60, angle)
#    plt.draw()
#    plt.pause(.001)
#    plt.show()
    
