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
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
import math
import cv2
import sys
sys.path.append(r'')
from ColorConversion00 import hsvdeg2hsvcart

# USER SPECIFICATION 
COLOR_FILTER = 'lavender' 
COLOR_FILTER2 = 'blue' 


# declare variables
PATH = r'D:\thesis\input_color_name_dictionaries\system_VIAN'
FILE = 'SRGBLABhsvhslLCHHEX_Eng_VIANHuesColorThesaurus.xlsx'


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

#%%
def convert_color(col, conversion=cv2.COLOR_BGR2Lab): #default 
    """" converts BGR to LAB by default supports all color spaces except lch 
    """
    color = cv2.cvtColor(np.array([[col]], dtype=np.float32), conversion)[0, 0]
    return color

def load_data(path, file): 
    os.chdir(path)
    data = pd.read_excel(file, sep=" ", index_col=0)
    data = data.dropna()
    data.head()
    data.info()
    return data
    
def get_counts(data): 
    """ counts per color """
    data['name'].nunique()
    data['cat1'].value_counts()
    return data

def thesaurusvian_in_rgbspace(data, save=False): 
    """ show all the RGB values of Thesaurus-VIAN in 3D """
    r = np.array(data['srgb_R'])
    g = np.array(data['srgb_G'])
    b = np.array(data['srgb_B'])
    
    p = [eval(l) for l in data['srgb'].tolist()]
    p = np.array(p)
        
    pixel_colors = p
    norm = colors.Normalize(vmin=-1.,vmax=1.) 
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()
    
    fig = plt.figure(figsize=(8,8))
    axis = fig.add_subplot(1, 1, 1, projection="3d")
        
    axis.scatter(r, g, b, facecolors=pixel_colors, marker=".")
    axis.set_xlabel("Red") 
    axis.set_ylabel("Green") 
    axis.set_zlabel("Blue")  
    
    plt.title(f"Color Thesaurus VIAN colors in RGB Space", fontsize=20, y=1.05)
    
    if save: 
        os.chdir(r'D:\thesis\output_images')
        plt.savefig('RGB_Space_VIAN_Color_Thesaurus.jpg')   
    plt.show()


def show_rgbsubset_in_rgbspace(data, color_filter): 
    subdata = data[data['cat2'] == COLOR_FILTER]   
    r = np.array(subdata['srgb_R'])
    g = np.array(subdata['srgb_G'])
    b = np.array(subdata['srgb_B'])
    
    p = [eval(l) for l in subdata['srgb'].tolist()]
    p = np.array(p)
      
    pixel_colors = p
    norm = colors.Normalize(vmin=-1.,vmax=1.)
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()
        
    fig = plt.figure(figsize=(8,8))
    axis = fig.add_subplot(1, 1, 1, projection="3d")
    
    for i in range(len(r)):
        axis.scatter(r[i],g[i],b[i], facecolors=pixel_colors[i], marker=".", s=100) 
        axis.text(r[i],g[i],b[i],  '%s' % (str(i)), size=10, zorder=1, color='k') 
     
    #axis.scatter(r, g, b, facecolors=pixel_colors, marker=".")
    axis.set_xlabel("Red") 
    axis.set_ylabel("Green") 
    axis.set_zlabel("Blue")  
     
    plt.title(f"Color Thesaurus VIAN colors in RGB Space: {color_filter.upper()}",fontsize=20, y=1.05)
    plt.show()

def show_labset_in_labspace(data, save=False): 
    l = np.array(data['cielab_L'])
    a = np.array(data['cielab_a'])
    b = np.array(data['cielab_b'])
    b.flatten().max()
     
    p = [eval(l) for l in data['srgb'].tolist()]
    p = np.array(p)
    
    pixel_colors = p
    norm = colors.Normalize(vmin=-1.,vmax=1.) 
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()
    
    fig = plt.figure(figsize=(8,8))
    axis = fig.add_subplot(1, 1, 1, projection="3d")
    
    axis.scatter(a, b, l, facecolors=pixel_colors, marker=".")
    axis.set_xlabel("a*: green-red") 
    axis.set_ylabel("b*: blue-yellow") 
    axis.set_zlabel("Luminance")  
    
    axis.plot([0,0], [b.flatten().min(), b.flatten().max()], zs=0, color='red', linestyle='dashed', linewidth=2, markersize=12)
    axis.plot([a.flatten().min(), a.flatten().max()], [0, 0], zs=0, color='red', linestyle='dashed', linewidth=2, markersize=12)
    axis.plot([0, 0], [0, 0], zs=[0,l.flatten().max()], color='red', linestyle='dashed', linewidth=2, markersize=12)
    
    plt.title(f"Color Thesaurus VIAN colors in L*ab Space",fontsize=20, y=1.05) 
    if save: 
        os.chdir(r'D:\thesis\output_images')
        plt.savefig('LAB_Space_VIAN_Color_Thesaurus.jpg')
    
    plt.show()


def vian_compute_label_avg(data): 
    pdf = data['cat1'].value_counts()
    cats = pdf.index    
    catdict = {}
    mean_labs = []
    for cat in cats:
        color = data['cielab'][data['cat1'] == cat]
        lst_ar = []
        for c in range(len(color)): 
            ar = np.array(eval(color.iloc[c]))
            lst_ar.append(ar)
        mean_lab = np.mean(lst_ar, axis=0).tolist()
        mean_labs.append(mean_lab)
        catdict[cat] = mean_lab    
    return catdict, cats


def plot_lab_avg(catdict, save=False): 
    l = np.array([val[0] for val in catdict.values()])
    a = np.array([val[1] for val in catdict.values()])
    b = np.array([val[2] for val in catdict.values()])
    b.flatten().max()
    
    p = []
    for col in catdict.values():
        rgb = convert_color(col, cv2.COLOR_Lab2RGB).tolist()
        p.append(rgb)
    
    p = np.array(p)
    
    pixel_colors = p
    norm = colors.Normalize(vmin=-1.,vmax=1.) 
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()
    
    fig = plt.figure(figsize=(10,10))
    axis = fig.add_subplot(1, 1, 1, projection="3d")
    
    axis.scatter(a, b, l, facecolors=pixel_colors, marker=".", s=150)
    axis.set_xlabel("a*: green-red") 
    axis.set_ylabel("b*: blue-yellow") 
    axis.set_zlabel("Luminance")  
    
    TEXT_SPACE = 1
    labels = [] 
    handles = []
    
    for i in range(len(l)):
         handle = axis.scatter(a[i], b[i], l[i], facecolors=pixel_colors[i], marker=".", s=150)
         axis.text(a[i]+TEXT_SPACE,b[i]+TEXT_SPACE,l[i]+TEXT_SPACE,  '%s' % (str(cats[i])), size=14, zorder=1, color='k') 
         labels.append(str(cats[i]))
         handles.append(handle)
    
    axis.plot([0,0], [b.flatten().min(), b.flatten().max()], zs=0, color='red', linestyle='dashed', linewidth=2, markersize=14)
    axis.plot([a.flatten().min(), a.flatten().max()], [0, 0], zs=0, color='red', linestyle='dashed', linewidth=2, markersize=14)
    axis.plot([0, 0], [0, 0], zs=[0,l.flatten().max()], color='red', linestyle='dashed', linewidth=2, markersize=14)
    lg = axis.legend(handles=tuple(handles), labels=tuple(labels), loc='best', title="VIAN colors \n(by frequency)", bbox_to_anchor=(1.3,.95), fontsize=14)
    title = lg.get_title()
    title.set_fontsize(16) 
    #plt.suptitle(f"Color Thesaurus Class Center Averages VIAN colors in L*ab Space",fontsize=20, y=.9) #, title="title"
    if save: 
        os.chdir(r'D:\thesis\output_images')
        plt.savefig('LAB_Space_AVG_VIAN_Color_Thesaurus.jpg')
    axis.view_init(60, 150)
    plt.show()

def show_labsubset_in_labspace(data, COLOR_FILTER2): 
    """ lab und lch plots are the same """
    subdata = data[data['cat1'] == COLOR_FILTER2]
    l = np.array(subdata['cielab_L'])
    a = np.array(subdata['cielab_a'])
    b = np.array(subdata['cielab_b'])
    p = [eval(l) for l in subdata['srgb'].tolist()]
    p = np.array(p)
    
    pixel_colors = p
    norm = colors.Normalize(vmin=-1.,vmax=1.) 
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()
    
    fig = plt.figure(figsize=(8,8))
    axis = fig.add_subplot(1, 1, 1, projection="3d")
    axis.scatter(a, b, l, facecolors=pixel_colors, marker=".")
    axis.set_xlabel("a*: green-red") 
    axis.set_ylabel("b*: blue-yellow") 
    axis.set_zlabel("Luminance")  
    axis.plot([0,0], [b.flatten().min(), b.flatten().max()], zs=0, color='red', linestyle='dashed', linewidth=2, markersize=12)
    axis.plot([a.flatten().min(), a.flatten().max()], [0, 0], zs=0, color='red', linestyle='dashed', linewidth=2, markersize=12)
    axis.plot([0, 0], [0, 0], zs=[0,l.flatten().max()], color='red', linestyle='dashed', linewidth=2, markersize=12)
    plt.title(f"Color Thesaurus VIAN colors in L*ab Space: {COLOR_FILTER}",fontsize=20, y=1.05)
    plt.show()

def plot_2d_polar_scatterplot_hsv(cats, save=False): 
    catdict = {}
    mean_hsvs = []
    for cat in cats:
        color = data['hsv'][data['cat1'] == cat]
        lst_ar = []
        for c in range(len(color)): 
            ar = np.array(eval(color.iloc[c]))
            lst_ar.append(ar)
        mean_hsv = np.mean(lst_ar, axis=0).tolist()
        mean_hsvs.append(mean_hsv)
        catdict[cat] = mean_hsv
     
    r = np.array([hsv[1] for hsv in mean_hsvs] )  
    theta = np.array([math.radians(l) for l in [hsv[0] for hsv in mean_hsvs] ]) 
    area = np.full(len(data), 10) 
    colors = theta

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='polar')
    c = ax.scatter(theta, r, c=colors, s=area, cmap='hsv', alpha=0.75)
    plt.title(f"Color Thesaurus AVG VIAN colors in HSV 2D",fontsize=20, y=1.1)
    
    if save: 
        os.chdir(r'D:\thesis\output_images')
        plt.savefig('HSV_PolarPlot_AVG_VIAN_Color_Thesaurus.jpg')
    plt.show()

def make_hsv_vars(data): 
    """ create new hsv data 
    mapping from radians to cartesian 
    (based on function lch to lab) """
    hsv_cart = hsvdeg2hsvcart([eval(l) for l in data['hsv']]).tolist()
    data['hsv_cart'] = hsv_cart
    data['hsv_cart_H'] = [i[0] for i in hsv_cart]
    data['hsv_cart_S'] = [i[1] for i in hsv_cart]
    data['hsv_cart_V'] = [i[2] for i in hsv_cart]
    return data 

def show_hsvset_in_hsvspace(data, save=False): 
    """ Visualizing all HSV Colors in HSV-Space """
    h = np.array(data['hsv_cart_H'])
    s = np.array(data['hsv_cart_S'])
    v = np.array(data['hsv_cart_V'])
    
    p = [eval(l) for l in data['srgb'].tolist()]
    p = np.array(p)
    pixel_colors = p
    norm = colors.Normalize(vmin=-1.,vmax=1.)
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()
    
    fig = plt.figure(figsize=(8,8))
    axis = fig.add_subplot(1, 1, 1, projection="3d")
    axis.scatter(h, s, v, facecolors=pixel_colors, marker=".")
    axis.set_xlabel("Hue") 
    axis.set_ylabel("Saturation") 
    axis.set_zlabel("Value")  
    axis.set_zlim([0,1])
    axis.plot([0,0], [s.flatten().min(), s.flatten().max()], zs=0, color='red', linestyle='dashed', linewidth=2, markersize=12)
    axis.plot([h.flatten().min(), h.flatten().max()], [0, 0], zs=0, color='red', linestyle='dashed', linewidth=2, markersize=12)
    axis.plot([0, 0], [0, 0], zs=[0,v.flatten().max()], color='red', linestyle='dashed', linewidth=2, markersize=12)
    plt.title(f"Color Thesaurus VIAN colors in HSV Space",fontsize=20, y=1.05)
    if save: 
        os.chdir(r'D:\thesis\output_images')
        plt.savefig('HSV_Space_VIAN_Color_Thesaurus.jpg')
    plt.show()

def show_hsvsubset_in_hsvspace(data, color_filter2): 
    subdata = data[data['cat1'] == color_filter2]
    h = np.array(subdata['hsv_cart_H'])
    s = np.array(subdata['hsv_cart_S'])
    v = np.array(subdata['hsv_cart_V'])
    s.flatten().min()
    
    p = [eval(l) for l in subdata['srgb'].tolist()]
    p = np.array(p)
    pixel_colors = p
    norm = colors.Normalize(vmin=-1.,vmax=1.) 
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()
    
    fig = plt.figure(figsize=(8,8))
    axis = fig.add_subplot(1, 1, 1, projection="3d")
    axis.scatter(h, s, v, facecolors=pixel_colors, marker=".")
    axis.set_xlabel("Hue") 
    axis.set_ylabel("Saturation") 
    axis.set_zlabel("Value")  
    axis.set_zlim([0,1])
    axis.plot([0,0], [s.flatten().min(), s.flatten().max()], zs=0, color='red', linestyle='dashed', linewidth=2, markersize=12)
    axis.plot([h.flatten().min(), h.flatten().max()], [0, 0], zs=0, color='red', linestyle='dashed', linewidth=2, markersize=12)
    axis.plot([0, 0], [0, 0], zs=[0,v.flatten().max()], color='red', linestyle='dashed', linewidth=2, markersize=12)
    plt.title(f"Color Thesaurus VIAN colors in HSV Space: {color_filter2}",fontsize=20, y=1.05)
    plt.show()   
 
def rotate_plot(plot, step=60): 
    for angle in np.linspace(0, 360, step).tolist():
        plot 
        ax.view_init(60, angle)
        plt.draw()
        plt.pause(.001)
        plt.show()
        
#%%
    
if __name__ == '__main__': 

    data = load_data(PATH, FILE)

    # rgb space
    thesaurusvian_in_rgbspace(data, save=False)
    show_rgbsubset_in_rgbspace(data, COLOR_FILTER)
    show_labset_in_labspace(data, save=False)

    # lab space
    catdict, cats = vian_compute_label_avg(data)
    plot_lab_avg(catdict, save=False)  
    show_labsubset_in_labspace(data, COLOR_FILTER2)
    plot_2d_polar_scatterplot_hsv(cats)

    # hsv space
    data = make_hsv_vars(data)
    show_hsvset_in_hsvspace(data, save=False)
    show_hsvsubset_in_hsvspace(data, COLOR_FILTER2)


    
