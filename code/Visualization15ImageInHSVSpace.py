# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 15:25:17 2020

@author: lsamsi
"""
import os
import matplotlib.pyplot as plt #2.1.2
import numpy as np
import cv2

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors

os.getcwd()
os.chdir(r'D:\thesis\images')

# all the color space conversions OpenCV provides 
flags = [i for i in dir(cv2) if i.startswith('COLOR_')]
# [ 'COLOR_BGR2BGR555','COLOR_BGR2BGR565']

nemo = cv2.imread('nemo.png')
nemo.shape
# (298, 198, 3) # (x, y, z channels)

# plot image
nemo = cv2.cvtColor(nemo, cv2.COLOR_BGR2RGB)
#plt.imshow(nemo)
#plt.show()

color = nemo[297,197] # most bottom right pixel 
print(color)

# get HSV 
hsv_nemo = cv2.cvtColor(nemo, cv2.COLOR_RGB2HSV)
#plt.imshow(hsv_nemo)
#plt.show()

# get HSV values at an image point
color = hsv_nemo[297,197] # most bottom right pixel 
print(color)


#TODO: get angle for hue, and normalize value and saturation, see problem line 68-69

# plot figure 
fig = plt.figure()
axis = fig.add_subplot(1, 1, 1, projection="3d")

# plot cylinder
def data_for_cylinder_along_z(center_x,center_y,radius,height_z):
    z = np.linspace(0, height_z, 50)
    theta = np.linspace(0, 2*np.pi, 50)
    theta_grid, z_grid=np.meshgrid(theta, z)
    x_grid = radius*np.cos(theta_grid) + center_x
    y_grid = radius*np.sin(theta_grid) + center_y
    return x_grid,y_grid,z_grid

Xc,Yc,Zc = data_for_cylinder_along_z(0,0,200,250) # center_x,center_y,radius,height_z
axis.plot_surface(Xc, Yc, Zc, alpha=0.2)

# get colors 
hsv_nemo = cv2.cvtColor(nemo, cv2.COLOR_RGB2HSV)
h, s, v = cv2.split(hsv_nemo)

h_max = h.flatten().max()
# 179
# hue max 360°
s_max = s.flatten().max()
# 255
# saturation max 100
v_max = v.flatten().max()
# 255
 # saturation max 100 

pixel_colors = nemo.reshape((np.shape(nemo)[0]*np.shape(nemo)[1], 3))
pixel_colors.shape
# (59004 = 298*198 (image size), 3)
norm = colors.Normalize(vmin=-1.,vmax=1.)
norm.autoscale(pixel_colors)
pixel_colors = norm(pixel_colors).tolist()

# plots color dots 
axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors='burlywood', marker=".") #axis.scatter(coordinate locations, coloring, shape)

# set axes
axis.set_xlim([-250,250])
axis.set_ylim([-250,250])

axis.set_xlabel("x: Hue") 
axis.set_ylabel("y: Saturation") 
axis.set_zlabel("z: Value")

axis.plot((0,300),(0,0), (0,0), '-k', label='x-axis: Hue') #hue (x1,y1), (x2,y2), (x2,y3)
axis.plot((0,0),(0,300), (250,250), '-k', label='y-axis: Saturation') #saturation
axis.plot((0,0),(0,0), (0,250), '-k', label='z-axis: Value') #value

 # rotate 3D to desired angle 
axis.view_init(30, 30)

axis.legend()
plt.show()



