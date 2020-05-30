# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 13:49:34 2020

@author: lsamsi
"""


import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np

# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import


def midpoints(x):
    sl = ()
    for i in range(x.ndim):
        x = (x[sl + np.index_exp[:-1]] + x[sl + np.index_exp[1:]]) / 2.0
        sl += np.index_exp[:]
    return x

# prepare some coordinates, and attach rgb values to each
r, theta, z = np.mgrid[0:1:6j, 0:np.pi*2:12j, -0.5:0.5:6j] #np.mgrid[saturation rgba 0 to 1 with 5 steps, hue 12 angles, value]
x = r*np.cos(theta)
y = r*np.sin(theta)

rc, thetac, zc = midpoints(r), midpoints(theta), midpoints(z)

# define a wobbly torus about [0.7, *, 0]
sphere = rc

# combine the color components
hsv = np.zeros(sphere.shape + (3,))
hsv[..., 0] = thetac / (np.pi*2) # hue
hsv[..., 1] = rc #saturation
hsv[..., 2] = zc + 0.5 #value

colors = matplotlib.colors.hsv_to_rgb(hsv)

# and plot everything
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.voxels(x, y, z, sphere,
          facecolors=colors,
          #edgecolors=np.clip(2*colors - 0.5, 0, 1),  # brighter
          linewidth=0.5
          )

# remove frame axis grid 
#ax.axis("off")
# set axes
# ax.set_xlim([-250,250])
# ax.set_ylim([-250,250])

ax.set_xlabel("x: Hue") 
ax.set_ylabel("y: Saturation") 
ax.set_zlabel("z: Value")

 # rotate 3D to desired angle 
ax.view_init(30, 30)

plt.show()