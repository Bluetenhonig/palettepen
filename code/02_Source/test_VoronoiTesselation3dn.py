# -*- coding: utf-8 -*-
"""
Created on Wed May  6 10:10:14 2020

@author: lsamsi
"""

import numpy as np
from matplotlib import colors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
from scipy.spatial import SphericalVoronoi
from mpl_toolkits.mplot3d import proj3d
# set input data
points = np.array([[0, 0, 1], [0, 0, -1], [1, 0, 0],
                    [0, 1, 0], [0, -1, 0], [-1, 0, 0], ])
# center = np.array([0, 0, 0])
# radius = 1
# calculate spherical Voronoi diagram
sv = SphericalVoronoi(points)
# sort vertices (optional, helpful for plotting)
sv.sort_vertices_of_regions()
# generate plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# plot the unit sphere for reference (optional)
# u = np.linspace(0, 2 * np.pi, 100)
# v = np.linspace(0, np.pi, 100)
# x = np.outer(np.cos(u), np.sin(v))
# y = np.outer(np.sin(u), np.sin(v))
# z = np.outer(np.ones(np.size(u)), np.cos(v))
# ax.plot_surface(x, y, z, color='y', alpha=0.1)

# # plot generator points
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b')
# plot Voronoi vertices
ax.scatter(sv.vertices[:, 0], sv.vertices[:, 1], sv.vertices[:, 2], c='g')
# indicate Voronoi regions (as Euclidean polygons)
for region in sv.regions:
    random_color = colors.rgb2hex(np.random.rand(3))
    polygon = Poly3DCollection([sv.vertices[region]], alpha=.3)
    polygon.set_color(random_color)
    ax.add_collection3d(polygon)
ax.azim = 10
ax.elev = 40
_ = ax.set_xticks([])
_ = ax.set_yticks([])
_ = ax.set_zticks([])
fig.set_size_inches(4, 4)
plt.show()

