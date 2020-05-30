# -*- coding: utf-8 -*-
"""
Created on Thu May  7 15:17:27 2020

@author: Linda Samsinger

Distances in Color Space 

"""

# distance between two colors 
# based on: https://docs.scipy.org/doc/scipy/reference/spatial.distance.html

# TODO: look at lit what dist was used in LAB  nbgvfcdxsay<asxy<>x  

# params: distance metrics
# params: color spaces 

# define colors
variablelabel = ['A', 'B', 'C','D']
# color wt/o weights
A = [1, 0, 0]
B = [0, 1, 0]
C = [1, 2, 0]
D = [1, 1, 0]

# color with weights
Aw = [1, 0, 0, 0.5]
Bw = [0, 1, 0, 0.2]
Cw = [1, 2, 0, 0.4]
Dw = [1, 1, 0, 0.8]

#%%
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
import numpy as np


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


ax.scatter([1,0,1,1], [0,1,2,1], [0,0,0,0], marker='.', s=200)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()

#%%
from scipy.spatial import distance


# Euclidean - OK
distance.euclidean(A, D)

# Canberra - OK
distance.canberra(A, D)

# Chebyshev - OK
distance.chebyshev(A, D)

# Cityblock - OK
distance.cityblock(A, D)

# Minkowski - OK
distance.minkowski(A, D)

# Jaccard
distance.jaccard(A, B)
distance.jaccard(A, C)
distance.jaccard(A, D)

# Correlation 
distance.correlation(A, D)

# Cosine
distance.cosine(A, D)

# Bray Curtis
distance.braycurtis(A, D)

#%%

# Euclidean 
distance.euclidean(Aw, Dw)

# Canberra 
distance.canberra(Aw, Dw)

# Chebyshev - OK
distance.chebyshev(Aw, Dw)

# Cityblock 
distance.cityblock(Aw, Dw)

# Minkowski
distance.minkowski(Aw, Dw)

# Jaccard
distance.jaccard(Aw, Bw)
distance.jaccard(Aw, Cw)
distance.jaccard(Aw, Dw)

# Correlation 
distance.correlation(Aw, Dw)

# Cosine
distance.cosine(Aw, Dw)

# Bray Curtis
distance.braycurtis(A, D)