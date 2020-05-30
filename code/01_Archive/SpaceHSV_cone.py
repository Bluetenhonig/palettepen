# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 14:30:24 2020

@author: lsamsi
"""


import numpy as np
import cv2

H = np.repeat([np.linspace(0, 179, 100)], 100, axis=0)
S = np.repeat([np.concatenate((np.linspace(0, 255, 50), np.linspace(255, 0, 50)))], 100, axis=0).transpose()
V = np.repeat([np.concatenate((np.ones(50)*255, np.linspace(255, 0, 50)))], 100, axis=0).transpose()

hsv = np.asarray(cv2.merge((H, S, V)), dtype=np.uint8)
C = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

theta = np.linspace(0, 2*np.pi, 100)
X = np.asarray([np.zeros(100), np.cos(theta), np.zeros(100)])
Y = np.asarray([np.zeros(100), np.sin(theta), np.zeros(100)])
Z = np.asarray([2*np.ones(100), 2*np.ones(100), np.zeros(100)])

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, Y, Z, facecolors=C/255.)