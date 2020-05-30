# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 11:47:56 2020

@author: lsamsi
"""

#based on: https://stackoverflow.com/questions/36232334/plotting-3d-decision-boundary-from-linear-svm

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.svm import SVC



# import data
os.chdir(r'D:\thesis\code\pd6hues')
df = pd.read_csv('rgbhsvlablch_6.csv', index_col=[0])
df.index.name = 'index'


df.head()


# build X 
# X is the input features by row.
# array([[ 1.02956195e+00,  1.12384202e+00,  1.28943006e+00], []]) 
lab2pts = df['LAB'].iloc[0:2]
lab2pts = [eval(l) for l in lab2pts]
X = np.array(lab2pts)

# Y is the class labels for each row of X.
# array([0.]) 
lab2pt = list(df.index[:2])
Y = np.array(lab2pt)

rs = np.random.RandomState(1234)

### Support Vector Classifier 
# Fit the data with an svm
svc = SVC(kernel='linear')
svc.fit(X,Y)

# The equation of the separating plane is given by all x in R^3 such that:
# np.dot(svc.coef_[0], x) + b = 0. We should solve for the last coordinate
# to plot the plane in terms of x and y.

z = lambda x,y: (-svc.intercept_[0]-svc.coef_[0][0]*x-svc.coef_[0][1]*y) / svc.coef_[0][2]

tmp = np.linspace(-100,100,51)
x,y = np.meshgrid(tmp,tmp)

# Plot stuff.
fig = plt.figure()
ax  = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z(x,y))
ax.plot3D(X[Y==0,0], X[Y==0,1], X[Y==0,2],'ob')
ax.plot3D(X[Y==1,0], X[Y==1,1], X[Y==1,2],'sr')
plt.show()