'''
======================
3D surface (color map)
======================

Demonstrates plotting a 3D surface colored with the coolwarm color map.
The surface is made opaque by using antialiased=False.

Also demonstrates using the LinearLocator and custom formatting for the
z axis tick labels.
'''
import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

# import data
os.chdir(r'D:\thesis\code\pd6hues')
df = pd.read_csv('rgbhsvlablch_6.csv', index_col=[0])
df.index.name = 'index'

print(df)


# build X 
# X is the input features by row.
# array([[ 1.02956195e+00,  1.12384202e+00,  1.28943006e+00], []]) 
lab2pts = df['LAB']
lab2pts = [eval(l) for l in lab2pts]
X = np.array(lab2pts)

# Y is the class labels for each row of X.
# array([0.]) 
lab2pt = list(df.index)
y = np.array(lab2pt)



#X = [[0, 0], [1, 1]]
#y = [0, 1]
clf = svm.SVC()
clf.fit(X, y)


# get support vectors
clf.support_vectors_
# get indices of support vectors
clf.support_
# get number of support vectors for each class
clf.n_support_

# decision function 
decision_function = clf.decision_function(X)

#%%
fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data X, Y, Z - with same shape .
X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
Z = np.arange(-5, 5, 0.25)
#
#R = np.sqrt(X**2 + Y**2)
#Z = np.sin(R)
# apply trained classifier to meshgrid to get hyperplanes
Z = clf.predict(np.c_[X.ravel(), Y.ravel(), Z.ravel()])
Z = clf.decision_function(np.c_[X.ravel(), Y.ravel(), Z.ravel()])
# reshape possible if new shape that has same product as old shape
np.reshape(X, (640,100)).shape
np.reshape(Y, (640,100)).shape
np.reshape(Z, (40,40)).shape
# all need to have same shape and Z needs to be in 2D to 3D plot decision surface
print(X.shape)
print(Y.shape)
print(Z.shape)

X, Y = np.meshgrid(X, Y)

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
