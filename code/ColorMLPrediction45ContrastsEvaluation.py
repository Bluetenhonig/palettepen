# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 08:05:47 2020

@author: Linda Samsinger 

Confusion matrix for color contrast classification

"""

# import modules

import os
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

#%%


# declare variables 
PATH = r'D:\thesis\film_colors_project\sample-dataset'
FILE = 'KNN21_dataset_ground_truth_HNBFLS.xlsx'
FILE2 = 'KNN21_dataset_2.xlsx'


#%%
def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):


    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 4))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, size=16)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)
        plt.xlim(0-0.5, 1+.5)
        plt.ylim(0-0.5, 1+.5)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label', size=14)
    plt.xlabel('Predicted label\n\naccuracy={:0.4f}; error={:0.4f}'.format(accuracy, misclass), size=14)
    plt.show()
    
    
#%%    
if __name__ == '__main__':

    # load data 
    os.chdir(PATH)
    data = pd.read_excel(FILE, index_col=[0])  
    data_pred = pd.read_excel(FILE2, index_col=[0])    
    data.info()
    # drop Nans
    data = data.dropna()

    # feature selection  
    cols_true = data.columns[25:35]
    print(cols_true)
    cols_pred = data_pred.columns[1:10]
    print(cols_pred)
    
#%%   
    # plot confusion matrix per contrast
    for i in range(len(cols_true)):         
        y_true = data[cols_true[i]].tolist()
        y_pred = [int(el) for el in data_pred[cols_pred[i]].tolist()]
        # get confusion matrix 
        cm = confusion_matrix(y_true, y_pred)
        # plot confusion matrix
        title = cols_pred[i]
        plot_confusion_matrix(cm           = cm, 
                              normalize    = False,
                              target_names = ['yes', 'no'],
                              title        =  f"{title} \nConfusion Matrix")