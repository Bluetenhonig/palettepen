/*-*- coding: utf-8 -*-
"""
Created on Fri Jul 17 20:52:26 2020

@author: Linda Samsinger

=====================
ML Classification Comparison
=====================

A comparison of a several classifiers in scikit-learn on colors dataset.
The point of this is to illustrate the nature of fuzzy decision boundaries
of different classifiers.

Multiclass-Multioutput Classification, non-binary (Warning!!! At present, no metric in sklearn.metrics supports the multioutput-multiclass classification task.)
transform to Multilabel classification, binary using make_multilabel_classification

"""



# import modules
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import math
import timeit
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.utils import shuffle

# sklearn ml multi-label classifiers: 
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifierCV

from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


#%%

# declare variables 
PATH = r'D:\thesis\input_color_name_dictionaries\thesaurus\datasets'
FILE = 'eeffcnd_thesaurus_basicvian_upinterval.xlsx'
SOURCE = 'THESAURUS'
SYSTEM = 'VIAN'
METHOD = 'INTERVAL' 

FEATURE = 'cielab'
LABEL =  'cat1'
LABEL2 = 'cat2'

#%%

# functions
 
def analyze_dataset(df, label, label2=None): 
    """ exploratory data analysis """
    print(f"Basic colors' distribution {SYSTEM} in the color name dictionary {SOURCE}: ")
    valcts = df[label].value_counts()
    print(valcts)
    valctsplt = df[label].value_counts().plot(kind='bar')
    print(valctsplt)

    print(f"Basic colors' second label distribution {SYSTEM} in the color name dictionary {SOURCE}: ")
    valcts = df[label2].value_counts()
    print(valcts)
    valctsplt = df[label2].value_counts().plot(kind='bar')
    print(valctsplt)


def get_X_y(df, lab_column, label, label2): 
    """get (X, y), original CS: LAB 
    input = X = [[0, 0], [1, 1]]; y = [0, 1]"""
    # X = lab color values 
    lab2pts = df[lab_column] 
    lab2pts = [eval(l) for l in lab2pts]
    # y = label encoding 
    lab2pt = df[label].tolist() 
    le = preprocessing.LabelEncoder()
    le.fit(lab2pt) 
    list(le.classes_) 
    lab2dt = le.transform(lab2pt) 
    list(le.inverse_transform(lab2dt)) 
    # y2 = multiclass-multioutput to multilabel 
    lab2pt2 = df[label2].tolist() #list(df.index)
    no_nanlab2pt2 = [x for x in lab2pt2 if str(x) != 'nan']
    lab2dt2 = le.transform(no_nanlab2pt2)  
    lab2pt2_enc = len(lab2pt2) * [None]
    j = 0 
    for i, el in enumerate(lab2pt2): 
        el = str(el)
        if el != 'nan': 
            lab2pt2_enc[i] = lab2dt2[j]
            j +=1  
    multioutput = [] 
    j = 0 
    for i in range(len(lab2dt)):
        row = [lab2dt[i]]
        if lab2pt2_enc[i] != None: 
            el2 = lab2dt2[j]
            j += 1
            row.append(el2)   
        multioutput.append(row)
    
    # X and y 
    X = np.array(lab2pts)
    y_multilabel = MultiLabelBinarizer().fit_transform(multioutput)
    print(f"You have {len(X)} colors in color name dictionary {SOURCE} for {df[LABEL].nunique()} color categories in LAB.")
    # shuffle 
    X, y_multilabel = shuffle(X, y_multilabel, random_state=24)  
    return X, y_multilabel 



def grid_search_pipeline(X, y): 
    """ making pipeline and gridsearchcv"""
clf = Pipeline(steps=[  
                    ('scaler', StandardScaler()) 
                #    ,('dtc', MultiOutputClassifier(DecisionTreeClassifier()))
                # , ('etc', MultiOutputClassifier(ExtraTreeClassifier())) 
                # , ('etcs', MultiOutputClassifier(ExtraTreesClassifier()))
                , ('clf', KNeighborsClassifier())
                # , ('mlp', MultiOutputClassifier(MLPClassifier()))
               #*- ('rnc', MultiOutputClassifier(RadiusNeighborsClassifier()))
               # , ('rfc', MultiOutputClassifier(RandomForestClassifier()))
               # , ('rcl', MultiOutputClassifier(RidgeClassifierCV()))
               ]) 

    param_grid = [
        { 'clf': [MultiOutputClassifier(KNeighborsClassifier(weights='distance', n_jobs = -1))], 
        'clf__estimator__n_neighbors': list(range(60)),
         'clf__estimator__leaf_size': list(range(60))  
    
        #'svc__C': np.linspace(0,10,200).tolist()
            }, 
        {'clf': [MultiOutputClassifier(RadiusNeighborsClassifier())], 
         'clf__estimator__radius': np.arange(0,10.5,.5)[1:].tolist(),
         'clf__estimator__leaf_size': np.arange(1,61).tolist(),
         } ]
    
        scoring = {'balanced_accuracy': 'balanced_accuracy'
                   , 'f1_micro': 'f1_micro'        
                   ,'f1_macro': 'f1_macro'
                   , 'f1_weighted': 'f1_weighted'
                   ,'precision_micro': 'precision_micro'
                   , 'precision_macro': 'precision_macro'
                   ,'precision_weighted': 'precision_weighted'
                   , 'recall_micro': 'recall_micro'
                   ,'recall_macro': 'recall_macro'
                   , 'recall_weighted': 'recall_weighted'
         }
    
    search = GridSearchCV(clf, param_grid, scoring='accuracy', n_jobs=-1, cv=5) 
    search.fit(X_train, y_train)
      
    print(f'Training Machine Learning Classifier for {SYSTEM} Color Categories: successful!')
    
    return search 

def best_result_gridsearch(search): 
    print("Best parameter (CV score=%0.3f):" % search.best_score_)
    print("Best parameter :", search.best_estimator_)
    print("Best parameter :", search.best_params_)
    print("Best parameters set:")
    best_parameters = search.best_estimator_.get_params()
    best_estimator = search.best_estimator_
    best_score = search.best_score_
    for param_name in sorted(param_grid.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))        
    return best_parameters, best_estimator, best_score 


#%%    
if __name__ == '__main__':

     # set directory 
    os.chdir(PATH)
    
    # load data 
    df = pd.read_excel(FILE, sep=" ", index_col=0)
    df.info()
 
    # analyze data   
    analyze_dataset(df, LABEL, LABEL2)   
    # preprocessing
    X, y = get_X_y(df, FEATURE, LABEL)
    # processing gridsearch
    start = timeit.timeit()
    gridsearch = grid_search_pipeline(X, y)
    end = timeit.timeit()
    duration = start-end
    print(duration)
    
    # best results of gridsearch
    best_parameters, best_estimator, best_score  = best_result_gridsearch(gridsearch)






