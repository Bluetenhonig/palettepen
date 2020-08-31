# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 16:25:48 2020

@author: Linda Samsinger

=====================
ML Classification Comparison
=====================

A comparison of a several classifiers in scikit-learn on colors dataset.
The point of this is to illustrate the nature of decision boundaries
of different classifiers.

Kernel Trick: Particularly in high-dimensional spaces (3D), data can more easily be separated
linearly and the simplicity of classifiers such as naive Bayes and linear SVMs
might lead to better generalization than is achieved by other classifiers.

The plots show training points in solid colors and can show testing points
semi-transparent.

"""

#import modules  
import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
import warnings
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
# machine learning classifiers: multi-class
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, RidgeClassifier, RidgeClassifierCV, SGDClassifier, Perceptron, PassiveAggressiveClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid, RadiusNeighborsClassifier
from sklearn.svm import LinearSVC, NuSVC, SVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.gaussian_process import GaussianProcessClassifier
# machine learning classifiers: scorer
from sklearn.model_selection import train_test_split
# machine learning classifiers: metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score # for multi-class
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import multilabel_confusion_matrix, f1_score, recall_score, precision_score, confusion_matrix, balanced_accuracy_score, roc_auc_score
from sklearn.metrics import hinge_loss, matthews_corrcoef, fbeta_score, precision_recall_fscore_support
from timeit import default_timer as timer

#%%

# declare variables 
PATH = r'D:\thesis\input_color_name_dictionaries\thesaurus\datasets'
FILE = 'eeffcnd_thesaurus_basicvian_upinterval.xlsx'
SOURCE = 'THESAURUS'
SYSTEM = 'VIAN'
METHOD = 'INTERVAL' 

FEATURE = 'cielab'
LABEL =  'cat1'
LABEL2 =  'cat2'

SAVE_PATH = r'D:\thesis\machine_learning\models'

warnings.filterwarnings('ignore')

#%%

# functions
 
def analyze_dataset(df, label, label2): 
    df = df[df[label2].apply(lambda x: isinstance(x, float))]
    print(f"Basic colors' distribution {SYSTEM} in the color name dictionary {SOURCE}: ")
    valcts = df[label].value_counts()
    print(valcts)
    valctsplt = df[label].value_counts().plot(kind='bar')
    print(valctsplt)
    

def get_X_y(df, lab_column, label): 
    """normalization / standardization
    get (X, y), original CS: LAB 
    input = X = [[0, 0], [1, 1]]; y = [0, 1]"""
    # X = lab color values 
    lab2pts = df[lab_column] 
    lab2pts = [eval(l) for l in lab2pts]
    # standard scaling
    scaler = StandardScaler()
    lab2pts = scaler.fit_transform(lab2pts)
    # y = label encoding 
    lab2pt = df[label].tolist() 
    le = preprocessing.LabelEncoder()
    le.fit(lab2pt) 
    list(le.classes_) 
    lab2dt = le.transform(lab2pt) 
    list(le.inverse_transform(lab2dt))  
    
    # X and y 
    X = np.array(lab2pts)
    y = np.array(lab2dt)
    print(f"You have {len(X)} colors in color name dictionary {SOURCE} for {df[LABEL].nunique()} color categories in LAB.")
    # shuffle 
    X, y = shuffle(X, y, random_state=24)   
    return X, y 

def test_train_split(X, y, test_size): 
    """ splits data set into test and training set"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    print(f"You have {len(X_train)} training colors and {len(X_test)} test colors - test_size: {test_size*100}.")
    print('Number of classes in y_test: ', len(set(y_test)))
    return X_train, X_test, y_train, y_test  


   
def multi_class_scorer_train_test_split(y_true, y_pred, y_pred2): 
    """ helper functions for train_test_split """
    train_balanced_accuracy_score_all = []
    train_confusion_matrix_all = []
    train_multilabel_confusion_matrix_all = []
    train_matthews_corrcoef_all = []
    train_fbeta_score_micro_all = []
    train_fbeta_score_macro_all = []
    train_fbeta_score_weighted_all = []
    train_precision_recall_fscore_support_micro_all =[]
    train_precision_recall_fscore_support_macro_all = []
    train_precision_recall_fscore_support_weighted_all = []
    test_balanced_accuracy_score_all= []
    test_confusion_matrix_all= []
    test_multilabel_confusion_matrix_all= []
    test_matthews_corrcoef_all= []
    test_fbeta_score_micro_all= []
    test_fbeta_score_macro_all= []
    test_fbeta_score_weighted_all= []
    test_precision_recall_fscore_support_micro_all= []
    test_precision_recall_fscore_support_macro_all= []
    test_precision_recall_fscore_support_weighted_all= []
    training = [train_balanced_accuracy_score_all, train_matthews_corrcoef_all, train_fbeta_score_micro_all, train_fbeta_score_macro_all, train_fbeta_score_weighted_all, train_precision_recall_fscore_support_micro_all, train_precision_recall_fscore_support_macro_all, train_precision_recall_fscore_support_weighted_all]
    testing = [test_balanced_accuracy_score_all, test_matthews_corrcoef_all, test_fbeta_score_micro_all, test_fbeta_score_macro_all, test_fbeta_score_weighted_all, test_precision_recall_fscore_support_micro_all, test_precision_recall_fscore_support_macro_all, test_precision_recall_fscore_support_weighted_all]
 
    train_balanced_accuracy_score_all.append(balanced_accuracy_score(y_train, y_pred))
    train_confusion_matrix_all.append(confusion_matrix(y_train, y_pred))
    train_multilabel_confusion_matrix_all.append(multilabel_confusion_matrix(y_train, y_pred))
    # d = hinge_loss(y_train, y_pred, labels)
    train_matthews_corrcoef_all.append(matthews_corrcoef(y_train, y_pred))
    # f= roc_auc_score(y_train, y_pred, average='micro', multi_class='ovo')
    train_fbeta_score_micro_all.append(fbeta_score(y_train, y_pred, average='micro', beta=1))
    train_fbeta_score_macro_all.append(fbeta_score(y_train, y_pred, average='macro', beta=1))
    train_fbeta_score_weighted_all.append(fbeta_score(y_train, y_pred, average='weighted', beta=1))
    train_precision_recall_fscore_support_micro_all.append(precision_recall_fscore_support(y_train, y_pred, beta=1, average='micro'))
    train_precision_recall_fscore_support_macro_all.append(precision_recall_fscore_support(y_train, y_pred, beta=1, average='macro'))
    train_precision_recall_fscore_support_weighted_all.append(precision_recall_fscore_support(y_train, y_pred, beta=1, average='weighted'))

    test_balanced_accuracy_score_all.append(balanced_accuracy_score(y_test, y_pred2))
    test_confusion_matrix_all.append(confusion_matrix(y_test, y_pred2))
    test_multilabel_confusion_matrix_all.append(multilabel_confusion_matrix(y_test, y_pred2))
    # d2 = hinge_loss(y_test, labels=y_pred)
    test_matthews_corrcoef_all.append(matthews_corrcoef(y_test, y_pred2))
    # f2= roc_auc_score(y_test, y_pred, average='micro', multi_class='ovo')
    test_fbeta_score_micro_all.append(fbeta_score(y_test, y_pred2, average='micro', beta=1))
    test_fbeta_score_macro_all.append(fbeta_score(y_test, y_pred2, average='macro', beta=1))
    test_fbeta_score_weighted_all.append(fbeta_score(y_test, y_pred2, average='weighted', beta=1))
    test_precision_recall_fscore_support_micro_all.append(precision_recall_fscore_support(y_test, y_pred2, beta=1, average='micro'))
    test_precision_recall_fscore_support_macro_all.append(precision_recall_fscore_support(y_test, y_pred2, beta=1, average='macro'))
    test_precision_recall_fscore_support_weighted_all.append(precision_recall_fscore_support(y_test, y_pred2, beta=1, average='weighted'))    
    return training, testing, test_balanced_accuracy_score_all

def scores(set_form, max_index): 
    """ helper function for multi_class_scorer_train_test_split"""
    scores = []
    for i in set_form: 
        i = i[max_index]
        if type(i) == tuple: 
            prec = i[0]
            recall = i[1]
            f1 = i[2]
            scores.append(prec)
            scores.append(recall)
            scores.append(f1)
        else:  
            scores.append(i) 
    return scores

    
def cross_validation_logisticregression(X, y, cv=5): 
    params = []
    
    solver = ['saga']
    multi_class=['auto', 'ovr', 'multinomial']
    print('Number of combinations: ', len(solver)*len(multi_class))
    
    score_names_all = []
    test_scores_all = []
    train_scores_all = []
    
    for s in solver: 
        for multi in multi_class:         
            clf = LogisticRegression(penalty='elasticnet',  solver = s, class_weight=None, random_state=24,  max_iter=1000, multi_class=multi, n_jobs = -1, l1_ratio=.5)
            scoring = ['balanced_accuracy', 'f1_micro', 'f1_macro', 'f1_weighted', 'precision_micro', 'precision_macro', 'precision_weighted','recall_micro', 'recall_macro', 'recall_weighted', 'roc_auc_ovr', 'roc_auc_ovo', 'roc_auc_ovr_weighted', 'roc_auc_ovo_weighted']
            # train on training set  
            scores = cross_validate(clf, X, y, cv=cv, scoring=scoring, return_train_score=True)
            score_names = []
            test_scores = [] 
            train_scores = []        
            for el in sorted(scores.keys()): 
                if el.startswith('test') and not el.endswith('matrix'): 
                    test_scores.append(scores[el].mean())
                    score_names.append(el[5:])
                elif el.startswith('train') and not el.endswith('matrix'): 
                    train_scores.append(scores[el].mean())
            assert len(train_scores) == len(test_scores)
            assert len(train_scores) == len(score_names)
    
            score_names_all.append(score_names)
            test_scores_all.append(test_scores)
            train_scores_all.append(train_scores)
            params.append(f'solver: {s}, multi_class: {multi}')
    
    # get max score's parameters 
    for el in test_scores_all: 
        if el[score_names.index('balanced_accuracy')] == max([i[score_names.index('balanced_accuracy')] for i in test_scores_all]): 
            windi = test_scores_all.index(el)
    
    print('Highest score: ', max([i[score_names.index('balanced_accuracy')] for i in test_scores_all]))
    print(params[windi])
  
    train_scores = train_scores_all[score_names.index('balanced_accuracy')]
    test_scores = test_scores_all[score_names.index('balanced_accuracy')]
    score_names =  score_names_all[score_names.index('balanced_accuracy')]
    
    cv5_1 = pd.DataFrame({'classifier': clf.__class__.__name__
                             , 'training': train_scores
                            , 'test': test_scores
                            , 'score': score_names})
    return cv5_1
    
def train_test_split_logisticregression(X_train, y_train):   
    params = []    
    solver = ['saga']
    multi_class=['auto', 'ovr', 'multinomial']
    print('Number of combinations: ', len(solver)*len(multi_class))
   
    train_balanced_accuracy_score_all = []
    train_confusion_matrix_all = []
    train_multilabel_confusion_matrix_all = []
    train_matthews_corrcoef_all = []
    train_fbeta_score_micro_all = []
    train_fbeta_score_macro_all = []
    train_fbeta_score_weighted_all = []
    train_precision_recall_fscore_support_micro_all =[]
    train_precision_recall_fscore_support_macro_all = []
    train_precision_recall_fscore_support_weighted_all = []
    test_balanced_accuracy_score_all= []
    test_confusion_matrix_all= []
    test_multilabel_confusion_matrix_all= []
    test_matthews_corrcoef_all= []
    test_fbeta_score_micro_all= []
    test_fbeta_score_macro_all= []
    test_fbeta_score_weighted_all= []
    test_precision_recall_fscore_support_micro_all= []
    test_precision_recall_fscore_support_macro_all= []
    test_precision_recall_fscore_support_weighted_all= []
    training = [train_balanced_accuracy_score_all, train_matthews_corrcoef_all, train_fbeta_score_micro_all, train_fbeta_score_macro_all, train_fbeta_score_weighted_all, train_precision_recall_fscore_support_micro_all, train_precision_recall_fscore_support_macro_all, train_precision_recall_fscore_support_weighted_all]
    testing = [test_balanced_accuracy_score_all, test_matthews_corrcoef_all, test_fbeta_score_micro_all, test_fbeta_score_macro_all, test_fbeta_score_weighted_all, test_precision_recall_fscore_support_micro_all, test_precision_recall_fscore_support_macro_all, test_precision_recall_fscore_support_weighted_all]
    
    for s in solver: 
        for multi in multi_class:         
            clf = LogisticRegression(penalty='elasticnet',  solver = s, class_weight=None, random_state=24,  max_iter=1000, multi_class=multi, n_jobs = -1, l1_ratio=.5)
            # train on training set  
            clf.fit(X_train, y_train)
            # training set 
            y_pred = clf.predict(X_train)  
            # multiclass evaluation scores 
            train_balanced_accuracy_score_all.append(balanced_accuracy_score(y_train, y_pred))
            train_confusion_matrix_all.append(confusion_matrix(y_train, y_pred))
            train_multilabel_confusion_matrix_all.append(multilabel_confusion_matrix(y_train, y_pred))
            # d = hinge_loss(y_train, y_pred, labels)
            train_matthews_corrcoef_all.append(matthews_corrcoef(y_train, y_pred))
            # f= roc_auc_score(y_train, y_pred, average='micro', multi_class='ovo')
            train_fbeta_score_micro_all.append(fbeta_score(y_train, y_pred, average='micro', beta=1))
            train_fbeta_score_macro_all.append(fbeta_score(y_train, y_pred, average='macro', beta=1))
            train_fbeta_score_weighted_all.append(fbeta_score(y_train, y_pred, average='weighted', beta=1))
            train_precision_recall_fscore_support_micro_all.append(precision_recall_fscore_support(y_train, y_pred, beta=1, average='micro'))
            train_precision_recall_fscore_support_macro_all.append(precision_recall_fscore_support(y_train, y_pred, beta=1, average='macro'))
            train_precision_recall_fscore_support_weighted_all.append(precision_recall_fscore_support(y_train, y_pred, beta=1, average='weighted'))
            
            # test set 
            y_pred2 = clf.predict(X_test)  
            # multiclass evaluation scores
            test_balanced_accuracy_score_all.append(balanced_accuracy_score(y_test, y_pred2))
            test_confusion_matrix_all.append(confusion_matrix(y_test, y_pred2))
            test_multilabel_confusion_matrix_all.append(multilabel_confusion_matrix(y_test, y_pred2))
            # d2 = hinge_loss(y_test, labels=y_pred)
            test_matthews_corrcoef_all.append(matthews_corrcoef(y_test, y_pred2))
            # f2= roc_auc_score(y_test, y_pred, average='micro', multi_class='ovo')
            test_fbeta_score_micro_all.append(fbeta_score(y_test, y_pred2, average='micro', beta=1))
            test_fbeta_score_macro_all.append(fbeta_score(y_test, y_pred2, average='macro', beta=1))
            test_fbeta_score_weighted_all.append(fbeta_score(y_test, y_pred2, average='weighted', beta=1))
            test_precision_recall_fscore_support_micro_all.append(precision_recall_fscore_support(y_test, y_pred2, beta=1, average='micro'))
            test_precision_recall_fscore_support_macro_all.append(precision_recall_fscore_support(y_test, y_pred2, beta=1, average='macro'))
            test_precision_recall_fscore_support_weighted_all.append(precision_recall_fscore_support(y_test, y_pred2, beta=1, average='weighted'))
    
            params.append(f'solver: {s}, multi_class: {multi}')
    
    # get max score's parameters 
    print('Highest score: ', max(test_balanced_accuracy_score_all))
    print(params[test_balanced_accuracy_score_all.index(max(test_balanced_accuracy_score_all))])
    print('Number of combinations: ', len(solver)*len(multi_class))
    
    max_index = test_balanced_accuracy_score_all.index(max(test_balanced_accuracy_score_all))
    
    train_scores = []
    for i in training: 
        i = i[max_index]
        if type(i) == tuple: 
            prec = i[0]
            recall = i[1]
            f1 = i[2]
            train_scores.append(prec)
            train_scores.append(recall)
            train_scores.append(f1)
        else:  
            train_scores.append(i)
    
    test_scores = []
    for i in testing: 
        i = i[max_index]
        if type(i) == tuple: 
            prec = i[0]
            recall = i[1]
            f1 = i[2]
            test_scores.append(prec)
            test_scores.append(recall)
            test_scores.append(f1)
        else:  
            test_scores.append(i)
    
    score_names =  ['balanced_accuracy', 'matthews_corrcoef', 'fbeta_score_micro', 'fbeta_score_macro', 'fbeta_score_weighted', 'precision_micro', 'recall_micro', 'f1_micro', 'precision_macro', 'precision_macro', 'f1_macro', 'precision_weighted', 'recall_weighted', 'f1_weighted']
    
    assert len(train_scores) == len(score_names)
    assert len(test_scores) == len(score_names)
          
    testdot1_1 = pd.DataFrame({'classifier': clf.__class__.__name__
                             , 'training': train_scores
                            , 'test': test_scores
                            , 'score': score_names})
    return testdot1_1


def cross_validation_logisticregressioncv(X, y, cv=5):
    params = []
    
    solver = ['saga']
    multi_class=['auto', 'ovr', 'multinomial']
    print('Number of combinations: ', len(solver)*len(multi_class))
    
    score_names_all = []
    test_scores_all = []
    train_scores_all = []
    
    for s in solver:  
        for multi in multi_class: 
            clf = LogisticRegressionCV(penalty='elasticnet', solver='saga', random_state=24,  class_weight=None, multi_class=multi, n_jobs = -1, l1_ratios=[.5])   
            scoring = ['balanced_accuracy', 'f1_micro', 'f1_macro', 'f1_weighted', 'precision_micro', 'precision_macro', 'precision_weighted','recall_micro', 'recall_macro', 'recall_weighted', 'roc_auc_ovr', 'roc_auc_ovo', 'roc_auc_ovr_weighted', 'roc_auc_ovo_weighted']
            # train on training set  
            scores = cross_validate(clf, X, y, cv=5, scoring=scoring, return_train_score=True)
            score_names = []
            test_scores = [] 
            train_scores = []        
            for el in sorted(scores.keys()): 
                if el.startswith('test') and not el.endswith('matrix'): 
                    test_scores.append(scores[el].mean())
                    score_names.append(el[5:])
                elif el.startswith('train') and not el.endswith('matrix'): 
                    train_scores.append(scores[el].mean())
            assert len(train_scores) == len(test_scores)
            assert len(train_scores) == len(score_names)
    
            score_names_all.append(score_names)
            test_scores_all.append(test_scores)
            train_scores_all.append(train_scores)
            params.append(f'solver: {s},  multi-class: {multi}')
    
    
    # get max score's parameters 
    for el in test_scores_all: 
        if el[score_names.index('balanced_accuracy')] == max([i[score_names.index('balanced_accuracy')] for i in test_scores_all]): 
            windi = test_scores_all.index(el)
    
    print('Highest score: ', max([i[score_names.index('balanced_accuracy')] for i in test_scores_all]))
    print(params[windi])
    
    train_scores = train_scores_all[score_names.index('balanced_accuracy')]
    test_scores = test_scores_all[score_names.index('balanced_accuracy')]
    score_names =  score_names_all[score_names.index('balanced_accuracy')]
    
    cv5_2 = pd.DataFrame({'classifier': clf.__class__.__name__
                             , 'training': train_scores
                            , 'test': test_scores
                            , 'score': score_names})
    return cv5_2

   
def train_test_split_logisticregressioncv(X_train, y_train):
    params = []
    
    solver = ['saga']
    multi_class=['auto', 'ovr', 'multinomial']
    print('Number of combinations: ', len(solver)*len(multi_class))
    
    train_balanced_accuracy_score_all = []
    train_confusion_matrix_all = []
    train_multilabel_confusion_matrix_all = []
    train_matthews_corrcoef_all = []
    train_fbeta_score_micro_all = []
    train_fbeta_score_macro_all = []
    train_fbeta_score_weighted_all = []
    train_precision_recall_fscore_support_micro_all =[]
    train_precision_recall_fscore_support_macro_all = []
    train_precision_recall_fscore_support_weighted_all = []
    test_balanced_accuracy_score_all= []
    test_confusion_matrix_all= []
    test_multilabel_confusion_matrix_all= []
    test_matthews_corrcoef_all= []
    test_fbeta_score_micro_all= []
    test_fbeta_score_macro_all= []
    test_fbeta_score_weighted_all= []
    test_precision_recall_fscore_support_micro_all= []
    test_precision_recall_fscore_support_macro_all= []
    test_precision_recall_fscore_support_weighted_all= []
    training = [train_balanced_accuracy_score_all, train_matthews_corrcoef_all, train_fbeta_score_micro_all, train_fbeta_score_macro_all, train_fbeta_score_weighted_all, train_precision_recall_fscore_support_micro_all, train_precision_recall_fscore_support_macro_all, train_precision_recall_fscore_support_weighted_all]
    testing = [test_balanced_accuracy_score_all, test_matthews_corrcoef_all, test_fbeta_score_micro_all, test_fbeta_score_macro_all, test_fbeta_score_weighted_all, test_precision_recall_fscore_support_micro_all, test_precision_recall_fscore_support_macro_all, test_precision_recall_fscore_support_weighted_all]
    
    for s in solver:  
            for multi in multi_class: 
                clf = LogisticRegressionCV(penalty='elasticnet', solver='saga', random_state=24,  class_weight=None, multi_class=multi, n_jobs = -1, l1_ratios=[.5])
                # train on training set  
                clf.fit(X_train, y_train)
                # training set 
                y_pred = clf.predict(X_train)    
                # multiclass evaluation scores 
                train_balanced_accuracy_score_all.append(balanced_accuracy_score(y_train, y_pred))
                train_confusion_matrix_all.append(confusion_matrix(y_train, y_pred))
                train_multilabel_confusion_matrix_all.append(multilabel_confusion_matrix(y_train, y_pred))
                # d = hinge_loss(y_train, y_pred, labels)
                train_matthews_corrcoef_all.append(matthews_corrcoef(y_train, y_pred))
                # f= roc_auc_score(y_train, y_pred, average='micro', multi_class='ovo')
                train_fbeta_score_micro_all.append(fbeta_score(y_train, y_pred, average='micro', beta=1))
                train_fbeta_score_macro_all.append(fbeta_score(y_train, y_pred, average='macro', beta=1))
                train_fbeta_score_weighted_all.append(fbeta_score(y_train, y_pred, average='weighted', beta=1))
                train_precision_recall_fscore_support_micro_all.append(precision_recall_fscore_support(y_train, y_pred, beta=1, average='micro'))
                train_precision_recall_fscore_support_macro_all.append(precision_recall_fscore_support(y_train, y_pred, beta=1, average='macro'))
                train_precision_recall_fscore_support_weighted_all.append(precision_recall_fscore_support(y_train, y_pred, beta=1, average='weighted'))
                
                # test set 
                y_pred2 = clf.predict(X_test)  
                # multiclass evaluation scores
                test_balanced_accuracy_score_all.append(balanced_accuracy_score(y_test, y_pred2))
                test_confusion_matrix_all.append(confusion_matrix(y_test, y_pred2))
                test_multilabel_confusion_matrix_all.append(multilabel_confusion_matrix(y_test, y_pred2))
                # d2 = hinge_loss(y_test, labels=y_pred)
                test_matthews_corrcoef_all.append(matthews_corrcoef(y_test, y_pred2))
                # f2= roc_auc_score(y_test, y_pred, average='micro', multi_class='ovo')
                test_fbeta_score_micro_all.append(fbeta_score(y_test, y_pred2, average='micro', beta=1))
                test_fbeta_score_macro_all.append(fbeta_score(y_test, y_pred2, average='macro', beta=1))
                test_fbeta_score_weighted_all.append(fbeta_score(y_test, y_pred2, average='weighted', beta=1))
                test_precision_recall_fscore_support_micro_all.append(precision_recall_fscore_support(y_test, y_pred2, beta=1, average='micro'))
                test_precision_recall_fscore_support_macro_all.append(precision_recall_fscore_support(y_test, y_pred2, beta=1, average='macro'))
                test_precision_recall_fscore_support_weighted_all.append(precision_recall_fscore_support(y_test, y_pred2, beta=1, average='weighted'))
                params.append(f'solver: {s},  multi-class: {multi}')
    
    
    # get max score's parameters 
    print('Highest score: ', max(test_balanced_accuracy_score_all))
    print(params[test_balanced_accuracy_score_all.index(max(test_balanced_accuracy_score_all))])

    max_index = test_balanced_accuracy_score_all.index(max(test_balanced_accuracy_score_all))
    
    train_scores = []
    for i in training: 
        i = i[max_index]
        if type(i) == tuple: 
            prec = i[0]
            recall = i[1]
            f1 = i[2]
            train_scores.append(prec)
            train_scores.append(recall)
            train_scores.append(f1)
        else:  
            train_scores.append(i)
    
    test_scores = []
    for i in testing: 
        i = i[max_index]
        if type(i) == tuple: 
            prec = i[0]
            recall = i[1]
            f1 = i[2]
            test_scores.append(prec)
            test_scores.append(recall)
            test_scores.append(f1)
        else:  
            test_scores.append(i)
    
    score_names =  ['balanced_accuracy', 'matthews_corrcoef', 'fbeta_score_micro', 'fbeta_score_macro', 'fbeta_score_weighted', 'precision_micro', 'recall_micro', 'f1_micro', 'precision_macro', 'precision_macro', 'f1_macro', 'precision_weighted', 'recall_weighted', 'f1_weighted']
    
    
    assert len(train_scores) == len(score_names)
    assert len(test_scores) == len(score_names)
          
    testdot1_2 = pd.DataFrame({'classifier': clf.__class__.__name__
                             , 'training': train_scores
                            , 'test': test_scores
                            , 'score': score_names})
    return testdot1_2


def train_test_split_ridgeclassifier(X_train, y_train): 
    params = []
    
    alphas = [100000, 50000, 10000, 5000]
    print('Number of combinations: ', len(alphas))
    
    
    train_balanced_accuracy_score_all = []
    train_confusion_matrix_all = []
    train_multilabel_confusion_matrix_all = []
    train_matthews_corrcoef_all = []
    train_fbeta_score_micro_all = []
    train_fbeta_score_macro_all = []
    train_fbeta_score_weighted_all = []
    train_precision_recall_fscore_support_micro_all =[]
    train_precision_recall_fscore_support_macro_all = []
    train_precision_recall_fscore_support_weighted_all = []
    test_balanced_accuracy_score_all= []
    test_confusion_matrix_all= []
    test_multilabel_confusion_matrix_all= []
    test_matthews_corrcoef_all= []
    test_fbeta_score_micro_all= []
    test_fbeta_score_macro_all= []
    test_fbeta_score_weighted_all= []
    test_precision_recall_fscore_support_micro_all= []
    test_precision_recall_fscore_support_macro_all= []
    test_precision_recall_fscore_support_weighted_all= []
    training = [train_balanced_accuracy_score_all, train_matthews_corrcoef_all, train_fbeta_score_micro_all, train_fbeta_score_macro_all, train_fbeta_score_weighted_all, train_precision_recall_fscore_support_micro_all, train_precision_recall_fscore_support_macro_all, train_precision_recall_fscore_support_weighted_all]
    testing = [test_balanced_accuracy_score_all, test_matthews_corrcoef_all, test_fbeta_score_micro_all, test_fbeta_score_macro_all, test_fbeta_score_weighted_all, test_precision_recall_fscore_support_micro_all, test_precision_recall_fscore_support_macro_all, test_precision_recall_fscore_support_weighted_all]
    
    
    for a in alphas:  
        clf = RidgeClassifier(alpha=a, max_iter=110, class_weight='balanced', random_state=24)
        # train on training set  
        clf.fit(X_train, y_train)
        # training set 
        y_pred = clf.predict(X_train)    
        # multiclass evaluation scores 
        train_balanced_accuracy_score_all.append(balanced_accuracy_score(y_train, y_pred))
        train_confusion_matrix_all.append(confusion_matrix(y_train, y_pred))
        train_multilabel_confusion_matrix_all.append(multilabel_confusion_matrix(y_train, y_pred))
        # d = hinge_loss(y_train, y_pred, labels)
        train_matthews_corrcoef_all.append(matthews_corrcoef(y_train, y_pred))
        # f= roc_auc_score(y_train, y_pred, average='micro', multi_class='ovo')
        train_fbeta_score_micro_all.append(fbeta_score(y_train, y_pred, average='micro', beta=1))
        train_fbeta_score_macro_all.append(fbeta_score(y_train, y_pred, average='macro', beta=1))
        train_fbeta_score_weighted_all.append(fbeta_score(y_train, y_pred, average='weighted', beta=1))
        train_precision_recall_fscore_support_micro_all.append(precision_recall_fscore_support(y_train, y_pred, beta=1, average='micro'))
        train_precision_recall_fscore_support_macro_all.append(precision_recall_fscore_support(y_train, y_pred, beta=1, average='macro'))
        train_precision_recall_fscore_support_weighted_all.append(precision_recall_fscore_support(y_train, y_pred, beta=1, average='weighted'))
        
        # test set 
        y_pred2 = clf.predict(X_test)  
        # multiclass evaluation scores
        test_balanced_accuracy_score_all.append(balanced_accuracy_score(y_test, y_pred2))
        test_confusion_matrix_all.append(confusion_matrix(y_test, y_pred2))
        test_multilabel_confusion_matrix_all.append(multilabel_confusion_matrix(y_test, y_pred2))
        # d2 = hinge_loss(y_test, labels=y_pred)
        test_matthews_corrcoef_all.append(matthews_corrcoef(y_test, y_pred2))
        # f2= roc_auc_score(y_test, y_pred, average='micro', multi_class='ovo')
        test_fbeta_score_micro_all.append(fbeta_score(y_test, y_pred2, average='micro', beta=1))
        test_fbeta_score_macro_all.append(fbeta_score(y_test, y_pred2, average='macro', beta=1))
        test_fbeta_score_weighted_all.append(fbeta_score(y_test, y_pred2, average='weighted', beta=1))
        test_precision_recall_fscore_support_micro_all.append(precision_recall_fscore_support(y_test, y_pred2, beta=1, average='micro'))
        test_precision_recall_fscore_support_macro_all.append(precision_recall_fscore_support(y_test, y_pred2, beta=1, average='macro'))
        test_precision_recall_fscore_support_weighted_all.append(precision_recall_fscore_support(y_test, y_pred2, beta=1, average='weighted'))  
        params.append(f'alpha: {a}')
    
    
    # get max score's parameters 
    print('Highest score: ', max(test_balanced_accuracy_score_all))
    print(params[test_balanced_accuracy_score_all.index(max(test_balanced_accuracy_score_all))])
    
    max_index = test_balanced_accuracy_score_all.index(max(test_balanced_accuracy_score_all))
    
    train_scores = []
    for i in training: 
        i = i[max_index]
        if type(i) == tuple: 
            prec = i[0]
            recall = i[1]
            f1 = i[2]
            train_scores.append(prec)
            train_scores.append(recall)
            train_scores.append(f1)
        else:  
            train_scores.append(i)
    
    test_scores = []
    for i in testing: 
        i = i[max_index]
        if type(i) == tuple: 
            prec = i[0]
            recall = i[1]
            f1 = i[2]
            test_scores.append(prec)
            test_scores.append(recall)
            test_scores.append(f1)
        else:  
            test_scores.append(i)
    
    score_names =  ['balanced_accuracy', 'matthews_corrcoef', 'fbeta_score_micro', 'fbeta_score_macro', 'fbeta_score_weighted', 'precision_micro', 'recall_micro', 'f1_micro', 'precision_macro', 'precision_macro', 'f1_macro', 'precision_weighted', 'recall_weighted', 'f1_weighted']
    
    
    assert len(train_scores) == len(score_names)
    assert len(test_scores) == len(score_names)
          
    testdot1_3 = pd.DataFrame({'classifier': clf.__class__.__name__
                             , 'training': train_scores
                            , 'test': test_scores
                            , 'score': score_names})
    return testdot1_3

   
def train_test_split_ridgeclassifiercv(X_train, y_train):
    params = []
    
    class_weights = [None, 'balanced']
    print('Number of combinations: ', len(class_weights))
    
    train_balanced_accuracy_score_all = []
    train_confusion_matrix_all = []
    train_multilabel_confusion_matrix_all = []
    train_matthews_corrcoef_all = []
    train_fbeta_score_micro_all = []
    train_fbeta_score_macro_all = []
    train_fbeta_score_weighted_all = []
    train_precision_recall_fscore_support_micro_all =[]
    train_precision_recall_fscore_support_macro_all = []
    train_precision_recall_fscore_support_weighted_all = []
    test_balanced_accuracy_score_all= []
    test_confusion_matrix_all= []
    test_multilabel_confusion_matrix_all= []
    test_matthews_corrcoef_all= []
    test_fbeta_score_micro_all= []
    test_fbeta_score_macro_all= []
    test_fbeta_score_weighted_all= []
    test_precision_recall_fscore_support_micro_all= []
    test_precision_recall_fscore_support_macro_all= []
    test_precision_recall_fscore_support_weighted_all= []
    training = [train_balanced_accuracy_score_all, train_matthews_corrcoef_all, train_fbeta_score_micro_all, train_fbeta_score_macro_all, train_fbeta_score_weighted_all, train_precision_recall_fscore_support_micro_all, train_precision_recall_fscore_support_macro_all, train_precision_recall_fscore_support_weighted_all]
    testing = [test_balanced_accuracy_score_all, test_matthews_corrcoef_all, test_fbeta_score_micro_all, test_fbeta_score_macro_all, test_fbeta_score_weighted_all, test_precision_recall_fscore_support_micro_all, test_precision_recall_fscore_support_macro_all, test_precision_recall_fscore_support_weighted_all]
       
    for cw in class_weights:
            clf = RidgeClassifierCV(cv=5, class_weight=cw)
            # train on training set  
            clf.fit(X_train, y_train)
            # training set 
            y_pred = clf.predict(X_train)    
            # multiclass evaluation scores 
            train_balanced_accuracy_score_all.append(balanced_accuracy_score(y_train, y_pred))
            train_confusion_matrix_all.append(confusion_matrix(y_train, y_pred))
            train_multilabel_confusion_matrix_all.append(multilabel_confusion_matrix(y_train, y_pred))
            # d = hinge_loss(y_train, y_pred, labels)
            train_matthews_corrcoef_all.append(matthews_corrcoef(y_train, y_pred))
            # f= roc_auc_score(y_train, y_pred, average='micro', multi_class='ovo')
            train_fbeta_score_micro_all.append(fbeta_score(y_train, y_pred, average='micro', beta=1))
            train_fbeta_score_macro_all.append(fbeta_score(y_train, y_pred, average='macro', beta=1))
            train_fbeta_score_weighted_all.append(fbeta_score(y_train, y_pred, average='weighted', beta=1))
            train_precision_recall_fscore_support_micro_all.append(precision_recall_fscore_support(y_train, y_pred, beta=1, average='micro'))
            train_precision_recall_fscore_support_macro_all.append(precision_recall_fscore_support(y_train, y_pred, beta=1, average='macro'))
            train_precision_recall_fscore_support_weighted_all.append(precision_recall_fscore_support(y_train, y_pred, beta=1, average='weighted'))
            
            # test set 
            y_pred2 = clf.predict(X_test)  
            # multiclass evaluation scores
            test_balanced_accuracy_score_all.append(balanced_accuracy_score(y_test, y_pred2))
            test_confusion_matrix_all.append(confusion_matrix(y_test, y_pred2))
            test_multilabel_confusion_matrix_all.append(multilabel_confusion_matrix(y_test, y_pred2))
            # d2 = hinge_loss(y_test, labels=y_pred)
            test_matthews_corrcoef_all.append(matthews_corrcoef(y_test, y_pred2))
            # f2= roc_auc_score(y_test, y_pred, average='micro', multi_class='ovo')
            test_fbeta_score_micro_all.append(fbeta_score(y_test, y_pred2, average='micro', beta=1))
            test_fbeta_score_macro_all.append(fbeta_score(y_test, y_pred2, average='macro', beta=1))
            test_fbeta_score_weighted_all.append(fbeta_score(y_test, y_pred2, average='weighted', beta=1))
            test_precision_recall_fscore_support_micro_all.append(precision_recall_fscore_support(y_test, y_pred2, beta=1, average='micro'))
            test_precision_recall_fscore_support_macro_all.append(precision_recall_fscore_support(y_test, y_pred2, beta=1, average='macro'))
            test_precision_recall_fscore_support_weighted_all.append(precision_recall_fscore_support(y_test, y_pred2, beta=1, average='weighted'))  
            params.append(f'class_weights: {cw}')
        
    # get max score's parameters 
    print('Highest score: ', max(test_balanced_accuracy_score_all))
    print(params[test_balanced_accuracy_score_all.index(max(test_balanced_accuracy_score_all))])
    
    max_index = test_balanced_accuracy_score_all.index(max(test_balanced_accuracy_score_all))
    
    train_scores = []
    for i in training: 
        i = i[max_index]
        if type(i) == tuple: 
            prec = i[0]
            recall = i[1]
            f1 = i[2]
            train_scores.append(prec)
            train_scores.append(recall)
            train_scores.append(f1)
        else:  
            train_scores.append(i)
    
    test_scores = []
    for i in testing: 
        i = i[max_index]
        if type(i) == tuple: 
            prec = i[0]
            recall = i[1]
            f1 = i[2]
            test_scores.append(prec)
            test_scores.append(recall)
            test_scores.append(f1)
        else:  
            test_scores.append(i)
    
    score_names =  ['balanced_accuracy', 'matthews_corrcoef', 'fbeta_score_micro', 'fbeta_score_macro', 'fbeta_score_weighted', 'precision_micro', 'recall_micro', 'f1_micro', 'precision_macro', 'precision_macro', 'f1_macro', 'precision_weighted', 'recall_weighted', 'f1_weighted']
    
    
    assert len(train_scores) == len(score_names)
    assert len(test_scores) == len(score_names)
          
    testdot1_4 = pd.DataFrame({'classifier': clf.__class__.__name__
                             , 'training': train_scores
                            , 'test': test_scores
                            , 'score': score_names})
    return testdot1_4


def cross_validation_gaussiannb(X, y, cv=5):       
    clf = GaussianNB()
    scoring = ['balanced_accuracy', 'f1_micro', 'f1_macro', 'f1_weighted', 'precision_micro', 'precision_macro', 'precision_weighted','recall_micro', 'recall_macro', 'recall_weighted', 'roc_auc_ovr', 'roc_auc_ovo', 'roc_auc_ovr_weighted', 'roc_auc_ovo_weighted']
    scores = cross_validate(clf, X, y, cv=5, scoring=scoring, return_train_score=True)
    key_scores = sorted(scores.keys())[2:]
    for i, score in enumerate(key_scores): 
        print(f"{key_scores[i]}: %0.2f (+/- %0.2f)" % (scores[key_scores[i]].mean(), scores[key_scores[i]].std() * 2))
    score_names = []
    test_scores = [] 
    train_scores = []  
      
    for el in sorted(scores.keys()): 
        if el.startswith('test') and not el.endswith('matrix'): 
            test_scores.append(scores[el].mean())
            score_names.append(el[5:])
        elif el.startswith('train') and not el.endswith('matrix'): 
            train_scores.append(scores[el].mean())
                   
    assert len(train_scores) == len(test_scores)
    assert len(train_scores) == len(score_names)
    
    cv5_5 = pd.DataFrame({'classifier': clf.__class__.__name__
                             , 'training': train_scores
                            , 'test': test_scores
                            , 'score': score_names})
    return cv5_5


def train_test_split_gaussiannb(X_train, y_train):
    multiclass_scores = {}
 
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_train)    
    a = balanced_accuracy_score(y_train, y_pred)
    b= confusion_matrix(y_train, y_pred)
    c= multilabel_confusion_matrix(y_train, y_pred)
    e= matthews_corrcoef(y_train, y_pred)
    g= fbeta_score(y_train, y_pred, average='micro', beta=1)
    h= fbeta_score(y_train, y_pred, average='macro', beta=1)
    i = fbeta_score(y_train, y_pred, average='weighted', beta=1)
    j= precision_recall_fscore_support(y_train, y_pred, beta=1, average='micro')
    k= precision_recall_fscore_support(y_train, y_pred, beta=1, average='macro')
    l= precision_recall_fscore_support(y_train, y_pred, beta=1, average='weighted')
    
    multiclass_scores['train_balanced_accuracy_score'] = a
    multiclass_scores['train_confusion_matrix'] = b
    multiclass_scores['train_multilabel_confusion_matrix'] = c
    multiclass_scores['train_matthews_corrcoef'] = e
    multiclass_scores['train_fbeta_score_micro'] = g
    multiclass_scores['train_fbeta_score_macro'] = h
    multiclass_scores['train_fbeta_score_weighted'] = i
    multiclass_scores['train_precision_recall_fscore_support_micro'] = j
    multiclass_scores['train_precision_recall_fscore_support_macro'] = k
    multiclass_scores['train_precision_recall_fscore_support_weighted'] = l
      
    # test set 
    y_pred2 = clf.predict(X_test)    
    a2 = balanced_accuracy_score(y_test, y_pred2)
    b2= confusion_matrix(y_test, y_pred2)
    c2= multilabel_confusion_matrix(y_test, y_pred2)
    e2= matthews_corrcoef(y_test, y_pred2)
    g2= fbeta_score(y_test, y_pred2, average='micro', beta=1)
    h2= fbeta_score(y_test, y_pred2, average='macro', beta=1)
    i2 = fbeta_score(y_test, y_pred2, average='weighted', beta=1)
    j2= precision_recall_fscore_support(y_test, y_pred2, beta=1, average='micro')
    k2= precision_recall_fscore_support(y_test, y_pred2, beta=1, average='macro')
    l2= precision_recall_fscore_support(y_test, y_pred2, beta=1, average='weighted')
        
    multiclass_scores['test_balanced_accuracy_score'] = a2
    multiclass_scores['test_confusion_matrix'] = b2
    multiclass_scores['test_multilabel_confusion_matrix'] = c2
    multiclass_scores['test_matthews_corrcoef'] = e2
    multiclass_scores['test_fbeta_score_micro'] = g2
    multiclass_scores['test_fbeta_score_macro'] = h2
    multiclass_scores['test_fbeta_score_weighted'] = i2
    multiclass_scores['test_precision_recall_fscore_support_micro'] = j2
    multiclass_scores['test_precision_recall_fscore_support_macro'] = k2
    multiclass_scores['test_precision_recall_fscore_support_weighted'] = l2
      
    score_names = []
    test_scores = [] 
    train_scores = []        
    for el in sorted(multiclass_scores.keys()): 
        if el.startswith('test') and not el.endswith('matrix'): 
            if el.startswith('test_precision_recall_fscore'): 
                prec, rec, fscore, nan = multiclass_scores[el]
                test_scores.append(prec)
                test_scores.append(rec)
                test_scores.append(fscore)
                if el.endswith('weighted'): 
                    score_names.append(str(el[5:14])+'_'+str(el[-8:]))
                    score_names.append(str(el[15:21])+'_'+str(el[-8:]))
                    score_names.append(str(el[22:28])+'_'+str(el[-8:]))
                else: 
                    score_names.append(str(el[5:14])+'_'+str(el[-5:]))
                    score_names.append(str(el[15:21])+'_'+str(el[-5:]))
                    score_names.append(str(el[22:28])+'_'+str(el[-5:]))                
            else: 
                test_scores.append(multiclass_scores[el])
                score_names.append(el[5:])
        elif el.startswith('train') and not el.endswith('matrix'): 
            if el.startswith('train_precision_recall_fscore'): 
                prec, rec, fscore, nan = multiclass_scores[el]
                train_scores.append(prec)
                train_scores.append(rec)
                train_scores.append(fscore)
            else: 
                train_scores.append(multiclass_scores[el])
                
    assert len(train_scores) == len(test_scores)
    assert len(train_scores) == len(score_names)
            
    testdot1_5 = pd.DataFrame({'classifier': clf.__class__.__name__
                             , 'training': train_scores
                            , 'test': test_scores
                            , 'score': score_names})
    return testdot1_5, clf


def cross_validation_knn(X, y, cv=5):
    params = []
    
    n_neighbors = np.arange(1,31).tolist()
    leaf_sizes = np.arange(1,31).tolist()
    print('Number of combinations: ', len(n_neighbors)*len(leaf_sizes))
    
    score_names_all = []
    test_scores_all = []
    train_scores_all = []
    
    for n in n_neighbors: 
        for l in leaf_sizes: 
            clf = KNeighborsClassifier(n_neighbors=n, weights='distance', leaf_size=l, n_jobs=-1)
            scoring = ['balanced_accuracy', 'f1_micro', 'f1_macro', 'f1_weighted', 'precision_micro', 'precision_macro', 'precision_weighted','recall_micro', 'recall_macro', 'recall_weighted']
            # train on training set  
            scores = cross_validate(clf, X, y, cv=5, scoring=scoring, return_train_score=True)
            score_names = []
            test_scores = [] 
            train_scores = []        
            for el in sorted(scores.keys()): 
                if el.startswith('test') and not el.endswith('matrix'): 
                    test_scores.append(scores[el].mean())
                    score_names.append(el[5:])
                elif el.startswith('train') and not el.endswith('matrix'): 
                    train_scores.append(scores[el].mean())
            assert len(train_scores) == len(test_scores)
            assert len(train_scores) == len(score_names)
        
            score_names_all.append(score_names)
            test_scores_all.append(test_scores)
            train_scores_all.append(train_scores)   
            params.append(f'n_neighbors: {n}, leaf_sizes: {l}')
       
    # get max score's parameters 
    for el in test_scores_all: 
        if el[score_names.index('balanced_accuracy')] == max([i[score_names.index('balanced_accuracy')] for i in test_scores_all]): 
            windi = test_scores_all.index(el)
    
    print('Highest score: ', max([i[score_names.index('balanced_accuracy')] for i in test_scores_all]))
    print(params[windi])
    
    train_scores = train_scores_all[score_names.index('balanced_accuracy')]
    test_scores = test_scores_all[score_names.index('balanced_accuracy')]
    score_names =  score_names_all[score_names.index('balanced_accuracy')]
    
    cv5_6 = pd.DataFrame({'classifier': clf.__class__.__name__
                             , 'training': train_scores
                            , 'test': test_scores
                            , 'score': score_names})
    return cv5_6


def train_test_split_knn(X_train, y_train):
    params = []
    
    n_neighbors = np.arange(1,31).tolist()
    leaf_sizes = np.arange(1,31).tolist()
    print('Number of combinations: ', len(n_neighbors)*len(leaf_sizes))
       
    # train-test split
    train_balanced_accuracy_score_all = []
    train_confusion_matrix_all = []
    train_multilabel_confusion_matrix_all = []
    train_matthews_corrcoef_all = []
    train_fbeta_score_micro_all = []
    train_fbeta_score_macro_all = []
    train_fbeta_score_weighted_all = []
    train_precision_recall_fscore_support_micro_all =[]
    train_precision_recall_fscore_support_macro_all = []
    train_precision_recall_fscore_support_weighted_all = []
    test_balanced_accuracy_score_all= []
    test_confusion_matrix_all= []
    test_multilabel_confusion_matrix_all= []
    test_matthews_corrcoef_all= []
    test_fbeta_score_micro_all= []
    test_fbeta_score_macro_all= []
    test_fbeta_score_weighted_all= []
    test_precision_recall_fscore_support_micro_all= []
    test_precision_recall_fscore_support_macro_all= []
    test_precision_recall_fscore_support_weighted_all= []
    training = [train_balanced_accuracy_score_all, train_matthews_corrcoef_all, train_fbeta_score_micro_all, train_fbeta_score_macro_all, train_fbeta_score_weighted_all, train_precision_recall_fscore_support_micro_all, train_precision_recall_fscore_support_macro_all, train_precision_recall_fscore_support_weighted_all]
    testing = [test_balanced_accuracy_score_all, test_matthews_corrcoef_all, test_fbeta_score_micro_all, test_fbeta_score_macro_all, test_fbeta_score_weighted_all, test_precision_recall_fscore_support_micro_all, test_precision_recall_fscore_support_macro_all, test_precision_recall_fscore_support_weighted_all]
  
    for n in n_neighbors: 
        for l in leaf_sizes: 
            clf = KNeighborsClassifier(n_neighbors=n, weights='distance', leaf_size=l, n_jobs=-1) 
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_train)    
            train_balanced_accuracy_score_all.append(balanced_accuracy_score(y_train, y_pred))
            train_confusion_matrix_all.append(confusion_matrix(y_train, y_pred))
            train_multilabel_confusion_matrix_all.append(multilabel_confusion_matrix(y_train, y_pred))
            train_matthews_corrcoef_all.append(matthews_corrcoef(y_train, y_pred))
            train_fbeta_score_micro_all.append(fbeta_score(y_train, y_pred, average='micro', beta=1))
            train_fbeta_score_macro_all.append(fbeta_score(y_train, y_pred, average='macro', beta=1))
            train_fbeta_score_weighted_all.append(fbeta_score(y_train, y_pred, average='weighted', beta=1))
            train_precision_recall_fscore_support_micro_all.append(precision_recall_fscore_support(y_train, y_pred, beta=1, average='micro'))
            train_precision_recall_fscore_support_macro_all.append(precision_recall_fscore_support(y_train, y_pred, beta=1, average='macro'))
            train_precision_recall_fscore_support_weighted_all.append(precision_recall_fscore_support(y_train, y_pred, beta=1, average='weighted'))
            
            # test set 
            y_pred2 = clf.predict(X_test)  
            test_balanced_accuracy_score_all.append(balanced_accuracy_score(y_test, y_pred2))
            test_confusion_matrix_all.append(confusion_matrix(y_test, y_pred2))
            test_multilabel_confusion_matrix_all.append(multilabel_confusion_matrix(y_test, y_pred2))
            test_matthews_corrcoef_all.append(matthews_corrcoef(y_test, y_pred2))
            test_fbeta_score_micro_all.append(fbeta_score(y_test, y_pred2, average='micro', beta=1))
            test_fbeta_score_macro_all.append(fbeta_score(y_test, y_pred2, average='macro', beta=1))
            test_fbeta_score_weighted_all.append(fbeta_score(y_test, y_pred2, average='weighted', beta=1))
            test_precision_recall_fscore_support_micro_all.append(precision_recall_fscore_support(y_test, y_pred2, beta=1, average='micro'))
            test_precision_recall_fscore_support_macro_all.append(precision_recall_fscore_support(y_test, y_pred2, beta=1, average='macro'))
            test_precision_recall_fscore_support_weighted_all.append(precision_recall_fscore_support(y_test, y_pred2, beta=1, average='weighted'))    
            params.append(f'n_neighbors: {n}, leaf_sizes: {l}')
        
    
    # get max score's parameters 
    print('Highest score: ', max(test_balanced_accuracy_score_all))
    print(params[test_balanced_accuracy_score_all.index(max(test_balanced_accuracy_score_all))])   
    max_index = test_balanced_accuracy_score_all.index(max(test_balanced_accuracy_score_all))
    
    train_scores = []
    for i in training: 
        i = i[max_index]
        if type(i) == tuple: 
            prec = i[0]
            recall = i[1]
            f1 = i[2]
            train_scores.append(prec)
            train_scores.append(recall)
            train_scores.append(f1)
        else:  
            train_scores.append(i)
    
    test_scores = []
    for i in testing: 
        i = i[max_index]
        if type(i) == tuple: 
            prec = i[0]
            recall = i[1]
            f1 = i[2]
            test_scores.append(prec)
            test_scores.append(recall)
            test_scores.append(f1)
        else:  
            test_scores.append(i)
    
    score_names =  ['balanced_accuracy', 'matthews_corrcoef', 'fbeta_score_micro', 'fbeta_score_macro', 'fbeta_score_weighted', 'precision_micro', 'recall_micro', 'f1_micro', 'precision_macro', 'precision_macro', 'f1_macro', 'precision_weighted', 'recall_weighted', 'f1_weighted']
        
    assert len(train_scores) == len(score_names)
    assert len(test_scores) == len(score_names)
          
    testdot1_6 = pd.DataFrame({'classifier': clf.__class__.__name__
                             , 'training': train_scores
                            , 'test': test_scores
                            , 'score': score_names})
    return testdot1_6

def cross_validation_nearestcentroid(X, y, cv=5):
    
    params = []
     
    score_names_all = []
    test_scores_all = []
    train_scores_all = []
    
    shrink_threshold = [6,5,4, 4.5, 3, 3.5, 2,1]
    print('Number of combinations: ', len(shrink_threshold))
    
    for sk in shrink_threshold: 
        clf = NearestCentroid(shrink_threshold=sk)
        scoring = ['balanced_accuracy', 'f1_micro', 'f1_macro', 'f1_weighted', 'precision_micro', 'precision_macro', 'precision_weighted','recall_micro', 'recall_macro', 'recall_weighted', 'roc_auc_ovr', 'roc_auc_ovo', 'roc_auc_ovr_weighted', 'roc_auc_ovo_weighted']
        # train on training set  
        scores = cross_validate(clf, X, y, cv=5, scoring=scoring, return_train_score=True)
        score_names = []
        test_scores = [] 
        train_scores = []        
        for el in sorted(scores.keys()): 
            if el.startswith('test') and not el.endswith('matrix'): 
                test_scores.append(scores[el].mean())
                score_names.append(el[5:])
            elif el.startswith('train') and not el.endswith('matrix'): 
                train_scores.append(scores[el].mean())
        assert len(train_scores) == len(test_scores)
        assert len(train_scores) == len(score_names)
    
        score_names_all.append(score_names)
        test_scores_all.append(test_scores)
        train_scores_all.append(train_scores)
        params.append(f'shrink_threshold: {s}')
        
    # get max score's parameters 
    for el in test_scores_all: 
        if el[score_names.index('balanced_accuracy')] == max([i[score_names.index('balanced_accuracy')] for i in test_scores_all]): 
            windi = test_scores_all.index(el)
    
    print('Highest score: ', max([i[score_names.index('balanced_accuracy')] for i in test_scores_all]))
    print(params[windi])
    
    train_scores = train_scores_all[score_names.index('balanced_accuracy')]
    test_scores = test_scores_all[score_names.index('balanced_accuracy')]
    score_names =  score_names_all[score_names.index('balanced_accuracy')]
    
    cv5_7 = pd.DataFrame({'classifier': clf.__class__.__name__
                             , 'training': train_scores
                            , 'test': test_scores
                            , 'score': score_names})
    return cv5_7


def train_test_split_nearestcentroid(X_train, y_train):
    params = []

    shrink_threshold = [6,5,4, 4.5, 3, 3.5, 2,1]
    print('Number of combinations: ', len(shrink_threshold))
    
    # train-test split
    train_balanced_accuracy_score_all = []
    train_confusion_matrix_all = []
    train_multilabel_confusion_matrix_all = []
    train_matthews_corrcoef_all = []
    train_fbeta_score_micro_all = []
    train_fbeta_score_macro_all = []
    train_fbeta_score_weighted_all = []
    train_precision_recall_fscore_support_micro_all =[]
    train_precision_recall_fscore_support_macro_all = []
    train_precision_recall_fscore_support_weighted_all = []
    test_balanced_accuracy_score_all= []
    test_confusion_matrix_all= []
    test_multilabel_confusion_matrix_all= []
    test_matthews_corrcoef_all= []
    test_fbeta_score_micro_all= []
    test_fbeta_score_macro_all= []
    test_fbeta_score_weighted_all= []
    test_precision_recall_fscore_support_micro_all= []
    test_precision_recall_fscore_support_macro_all= []
    test_precision_recall_fscore_support_weighted_all= []
    training = [train_balanced_accuracy_score_all, train_matthews_corrcoef_all, train_fbeta_score_micro_all, train_fbeta_score_macro_all, train_fbeta_score_weighted_all, train_precision_recall_fscore_support_micro_all, train_precision_recall_fscore_support_macro_all, train_precision_recall_fscore_support_weighted_all]
    testing = [test_balanced_accuracy_score_all, test_matthews_corrcoef_all, test_fbeta_score_micro_all, test_fbeta_score_macro_all, test_fbeta_score_weighted_all, test_precision_recall_fscore_support_micro_all, test_precision_recall_fscore_support_macro_all, test_precision_recall_fscore_support_weighted_all]
    
       
    for sk in shrink_threshold: 
        clf = NearestCentroid(shrink_threshold=sk)
        # train on training set  
        clf.fit(X_train, y_train)
        # training set 
        y_pred = clf.predict(X_train)    
        # multiclass evaluation scores 
        train_balanced_accuracy_score_all.append(balanced_accuracy_score(y_train, y_pred))
        train_confusion_matrix_all.append(confusion_matrix(y_train, y_pred))
        train_multilabel_confusion_matrix_all.append(multilabel_confusion_matrix(y_train, y_pred))
        # d = hinge_loss(y_train, y_pred, labels)
        train_matthews_corrcoef_all.append(matthews_corrcoef(y_train, y_pred))
        # f= roc_auc_score(y_train, y_pred, average='micro', multi_class='ovo')
        train_fbeta_score_micro_all.append(fbeta_score(y_train, y_pred, average='micro', beta=1))
        train_fbeta_score_macro_all.append(fbeta_score(y_train, y_pred, average='macro', beta=1))
        train_fbeta_score_weighted_all.append(fbeta_score(y_train, y_pred, average='weighted', beta=1))
        train_precision_recall_fscore_support_micro_all.append(precision_recall_fscore_support(y_train, y_pred, beta=1, average='micro'))
        train_precision_recall_fscore_support_macro_all.append(precision_recall_fscore_support(y_train, y_pred, beta=1, average='macro'))
        train_precision_recall_fscore_support_weighted_all.append(precision_recall_fscore_support(y_train, y_pred, beta=1, average='weighted'))
        
        # test set 
        y_pred2 = clf.predict(X_test)  
        # multiclass evaluation scores
        test_balanced_accuracy_score_all.append(balanced_accuracy_score(y_test, y_pred2))
        test_confusion_matrix_all.append(confusion_matrix(y_test, y_pred2))
        test_multilabel_confusion_matrix_all.append(multilabel_confusion_matrix(y_test, y_pred2))
        # d2 = hinge_loss(y_test, labels=y_pred)
        test_matthews_corrcoef_all.append(matthews_corrcoef(y_test, y_pred2))
        # f2= roc_auc_score(y_test, y_pred, average='micro', multi_class='ovo')
        test_fbeta_score_micro_all.append(fbeta_score(y_test, y_pred2, average='micro', beta=1))
        test_fbeta_score_macro_all.append(fbeta_score(y_test, y_pred2, average='macro', beta=1))
        test_fbeta_score_weighted_all.append(fbeta_score(y_test, y_pred2, average='weighted', beta=1))
        test_precision_recall_fscore_support_micro_all.append(precision_recall_fscore_support(y_test, y_pred2, beta=1, average='micro'))
        test_precision_recall_fscore_support_macro_all.append(precision_recall_fscore_support(y_test, y_pred2, beta=1, average='macro'))
        test_precision_recall_fscore_support_weighted_all.append(precision_recall_fscore_support(y_test, y_pred2, beta=1, average='weighted'))    
       
        params.append(f'shrink_threshold: {sk}')
           
    # get max score's parameters 
    print('Highest score: ', max(test_balanced_accuracy_score_all))
    print(params[test_balanced_accuracy_score_all.index(max(test_balanced_accuracy_score_all))])

    max_index = test_balanced_accuracy_score_all.index(max(test_balanced_accuracy_score_all))
    
    train_scores = []
    for i in training: 
        i = i[max_index]
        if type(i) == tuple: 
            prec = i[0]
            recall = i[1]
            f1 = i[2]
            train_scores.append(prec)
            train_scores.append(recall)
            train_scores.append(f1)
        else:  
            train_scores.append(i)
    
    test_scores = []
    for i in testing: 
        i = i[max_index]
        if type(i) == tuple: 
            prec = i[0]
            recall = i[1]
            f1 = i[2]
            test_scores.append(prec)
            test_scores.append(recall)
            test_scores.append(f1)
        else:  
            test_scores.append(i)
    
    score_names =  ['balanced_accuracy', 'matthews_corrcoef', 'fbeta_score_micro', 'fbeta_score_macro', 'fbeta_score_weighted', 'precision_micro', 'recall_micro', 'f1_micro', 'precision_macro', 'precision_macro', 'f1_macro', 'precision_weighted', 'recall_weighted', 'f1_weighted']
    
    
    assert len(train_scores) == len(score_names)
    assert len(test_scores) == len(score_names)
          
    testdot1_7 = pd.DataFrame({'classifier': clf.__class__.__name__
                             , 'training': train_scores
                            , 'test': test_scores
                            , 'score': score_names})
    return testdot1_7
    

def cross_validation_radiusneighborsclassifier(X, y, cv=5): 
    params = []
    
    score_names_all = []
    test_scores_all = []
    train_scores_all = []
    
    radius = [300, 200, 100]
    leaf_size = [100, 50, 15]
    print('Number of combinations: ', len(radius)*len(leaf_size))
    
    for r in radius:
        for l in leaf_size: 
            clf = RadiusNeighborsClassifier(radius=r, weights='distance', leaf_size=l, n_jobs=-1)
            scoring = ['balanced_accuracy', 'f1_micro', 'f1_macro', 'f1_weighted', 'precision_micro', 'precision_macro', 'precision_weighted','recall_micro', 'recall_macro', 'recall_weighted', 'roc_auc_ovr', 'roc_auc_ovo', 'roc_auc_ovr_weighted', 'roc_auc_ovo_weighted']
            # train on training set  
            scores = cross_validate(clf, X, y, cv=5, scoring=scoring, return_train_score=True)
            score_names = []
            test_scores = [] 
            train_scores = []        
            for el in sorted(scores.keys()): 
                if el.startswith('test') and not el.endswith('matrix'): 
                    test_scores.append(scores[el].mean())
                    score_names.append(el[5:])
                elif el.startswith('train') and not el.endswith('matrix'): 
                    train_scores.append(scores[el].mean())
            assert len(train_scores) == len(test_scores)
            assert len(train_scores) == len(score_names)
        
            score_names_all.append(score_names)
            test_scores_all.append(test_scores)
            train_scores_all.append(train_scores)
            params.append(f'radius: {r}, leaf_size: {l}')
    
    # get max score's parameters 
    for el in test_scores_all: 
        if el[score_names.index('balanced_accuracy')] == max([i[score_names.index('balanced_accuracy')] for i in test_scores_all]): 
            windi = test_scores_all.index(el)
    
    print('Highest score: ', max([i[score_names.index('balanced_accuracy')] for i in test_scores_all]))
    print(params[windi])
    
    train_scores = train_scores_all[score_names.index('balanced_accuracy')]
    test_scores = test_scores_all[score_names.index('balanced_accuracy')]
    score_names =  score_names_all[score_names.index('balanced_accuracy')]
    
    cv5_8 = pd.DataFrame({'classifier': clf.__class__.__name__
                             , 'training': train_scores
                            , 'test': test_scores
                            , 'score': score_names})
    return cv5_8


def train_test_split_radiusneighborsclassifier(X_train, y_train):
    params = []
      
    radius = [300, 200, 100]
    leaf_size = [100, 50, 15]
    print('Number of combinations: ', len(radius)*len(leaf_size))
    
    
    # train-test split
    train_balanced_accuracy_score_all = []
    train_confusion_matrix_all = []
    train_multilabel_confusion_matrix_all = []
    train_matthews_corrcoef_all = []
    train_fbeta_score_micro_all = []
    train_fbeta_score_macro_all = []
    train_fbeta_score_weighted_all = []
    train_precision_recall_fscore_support_micro_all =[]
    train_precision_recall_fscore_support_macro_all = []
    train_precision_recall_fscore_support_weighted_all = []
    test_balanced_accuracy_score_all= []
    test_confusion_matrix_all= []
    test_multilabel_confusion_matrix_all= []
    test_matthews_corrcoef_all= []
    test_fbeta_score_micro_all= []
    test_fbeta_score_macro_all= []
    test_fbeta_score_weighted_all= []
    test_precision_recall_fscore_support_micro_all= []
    test_precision_recall_fscore_support_macro_all= []
    test_precision_recall_fscore_support_weighted_all= []
    training = [train_balanced_accuracy_score_all, train_matthews_corrcoef_all, train_fbeta_score_micro_all, train_fbeta_score_macro_all, train_fbeta_score_weighted_all, train_precision_recall_fscore_support_micro_all, train_precision_recall_fscore_support_macro_all, train_precision_recall_fscore_support_weighted_all]
    testing = [test_balanced_accuracy_score_all, test_matthews_corrcoef_all, test_fbeta_score_micro_all, test_fbeta_score_macro_all, test_fbeta_score_weighted_all, test_precision_recall_fscore_support_micro_all, test_precision_recall_fscore_support_macro_all, test_precision_recall_fscore_support_weighted_all]
    
    for r in radius:
        for l in leaf_size: 
            clf = RadiusNeighborsClassifier(radius=r, weights='distance', leaf_size=l, n_jobs=-1)
            # train on training set  
            clf.fit(X_train, y_train)
            # training set 
            y_pred = clf.predict(X_train)    
            # multiclass evaluation scores 
            train_balanced_accuracy_score_all.append(balanced_accuracy_score(y_train, y_pred))
            train_confusion_matrix_all.append(confusion_matrix(y_train, y_pred))
            train_multilabel_confusion_matrix_all.append(multilabel_confusion_matrix(y_train, y_pred))
            # d = hinge_loss(y_train, y_pred, labels)
            train_matthews_corrcoef_all.append(matthews_corrcoef(y_train, y_pred))
            # f= roc_auc_score(y_train, y_pred, average='micro', multi_class='ovo')
            train_fbeta_score_micro_all.append(fbeta_score(y_train, y_pred, average='micro', beta=1))
            train_fbeta_score_macro_all.append(fbeta_score(y_train, y_pred, average='macro', beta=1))
            train_fbeta_score_weighted_all.append(fbeta_score(y_train, y_pred, average='weighted', beta=1))
            train_precision_recall_fscore_support_micro_all.append(precision_recall_fscore_support(y_train, y_pred, beta=1, average='micro'))
            train_precision_recall_fscore_support_macro_all.append(precision_recall_fscore_support(y_train, y_pred, beta=1, average='macro'))
            train_precision_recall_fscore_support_weighted_all.append(precision_recall_fscore_support(y_train, y_pred, beta=1, average='weighted'))
            
            # test set 
            y_pred2 = clf.predict(X_test)  
            # multiclass evaluation scores
            test_balanced_accuracy_score_all.append(balanced_accuracy_score(y_test, y_pred2))
            test_confusion_matrix_all.append(confusion_matrix(y_test, y_pred2))
            test_multilabel_confusion_matrix_all.append(multilabel_confusion_matrix(y_test, y_pred2))
            # d2 = hinge_loss(y_test, labels=y_pred)
            test_matthews_corrcoef_all.append(matthews_corrcoef(y_test, y_pred2))
            # f2= roc_auc_score(y_test, y_pred, average='micro', multi_class='ovo')
            test_fbeta_score_micro_all.append(fbeta_score(y_test, y_pred2, average='micro', beta=1))
            test_fbeta_score_macro_all.append(fbeta_score(y_test, y_pred2, average='macro', beta=1))
            test_fbeta_score_weighted_all.append(fbeta_score(y_test, y_pred2, average='weighted', beta=1))
            test_precision_recall_fscore_support_micro_all.append(precision_recall_fscore_support(y_test, y_pred2, beta=1, average='micro'))
            test_precision_recall_fscore_support_macro_all.append(precision_recall_fscore_support(y_test, y_pred2, beta=1, average='macro'))
            test_precision_recall_fscore_support_weighted_all.append(precision_recall_fscore_support(y_test, y_pred2, beta=1, average='weighted'))    
           
            params.append(f'radius: {r}, leaf_size: {l}')
        
    
    # get max score's parameters 
    print('Highest score: ', max(test_balanced_accuracy_score_all))
    print(params[test_balanced_accuracy_score_all.index(max(test_balanced_accuracy_score_all))])
    # print('Number of combinations: ', len(solver)*len(multi_class))
    
    max_index = test_balanced_accuracy_score_all.index(max(test_balanced_accuracy_score_all))
    
    train_scores = []
    for i in training: 
        i = i[max_index]
        if type(i) == tuple: 
            prec = i[0]
            recall = i[1]
            f1 = i[2]
            train_scores.append(prec)
            train_scores.append(recall)
            train_scores.append(f1)
        else:  
            train_scores.append(i)
    
    test_scores = []
    for i in testing: 
        i = i[max_index]
        if type(i) == tuple: 
            prec = i[0]
            recall = i[1]
            f1 = i[2]
            test_scores.append(prec)
            test_scores.append(recall)
            test_scores.append(f1)
        else:  
            test_scores.append(i)
    
    score_names =  ['balanced_accuracy', 'matthews_corrcoef', 'fbeta_score_micro', 'fbeta_score_macro', 'fbeta_score_weighted', 'precision_micro', 'recall_micro', 'f1_micro', 'precision_macro', 'precision_macro', 'f1_macro', 'precision_weighted', 'recall_weighted', 'f1_weighted']
    
    
    assert len(train_scores) == len(score_names)
    assert len(test_scores) == len(score_names)
          
    testdot1_8 = pd.DataFrame({'classifier': clf.__class__.__name__
                             , 'training': train_scores
                            , 'test': test_scores
                            , 'score': score_names})
    return testdot1_8


def cross_validation_linearsvc(X, y, cv=5):
    params = []
    
    score_names_all = []
    test_scores_all = []
    train_scores_all = []
    
    multi_class = ['ovr', 'crammer_singer'] #ovo = crammer_singer
    print('Number of combinations: ', len(multi_class))
    
    for multi in multi_class: 
        clf = LinearSVC( dual=False, multi_class=multi, class_weight=None, random_state=24, max_iter=1000000000)
        scoring = ['balanced_accuracy', 'f1_micro', 'f1_macro', 'f1_weighted', 'precision_micro', 'precision_macro', 'precision_weighted','recall_micro', 'recall_macro', 'recall_weighted', 'roc_auc_ovr', 'roc_auc_ovo', 'roc_auc_ovr_weighted', 'roc_auc_ovo_weighted']
        # train on training set  
        scores = cross_validate(clf, X, y, cv=5, scoring=scoring, return_train_score=True)
        score_names = []
        test_scores = [] 
        train_scores = []        
        for el in sorted(scores.keys()): 
            if el.startswith('test') and not el.endswith('matrix'): 
                test_scores.append(scores[el].mean())
                score_names.append(el[5:])
            elif el.startswith('train') and not el.endswith('matrix'): 
                train_scores.append(scores[el].mean())
        assert len(train_scores) == len(test_scores)
        assert len(train_scores) == len(score_names)
    
        score_names_all.append(score_names)
        test_scores_all.append(test_scores)
        train_scores_all.append(train_scores)
        params.append(f'multi_class: {multi}')
        
    # get max score's parameters 
    for el in test_scores_all: 
        if el[score_names.index('balanced_accuracy')] == max([i[score_names.index('balanced_accuracy')] for i in test_scores_all]): 
            windi = test_scores_all.index(el)
    
    print('Highest score: ', max([i[score_names.index('balanced_accuracy')] for i in test_scores_all]))
    print(params[windi])
    
    train_scores = train_scores_all[score_names.index('balanced_accuracy')]
    test_scores = test_scores_all[score_names.index('balanced_accuracy')]
    score_names =  score_names_all[score_names.index('balanced_accuracy')]
    
    cv5_9 = pd.DataFrame({'classifier': clf.__class__.__name__
                             , 'training': train_scores
                            , 'test': test_scores
                            , 'score': score_names})
    return cv5_9




def train_test_split_linearsvc(X_train, y_train):
    params = []
    
    multi_class = ['ovr', 'crammer_singer'] #ovo = crammer_singer
    print('Number of combinations: ', len(multi_class))

    # train-test split
    train_balanced_accuracy_score_all = []
    train_confusion_matrix_all = []
    train_multilabel_confusion_matrix_all = []
    train_matthews_corrcoef_all = []
    train_fbeta_score_micro_all = []
    train_fbeta_score_macro_all = []
    train_fbeta_score_weighted_all = []
    train_precision_recall_fscore_support_micro_all =[]
    train_precision_recall_fscore_support_macro_all = []
    train_precision_recall_fscore_support_weighted_all = []
    test_balanced_accuracy_score_all= []
    test_confusion_matrix_all= []
    test_multilabel_confusion_matrix_all= []
    test_matthews_corrcoef_all= []
    test_fbeta_score_micro_all= []
    test_fbeta_score_macro_all= []
    test_fbeta_score_weighted_all= []
    test_precision_recall_fscore_support_micro_all= []
    test_precision_recall_fscore_support_macro_all= []
    test_precision_recall_fscore_support_weighted_all= []
    training = [train_balanced_accuracy_score_all, train_matthews_corrcoef_all, train_fbeta_score_micro_all, train_fbeta_score_macro_all, train_fbeta_score_weighted_all, train_precision_recall_fscore_support_micro_all, train_precision_recall_fscore_support_macro_all, train_precision_recall_fscore_support_weighted_all]
    testing = [test_balanced_accuracy_score_all, test_matthews_corrcoef_all, test_fbeta_score_micro_all, test_fbeta_score_macro_all, test_fbeta_score_weighted_all, test_precision_recall_fscore_support_micro_all, test_precision_recall_fscore_support_macro_all, test_precision_recall_fscore_support_weighted_all]
       
        
    for multi in multi_class: 
        clf = LinearSVC( dual=False, multi_class=multi, class_weight=None, random_state=24, max_iter=1000000000)
        # train on training set  
        clf.fit(X_train, y_train)
        # training set 
        y_pred = clf.predict(X_train)    
        # multiclass evaluation scores 
        train_balanced_accuracy_score_all.append(balanced_accuracy_score(y_train, y_pred))
        train_confusion_matrix_all.append(confusion_matrix(y_train, y_pred))
        train_multilabel_confusion_matrix_all.append(multilabel_confusion_matrix(y_train, y_pred))
        # d = hinge_loss(y_train, y_pred, labels)
        train_matthews_corrcoef_all.append(matthews_corrcoef(y_train, y_pred))
        # f= roc_auc_score(y_train, y_pred, average='micro', multi_class='ovo')
        train_fbeta_score_micro_all.append(fbeta_score(y_train, y_pred, average='micro', beta=1))
        train_fbeta_score_macro_all.append(fbeta_score(y_train, y_pred, average='macro', beta=1))
        train_fbeta_score_weighted_all.append(fbeta_score(y_train, y_pred, average='weighted', beta=1))
        train_precision_recall_fscore_support_micro_all.append(precision_recall_fscore_support(y_train, y_pred, beta=1, average='micro'))
        train_precision_recall_fscore_support_macro_all.append(precision_recall_fscore_support(y_train, y_pred, beta=1, average='macro'))
        train_precision_recall_fscore_support_weighted_all.append(precision_recall_fscore_support(y_train, y_pred, beta=1, average='weighted'))
        
        # test set 
        y_pred2 = clf.predict(X_test)  
        # multiclass evaluation scores
        test_balanced_accuracy_score_all.append(balanced_accuracy_score(y_test, y_pred2))
        test_confusion_matrix_all.append(confusion_matrix(y_test, y_pred2))
        test_multilabel_confusion_matrix_all.append(multilabel_confusion_matrix(y_test, y_pred2))
        # d2 = hinge_loss(y_test, labels=y_pred)
        test_matthews_corrcoef_all.append(matthews_corrcoef(y_test, y_pred2))
        # f2= roc_auc_score(y_test, y_pred, average='micro', multi_class='ovo')
        test_fbeta_score_micro_all.append(fbeta_score(y_test, y_pred2, average='micro', beta=1))
        test_fbeta_score_macro_all.append(fbeta_score(y_test, y_pred2, average='macro', beta=1))
        test_fbeta_score_weighted_all.append(fbeta_score(y_test, y_pred2, average='weighted', beta=1))
        test_precision_recall_fscore_support_micro_all.append(precision_recall_fscore_support(y_test, y_pred2, beta=1, average='micro'))
        test_precision_recall_fscore_support_macro_all.append(precision_recall_fscore_support(y_test, y_pred2, beta=1, average='macro'))
        test_precision_recall_fscore_support_weighted_all.append(precision_recall_fscore_support(y_test, y_pred2, beta=1, average='weighted'))    
        params.append(f'multi_class: {multi}')
         
            
    
    # get max score's parameters 
    print('Highest score: ', max(test_balanced_accuracy_score_all))
    print(params[test_balanced_accuracy_score_all.index(max(test_balanced_accuracy_score_all))])
    # print('Number of combinations: ', len(solver)*len(multi_class))
    
    max_index = test_balanced_accuracy_score_all.index(max(test_balanced_accuracy_score_all))
    
    train_scores = []
    for i in training: 
        i = i[max_index]
        if type(i) == tuple: 
            prec = i[0]
            recall = i[1]
            f1 = i[2]
            train_scores.append(prec)
            train_scores.append(recall)
            train_scores.append(f1)
        else:  
            train_scores.append(i)
    
    test_scores = []
    for i in testing: 
        i = i[max_index]
        if type(i) == tuple: 
            prec = i[0]
            recall = i[1]
            f1 = i[2]
            test_scores.append(prec)
            test_scores.append(recall)
            test_scores.append(f1)
        else:  
            test_scores.append(i)
    
    score_names =  ['balanced_accuracy', 'matthews_corrcoef', 'fbeta_score_micro', 'fbeta_score_macro', 'fbeta_score_weighted', 'precision_micro', 'recall_micro', 'f1_micro', 'precision_macro', 'precision_macro', 'f1_macro', 'precision_weighted', 'recall_weighted', 'f1_weighted']
    
    
    assert len(train_scores) == len(score_names)
    assert len(test_scores) == len(score_names)
          
    testdot1_9 = pd.DataFrame({'classifier': clf.__class__.__name__
                             , 'training': train_scores
                            , 'test': test_scores
                            , 'score': score_names})
    return testdot1_9


def cross_validation_decisiontreeclassifier(X, y, cv=5):
    params = []
    
    score_names_all = []
    test_scores_all = []
    train_scores_all = []
    max_depth = [5, 7, 10, 11, 12, 13, 14,  15, 16, 17]
    print('Number of combinations: ', len(max_depth))
    
    for d in max_depth:
        clf = DecisionTreeClassifier(max_depth=d, random_state=24, max_leaf_nodes=1000, class_weight='balanced')
        scoring = ['balanced_accuracy', 'f1_micro', 'f1_macro', 'f1_weighted', 'precision_micro', 'precision_macro', 'precision_weighted','recall_micro', 'recall_macro', 'recall_weighted', 'roc_auc_ovr', 'roc_auc_ovo', 'roc_auc_ovr_weighted', 'roc_auc_ovo_weighted']
        # train on training set  
        scores = cross_validate(clf, X, y, cv=5, scoring=scoring, return_train_score=True)
        score_names = []
        test_scores = [] 
        train_scores = []        
        for el in sorted(scores.keys()): 
            if el.startswith('test') and not el.endswith('matrix'): 
                test_scores.append(scores[el].mean())
                score_names.append(el[5:])
            elif el.startswith('train') and not el.endswith('matrix'): 
                train_scores.append(scores[el].mean())
        assert len(train_scores) == len(test_scores)
        assert len(train_scores) == len(score_names)
    
        score_names_all.append(score_names)
        test_scores_all.append(test_scores)
        train_scores_all.append(train_scores)
        params.append(f'max_depth: {d}')
        
    # get max score's parameters 
    for el in test_scores_all: 
        if el[score_names.index('balanced_accuracy')] == max([i[score_names.index('balanced_accuracy')] for i in test_scores_all]): 
            windi = test_scores_all.index(el)
    
    print('Highest score: ', max([i[score_names.index('balanced_accuracy')] for i in test_scores_all]))
    print(params[windi])
    
    train_scores = train_scores_all[score_names.index('balanced_accuracy')]
    test_scores = test_scores_all[score_names.index('balanced_accuracy')]
    score_names =  score_names_all[score_names.index('balanced_accuracy')]
    
    cv5_10 = pd.DataFrame({'classifier': clf.__class__.__name__
                             , 'training': train_scores
                            , 'test': test_scores
                            , 'score': score_names})
    return cv5_10


def train_test_split_decisiontreeclassifier(X_train, y_train): 
    params = []

    max_depth = [5, 7, 10, 11, 12, 13, 14,  15, 16, 17]
    print('Number of combinations: ', len(max_depth))
    
    # train-test split
    train_balanced_accuracy_score_all = []
    train_confusion_matrix_all = []
    train_multilabel_confusion_matrix_all = []
    train_matthews_corrcoef_all = []
    train_fbeta_score_micro_all = []
    train_fbeta_score_macro_all = []
    train_fbeta_score_weighted_all = []
    train_precision_recall_fscore_support_micro_all =[]
    train_precision_recall_fscore_support_macro_all = []
    train_precision_recall_fscore_support_weighted_all = []
    test_balanced_accuracy_score_all= []
    test_confusion_matrix_all= []
    test_multilabel_confusion_matrix_all= []
    test_matthews_corrcoef_all= []
    test_fbeta_score_micro_all= []
    test_fbeta_score_macro_all= []
    test_fbeta_score_weighted_all= []
    test_precision_recall_fscore_support_micro_all= []
    test_precision_recall_fscore_support_macro_all= []
    test_precision_recall_fscore_support_weighted_all= []
    training = [train_balanced_accuracy_score_all, train_matthews_corrcoef_all, train_fbeta_score_micro_all, train_fbeta_score_macro_all, train_fbeta_score_weighted_all, train_precision_recall_fscore_support_micro_all, train_precision_recall_fscore_support_macro_all, train_precision_recall_fscore_support_weighted_all]
    testing = [test_balanced_accuracy_score_all, test_matthews_corrcoef_all, test_fbeta_score_micro_all, test_fbeta_score_macro_all, test_fbeta_score_weighted_all, test_precision_recall_fscore_support_micro_all, test_precision_recall_fscore_support_macro_all, test_precision_recall_fscore_support_weighted_all]
        
        
    for d in max_depth:
        # for l in leaf_size: 
        clf = DecisionTreeClassifier(max_depth=d, random_state=24, max_leaf_nodes=1000, class_weight='balanced')
        # train on training set  
        clf.fit(X_train, y_train)
        # training set 
        y_pred = clf.predict(X_train)    
        # multiclass evaluation scores 
        train_balanced_accuracy_score_all.append(balanced_accuracy_score(y_train, y_pred))
        train_confusion_matrix_all.append(confusion_matrix(y_train, y_pred))
        train_multilabel_confusion_matrix_all.append(multilabel_confusion_matrix(y_train, y_pred))
        # d = hinge_loss(y_train, y_pred, labels)
        train_matthews_corrcoef_all.append(matthews_corrcoef(y_train, y_pred))
        # f= roc_auc_score(y_train, y_pred, average='micro', multi_class='ovo')
        train_fbeta_score_micro_all.append(fbeta_score(y_train, y_pred, average='micro', beta=1))
        train_fbeta_score_macro_all.append(fbeta_score(y_train, y_pred, average='macro', beta=1))
        train_fbeta_score_weighted_all.append(fbeta_score(y_train, y_pred, average='weighted', beta=1))
        train_precision_recall_fscore_support_micro_all.append(precision_recall_fscore_support(y_train, y_pred, beta=1, average='micro'))
        train_precision_recall_fscore_support_macro_all.append(precision_recall_fscore_support(y_train, y_pred, beta=1, average='macro'))
        train_precision_recall_fscore_support_weighted_all.append(precision_recall_fscore_support(y_train, y_pred, beta=1, average='weighted'))
        
        # test set 
        y_pred2 = clf.predict(X_test)  
        # multiclass evaluation scores
        test_balanced_accuracy_score_all.append(balanced_accuracy_score(y_test, y_pred2))
        test_confusion_matrix_all.append(confusion_matrix(y_test, y_pred2))
        test_multilabel_confusion_matrix_all.append(multilabel_confusion_matrix(y_test, y_pred2))
        # d2 = hinge_loss(y_test, labels=y_pred)
        test_matthews_corrcoef_all.append(matthews_corrcoef(y_test, y_pred2))
        # f2= roc_auc_score(y_test, y_pred, average='micro', multi_class='ovo')
        test_fbeta_score_micro_all.append(fbeta_score(y_test, y_pred2, average='micro', beta=1))
        test_fbeta_score_macro_all.append(fbeta_score(y_test, y_pred2, average='macro', beta=1))
        test_fbeta_score_weighted_all.append(fbeta_score(y_test, y_pred2, average='weighted', beta=1))
        test_precision_recall_fscore_support_micro_all.append(precision_recall_fscore_support(y_test, y_pred2, beta=1, average='micro'))
        test_precision_recall_fscore_support_macro_all.append(precision_recall_fscore_support(y_test, y_pred2, beta=1, average='macro'))
        test_precision_recall_fscore_support_weighted_all.append(precision_recall_fscore_support(y_test, y_pred2, beta=1, average='weighted'))    
        params.append(f'max_depth: {d}')
        
    # get max score's parameters 
    print('Highest score: ', max(test_balanced_accuracy_score_all))
    print(params[test_balanced_accuracy_score_all.index(max(test_balanced_accuracy_score_all))])
   
    max_index = test_balanced_accuracy_score_all.index(max(test_balanced_accuracy_score_all))
    
    train_scores = []
    for i in training: 
        i = i[max_index]
        if type(i) == tuple: 
            prec = i[0]
            recall = i[1]
            f1 = i[2]
            train_scores.append(prec)
            train_scores.append(recall)
            train_scores.append(f1)
        else:  
            train_scores.append(i)
    
    test_scores = []
    for i in testing: 
        i = i[max_index]
        if type(i) == tuple: 
            prec = i[0]
            recall = i[1]
            f1 = i[2]
            test_scores.append(prec)
            test_scores.append(recall)
            test_scores.append(f1)
        else:  
            test_scores.append(i)
    
    score_names =  ['balanced_accuracy', 'matthews_corrcoef', 'fbeta_score_micro', 'fbeta_score_macro', 'fbeta_score_weighted', 'precision_micro', 'recall_micro', 'f1_micro', 'precision_macro', 'precision_macro', 'f1_macro', 'precision_weighted', 'recall_weighted', 'f1_weighted']
    
    
    assert len(train_scores) == len(score_names)
    assert len(test_scores) == len(score_names)
          
    testdot1_10 = pd.DataFrame({'classifier': clf.__class__.__name__
                             , 'training': train_scores
                            , 'test': test_scores
                            , 'score': score_names})
    return testdot1_10

    
def cross_validation_extratreeclassifier(X, y, cv=5): 
    params = []
    
    score_names_all = []
    test_scores_all = []
    train_scores_all = []
    
    max_depth = [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 30, 50, 100]
    print('Number of combinations: ', len(max_depth))
    
    for d in max_depth:
        clf = ExtraTreeClassifier(max_depth=d, max_features=None, random_state=24)
        scoring = ['balanced_accuracy', 'f1_micro', 'f1_macro', 'f1_weighted', 'precision_micro', 'precision_macro', 'precision_weighted','recall_micro', 'recall_macro', 'recall_weighted', 'roc_auc_ovr', 'roc_auc_ovo', 'roc_auc_ovr_weighted', 'roc_auc_ovo_weighted']
        # train on training set  
        scores = cross_validate(clf, X, y, cv=5, scoring=scoring, return_train_score=True)
        score_names = []
        test_scores = [] 
        train_scores = []        
        for el in sorted(scores.keys()): 
            if el.startswith('test') and not el.endswith('matrix'): 
                test_scores.append(scores[el].mean())
                score_names.append(el[5:])
            elif el.startswith('train') and not el.endswith('matrix'): 
                train_scores.append(scores[el].mean())
        assert len(train_scores) == len(test_scores)
        assert len(train_scores) == len(score_names)
    
        score_names_all.append(score_names)
        test_scores_all.append(test_scores)
        train_scores_all.append(train_scores)
        params.append(f'max_depth: {d}')
        
    # get max score's parameters 
    for el in test_scores_all: 
        if el[score_names.index('balanced_accuracy')] == max([i[score_names.index('balanced_accuracy')] for i in test_scores_all]): 
            windi = test_scores_all.index(el)
    
    print('Highest score: ', max([i[score_names.index('balanced_accuracy')] for i in test_scores_all]))
    print(params[windi])
    
    train_scores = train_scores_all[score_names.index('balanced_accuracy')]
    test_scores = test_scores_all[score_names.index('balanced_accuracy')]
    score_names =  score_names_all[score_names.index('balanced_accuracy')]
    
    cv5_11 = pd.DataFrame({'classifier': clf.__class__.__name__
                             , 'training': train_scores
                            , 'test': test_scores
                            , 'score': score_names})
    return cv5_11


def train_test_split_extratreeclassifier(X_train, y_train): 
    params = []
    
    max_depth = [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 30, 50, 100]
    print('Number of combinations: ', len(max_depth))
    
    train_balanced_accuracy_score_all = []
    train_confusion_matrix_all = []
    train_multilabel_confusion_matrix_all = []
    train_matthews_corrcoef_all = []
    train_fbeta_score_micro_all = []
    train_fbeta_score_macro_all = []
    train_fbeta_score_weighted_all = []
    train_precision_recall_fscore_support_micro_all =[]
    train_precision_recall_fscore_support_macro_all = []
    train_precision_recall_fscore_support_weighted_all = []
    test_balanced_accuracy_score_all= []
    test_confusion_matrix_all= []
    test_multilabel_confusion_matrix_all= []
    test_matthews_corrcoef_all= []
    test_fbeta_score_micro_all= []
    test_fbeta_score_macro_all= []
    test_fbeta_score_weighted_all= []
    test_precision_recall_fscore_support_micro_all= []
    test_precision_recall_fscore_support_macro_all= []
    test_precision_recall_fscore_support_weighted_all= []
    training = [train_balanced_accuracy_score_all, train_matthews_corrcoef_all, train_fbeta_score_micro_all, train_fbeta_score_macro_all, train_fbeta_score_weighted_all, train_precision_recall_fscore_support_micro_all, train_precision_recall_fscore_support_macro_all, train_precision_recall_fscore_support_weighted_all]
    testing = [test_balanced_accuracy_score_all, test_matthews_corrcoef_all, test_fbeta_score_micro_all, test_fbeta_score_macro_all, test_fbeta_score_weighted_all, test_precision_recall_fscore_support_micro_all, test_precision_recall_fscore_support_macro_all, test_precision_recall_fscore_support_weighted_all]
      
    for d in max_depth:
        clf = ExtraTreeClassifier(max_depth=d, max_features=None, random_state=24) 
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_train)    
        train_balanced_accuracy_score_all.append(balanced_accuracy_score(y_train, y_pred))
        train_confusion_matrix_all.append(confusion_matrix(y_train, y_pred))
        train_multilabel_confusion_matrix_all.append(multilabel_confusion_matrix(y_train, y_pred))
        # d = hinge_loss(y_train, y_pred, labels)
        train_matthews_corrcoef_all.append(matthews_corrcoef(y_train, y_pred))
        # f= roc_auc_score(y_train, y_pred, average='micro', multi_class='ovo')
        train_fbeta_score_micro_all.append(fbeta_score(y_train, y_pred, average='micro', beta=1))
        train_fbeta_score_macro_all.append(fbeta_score(y_train, y_pred, average='macro', beta=1))
        train_fbeta_score_weighted_all.append(fbeta_score(y_train, y_pred, average='weighted', beta=1))
        train_precision_recall_fscore_support_micro_all.append(precision_recall_fscore_support(y_train, y_pred, beta=1, average='micro'))
        train_precision_recall_fscore_support_macro_all.append(precision_recall_fscore_support(y_train, y_pred, beta=1, average='macro'))
        train_precision_recall_fscore_support_weighted_all.append(precision_recall_fscore_support(y_train, y_pred, beta=1, average='weighted'))
        
        # test set 
        y_pred2 = clf.predict(X_test)  
        # multiclass evaluation scores
        test_balanced_accuracy_score_all.append(balanced_accuracy_score(y_test, y_pred2))
        test_confusion_matrix_all.append(confusion_matrix(y_test, y_pred2))
        test_multilabel_confusion_matrix_all.append(multilabel_confusion_matrix(y_test, y_pred2))
        # d2 = hinge_loss(y_test, labels=y_pred)
        test_matthews_corrcoef_all.append(matthews_corrcoef(y_test, y_pred2))
        # f2= roc_auc_score(y_test, y_pred, average='micro', multi_class='ovo')
        test_fbeta_score_micro_all.append(fbeta_score(y_test, y_pred2, average='micro', beta=1))
        test_fbeta_score_macro_all.append(fbeta_score(y_test, y_pred2, average='macro', beta=1))
        test_fbeta_score_weighted_all.append(fbeta_score(y_test, y_pred2, average='weighted', beta=1))
        test_precision_recall_fscore_support_micro_all.append(precision_recall_fscore_support(y_test, y_pred2, beta=1, average='micro'))
        test_precision_recall_fscore_support_macro_all.append(precision_recall_fscore_support(y_test, y_pred2, beta=1, average='macro'))
        test_precision_recall_fscore_support_weighted_all.append(precision_recall_fscore_support(y_test, y_pred2, beta=1, average='weighted'))    
        params.append(f'max_depth: {d}')
        
    
    # get max score's parameters 
    print('Highest score: ', max(test_balanced_accuracy_score_all))
    print(params[test_balanced_accuracy_score_all.index(max(test_balanced_accuracy_score_all))])
    
    max_index = test_balanced_accuracy_score_all.index(max(test_balanced_accuracy_score_all))
    
    train_scores = []
    for i in training: 
        i = i[max_index]
        if type(i) == tuple: 
            prec = i[0]
            recall = i[1]
            f1 = i[2]
            train_scores.append(prec)
            train_scores.append(recall)
            train_scores.append(f1)
        else:  
            train_scores.append(i)
    
    test_scores = []
    for i in testing: 
        i = i[max_index]
        if type(i) == tuple: 
            prec = i[0]
            recall = i[1]
            f1 = i[2]
            test_scores.append(prec)
            test_scores.append(recall)
            test_scores.append(f1)
        else:  
            test_scores.append(i)
    
    score_names =  ['balanced_accuracy', 'matthews_corrcoef', 'fbeta_score_micro', 'fbeta_score_macro', 'fbeta_score_weighted', 'precision_micro', 'recall_micro', 'f1_micro', 'precision_macro', 'precision_macro', 'f1_macro', 'precision_weighted', 'recall_weighted', 'f1_weighted']
    
    
    assert len(train_scores) == len(score_names)
    assert len(test_scores) == len(score_names)
          
    testdot1_11 = pd.DataFrame({'classifier': clf.__class__.__name__
                             , 'training': train_scores
                            , 'test': test_scores
                            , 'score': score_names})
    return testdot1_11


def cross_validation_extratreesclassifier(X, y, cv=5): 
    params = []
     
    score_names_all = []
    test_scores_all = []
    train_scores_all = []
    
    n_estimators = [1, 2, 5, 7, 10, 15, 20]
    max_depths = [2, 4, 6, 8, 10, 12, 15, 25, 30]
    print('Number of combinations: ', len(n_estimators)*len(max_depths))
   
    for ne in n_estimators:
        for d in max_depths: 
            clf = ExtraTreesClassifier(n_estimators=ne, max_depth = d, max_features=None, n_jobs=-1, random_state=24)
            scoring = ['balanced_accuracy', 'f1_micro', 'f1_macro', 'f1_weighted', 'precision_micro', 'precision_macro', 'precision_weighted','recall_micro', 'recall_macro', 'recall_weighted', 'roc_auc_ovr', 'roc_auc_ovo', 'roc_auc_ovr_weighted', 'roc_auc_ovo_weighted']
            # train on training set  
            scores = cross_validate(clf, X, y, cv=5, scoring=scoring, return_train_score=True)
            score_names = []
            test_scores = [] 
            train_scores = []        
            for el in sorted(scores.keys()): 
                if el.startswith('test') and not el.endswith('matrix'): 
                    test_scores.append(scores[el].mean())
                    score_names.append(el[5:])
                elif el.startswith('train') and not el.endswith('matrix'): 
                    train_scores.append(scores[el].mean())
            assert len(train_scores) == len(test_scores)
            assert len(train_scores) == len(score_names)
        
            score_names_all.append(score_names)
            test_scores_all.append(test_scores)
            train_scores_all.append(train_scores)
            params.append(f'n_estimators: {ne}, max_depth: {d}')
            
    # get max score's parameters 
    for el in test_scores_all: 
        if el[score_names.index('balanced_accuracy')] == max([i[score_names.index('balanced_accuracy')] for i in test_scores_all]): 
            windi = test_scores_all.index(el)
    
    print('Highest score: ', max([i[score_names.index('balanced_accuracy')] for i in test_scores_all]))
    print(params[windi])
   
    nb =  max([i[score_names.index('balanced_accuracy')] for i in test_scores_all])
    for el in test_scores_all: 
        if el[score_names.index('balanced_accuracy')] == nb: 
            test_scores = el
    nb =  max([i[score_names.index('balanced_accuracy')] for i in train_scores_all])
    for el in train_scores_all: 
        if el[score_names.index('balanced_accuracy')] == nb: 
            train_scores = el
            
    score_names =  score_names_all[score_names.index('balanced_accuracy')]
    
    cv5_12 = pd.DataFrame({'classifier': clf.__class__.__name__
                             , 'training': train_scores
                            , 'test': test_scores
                            , 'score': score_names})
    return cv5_12


def train_test_split_extratreesclassifier(X_train, y_train): 
    params = []
    
    n_estimators = [1, 2, 5, 7, 10, 15, 20]
    max_depths = [2, 4, 6, 8, 10, 12, 15, 25, 30]
    print('Number of combinations: ', len(n_estimators)*len(max_depths))
   
    train_balanced_accuracy_score_all = []
    train_confusion_matrix_all = []
    train_multilabel_confusion_matrix_all = []
    train_matthews_corrcoef_all = []
    train_fbeta_score_micro_all = []
    train_fbeta_score_macro_all = []
    train_fbeta_score_weighted_all = []
    train_precision_recall_fscore_support_micro_all =[]
    train_precision_recall_fscore_support_macro_all = []
    train_precision_recall_fscore_support_weighted_all = []
    test_balanced_accuracy_score_all= []
    test_confusion_matrix_all= []
    test_multilabel_confusion_matrix_all= []
    test_matthews_corrcoef_all= []
    test_fbeta_score_micro_all= []
    test_fbeta_score_macro_all= []
    test_fbeta_score_weighted_all= []
    test_precision_recall_fscore_support_micro_all= []
    test_precision_recall_fscore_support_macro_all= []
    test_precision_recall_fscore_support_weighted_all= []
    training = [train_balanced_accuracy_score_all, train_matthews_corrcoef_all, train_fbeta_score_micro_all, train_fbeta_score_macro_all, train_fbeta_score_weighted_all, train_precision_recall_fscore_support_micro_all, train_precision_recall_fscore_support_macro_all, train_precision_recall_fscore_support_weighted_all]
    testing = [test_balanced_accuracy_score_all, test_matthews_corrcoef_all, test_fbeta_score_micro_all, test_fbeta_score_macro_all, test_fbeta_score_weighted_all, test_precision_recall_fscore_support_micro_all, test_precision_recall_fscore_support_macro_all, test_precision_recall_fscore_support_weighted_all]
            
            
    for ne in n_estimators:
        for d in max_depths: 
            clf = ExtraTreesClassifier(n_estimators=ne, max_depth = d, max_features=None, n_jobs=-1, random_state=24)
            # train on training set  
            clf.fit(X_train, y_train)
            # training set 
            y_pred = clf.predict(X_train)    
            train_balanced_accuracy_score_all.append(balanced_accuracy_score(y_train, y_pred))
            train_confusion_matrix_all.append(confusion_matrix(y_train, y_pred))
            train_multilabel_confusion_matrix_all.append(multilabel_confusion_matrix(y_train, y_pred))
            # d = hinge_loss(y_train, y_pred, labels)
            train_matthews_corrcoef_all.append(matthews_corrcoef(y_train, y_pred))
            # f= roc_auc_score(y_train, y_pred, average='micro', multi_class='ovo')
            train_fbeta_score_micro_all.append(fbeta_score(y_train, y_pred, average='micro', beta=1))
            train_fbeta_score_macro_all.append(fbeta_score(y_train, y_pred, average='macro', beta=1))
            train_fbeta_score_weighted_all.append(fbeta_score(y_train, y_pred, average='weighted', beta=1))
            train_precision_recall_fscore_support_micro_all.append(precision_recall_fscore_support(y_train, y_pred, beta=1, average='micro'))
            train_precision_recall_fscore_support_macro_all.append(precision_recall_fscore_support(y_train, y_pred, beta=1, average='macro'))
            train_precision_recall_fscore_support_weighted_all.append(precision_recall_fscore_support(y_train, y_pred, beta=1, average='weighted'))
            
            # test set 
            y_pred2 = clf.predict(X_test)  
            test_balanced_accuracy_score_all.append(balanced_accuracy_score(y_test, y_pred2))
            test_confusion_matrix_all.append(confusion_matrix(y_test, y_pred2))
            test_multilabel_confusion_matrix_all.append(multilabel_confusion_matrix(y_test, y_pred2))
            # d2 = hinge_loss(y_test, labels=y_pred)
            test_matthews_corrcoef_all.append(matthews_corrcoef(y_test, y_pred2))
            # f2= roc_auc_score(y_test, y_pred, average='micro', multi_class='ovo')
            test_fbeta_score_micro_all.append(fbeta_score(y_test, y_pred2, average='micro', beta=1))
            test_fbeta_score_macro_all.append(fbeta_score(y_test, y_pred2, average='macro', beta=1))
            test_fbeta_score_weighted_all.append(fbeta_score(y_test, y_pred2, average='weighted', beta=1))
            test_precision_recall_fscore_support_micro_all.append(precision_recall_fscore_support(y_test, y_pred2, beta=1, average='micro'))
            test_precision_recall_fscore_support_macro_all.append(precision_recall_fscore_support(y_test, y_pred2, beta=1, average='macro'))
            test_precision_recall_fscore_support_weighted_all.append(precision_recall_fscore_support(y_test, y_pred2, beta=1, average='weighted'))    
           
            params.append(f'n_estimators: {ne}, max_depth: {d}')
            
    
    # get max score's parameters 
    print('Highest score: ', max(test_balanced_accuracy_score_all))
    print(params[test_balanced_accuracy_score_all.index(max(test_balanced_accuracy_score_all))])
  
    max_index = test_balanced_accuracy_score_all.index(max(test_balanced_accuracy_score_all))
    
    train_scores = []
    for i in training: 
        i = i[max_index]
        if type(i) == tuple: 
            prec = i[0]
            recall = i[1]
            f1 = i[2]
            train_scores.append(prec)
            train_scores.append(recall)
            train_scores.append(f1)
        else:  
            train_scores.append(i)
    
    test_scores = []
    for i in testing: 
        i = i[max_index]
        if type(i) == tuple: 
            prec = i[0]
            recall = i[1]
            f1 = i[2]
            test_scores.append(prec)
            test_scores.append(recall)
            test_scores.append(f1)
        else:  
            test_scores.append(i)
    
    score_names =  ['balanced_accuracy', 'matthews_corrcoef', 'fbeta_score_micro', 'fbeta_score_macro', 'fbeta_score_weighted', 'precision_micro', 'recall_micro', 'f1_micro', 'precision_macro', 'precision_macro', 'f1_macro', 'precision_weighted', 'recall_weighted', 'f1_weighted']
    
    
    assert len(train_scores) == len(score_names)
    assert len(test_scores) == len(score_names)
          
    testdot1_12 = pd.DataFrame({'classifier': clf.__class__.__name__
                             , 'training': train_scores
                            , 'test': test_scores
                            , 'score': score_names})
    return testdot1_12


    
def cross_validation_randomforestclassifier(X, y, cv=5): 

    params = []
     
    score_names_all = []
    test_scores_all = []
    train_scores_all = []
    n_estimators = [61, 62, 63, 64, 65, 66, 67, 68, 69, 70]
    max_depths = [2, 4, 6, 8, 10, 12, 15, 25, 30]
    print('Number of combinations: ', len(n_estimators)*len(max_depths))
        
    for ne in n_estimators:
        for d in max_depths: 
            clf = RandomForestClassifier(n_estimators=ne, max_depth = d, max_features=None, n_jobs=-1, random_state=24)
            scoring = ['balanced_accuracy', 'f1_micro', 'f1_macro', 'f1_weighted', 'precision_micro', 'precision_macro', 'precision_weighted','recall_micro', 'recall_macro', 'recall_weighted', 'roc_auc_ovr', 'roc_auc_ovo', 'roc_auc_ovr_weighted', 'roc_auc_ovo_weighted']
            # train on training set  
            scores = cross_validate(clf, X, y, cv=5, scoring=scoring, return_train_score=True)
            score_names = []
            test_scores = [] 
            train_scores = []        
            for el in sorted(scores.keys()): 
                if el.startswith('test') and not el.endswith('matrix'): 
                    test_scores.append(scores[el].mean())
                    score_names.append(el[5:])
                elif el.startswith('train') and not el.endswith('matrix'): 
                    train_scores.append(scores[el].mean())
            assert len(train_scores) == len(test_scores)
            assert len(train_scores) == len(score_names)
        
            score_names_all.append(score_names)
            test_scores_all.append(test_scores)
            train_scores_all.append(train_scores)
            params.append(f'n_estimators: {ne}, max_depth: {d}')
            
    # get max score's parameters 
    for el in test_scores_all: 
        if el[score_names.index('balanced_accuracy')] == max([i[score_names.index('balanced_accuracy')] for i in test_scores_all]): 
            windi = test_scores_all.index(el)
    
    print('Highest score: ', max([i[score_names.index('balanced_accuracy')] for i in test_scores_all]))
    print(params[windi])
    
    nb =  max([i[score_names.index('balanced_accuracy')] for i in test_scores_all])
    for el in test_scores_all: 
        if el[score_names.index('balanced_accuracy')] == nb: 
            test_scores = el
    nb =  max([i[score_names.index('balanced_accuracy')] for i in train_scores_all])
    for el in train_scores_all: 
        if el[score_names.index('balanced_accuracy')] == nb: 
            train_scores = el
            
    score_names =  score_names_all[score_names.index('balanced_accuracy')]
    
    cv5_13 = pd.DataFrame({'classifier': clf.__class__.__name__
                             , 'training': train_scores
                            , 'test': test_scores
                            , 'score': score_names})
    return cv5_13



def train_test_split_randomforestclassifier(X_train, y_train): 
    
    params = []

    n_estimators = [61, 62, 63, 64, 65, 66, 67, 68, 69, 70]
    max_depths = [2, 4, 6, 8, 10, 12, 15, 25, 30]
    print('Number of combinations: ', len(n_estimators)*len(max_depths))
    
    # train-test split
    train_balanced_accuracy_score_all = []
    train_confusion_matrix_all = []
    train_multilabel_confusion_matrix_all = []
    train_matthews_corrcoef_all = []
    train_fbeta_score_micro_all = []
    train_fbeta_score_macro_all = []
    train_fbeta_score_weighted_all = []
    train_precision_recall_fscore_support_micro_all =[]
    train_precision_recall_fscore_support_macro_all = []
    train_precision_recall_fscore_support_weighted_all = []
    test_balanced_accuracy_score_all= []
    test_confusion_matrix_all= []
    test_multilabel_confusion_matrix_all= []
    test_matthews_corrcoef_all= []
    test_fbeta_score_micro_all= []
    test_fbeta_score_macro_all= []
    test_fbeta_score_weighted_all= []
    test_precision_recall_fscore_support_micro_all= []
    test_precision_recall_fscore_support_macro_all= []
    test_precision_recall_fscore_support_weighted_all= []
    training = [train_balanced_accuracy_score_all, train_matthews_corrcoef_all, train_fbeta_score_micro_all, train_fbeta_score_macro_all, train_fbeta_score_weighted_all, train_precision_recall_fscore_support_micro_all, train_precision_recall_fscore_support_macro_all, train_precision_recall_fscore_support_weighted_all]
    testing = [test_balanced_accuracy_score_all, test_matthews_corrcoef_all, test_fbeta_score_micro_all, test_fbeta_score_macro_all, test_fbeta_score_weighted_all, test_precision_recall_fscore_support_micro_all, test_precision_recall_fscore_support_macro_all, test_precision_recall_fscore_support_weighted_all]
           
    for ne in n_estimators:
        for d in max_depths: 
            clf = RandomForestClassifier(n_estimators=ne, max_depth = d, max_features=None, n_jobs=-1, random_state=24)
            # train on training set  
            clf.fit(X_train, y_train)
            # training set 
            y_pred = clf.predict(X_train)    
            # multiclass evaluation scores 
            train_balanced_accuracy_score_all.append(balanced_accuracy_score(y_train, y_pred))
            train_confusion_matrix_all.append(confusion_matrix(y_train, y_pred))
            train_multilabel_confusion_matrix_all.append(multilabel_confusion_matrix(y_train, y_pred))
            # d = hinge_loss(y_train, y_pred, labels)
            train_matthews_corrcoef_all.append(matthews_corrcoef(y_train, y_pred))
            # f= roc_auc_score(y_train, y_pred, average='micro', multi_class='ovo')
            train_fbeta_score_micro_all.append(fbeta_score(y_train, y_pred, average='micro', beta=1))
            train_fbeta_score_macro_all.append(fbeta_score(y_train, y_pred, average='macro', beta=1))
            train_fbeta_score_weighted_all.append(fbeta_score(y_train, y_pred, average='weighted', beta=1))
            train_precision_recall_fscore_support_micro_all.append(precision_recall_fscore_support(y_train, y_pred, beta=1, average='micro'))
            train_precision_recall_fscore_support_macro_all.append(precision_recall_fscore_support(y_train, y_pred, beta=1, average='macro'))
            train_precision_recall_fscore_support_weighted_all.append(precision_recall_fscore_support(y_train, y_pred, beta=1, average='weighted'))
            
            # test set 
            y_pred2 = clf.predict(X_test)  
            # multiclass evaluation scores
            test_balanced_accuracy_score_all.append(balanced_accuracy_score(y_test, y_pred2))
            test_confusion_matrix_all.append(confusion_matrix(y_test, y_pred2))
            test_multilabel_confusion_matrix_all.append(multilabel_confusion_matrix(y_test, y_pred2))
            # d2 = hinge_loss(y_test, labels=y_pred)
            test_matthews_corrcoef_all.append(matthews_corrcoef(y_test, y_pred2))
            # f2= roc_auc_score(y_test, y_pred, average='micro', multi_class='ovo')
            test_fbeta_score_micro_all.append(fbeta_score(y_test, y_pred2, average='micro', beta=1))
            test_fbeta_score_macro_all.append(fbeta_score(y_test, y_pred2, average='macro', beta=1))
            test_fbeta_score_weighted_all.append(fbeta_score(y_test, y_pred2, average='weighted', beta=1))
            test_precision_recall_fscore_support_micro_all.append(precision_recall_fscore_support(y_test, y_pred2, beta=1, average='micro'))
            test_precision_recall_fscore_support_macro_all.append(precision_recall_fscore_support(y_test, y_pred2, beta=1, average='macro'))
            test_precision_recall_fscore_support_weighted_all.append(precision_recall_fscore_support(y_test, y_pred2, beta=1, average='weighted'))    
               
            params.append(f'n_estimators: {ne}, max_depth: {d}')
                
    # get max score's parameters 
    print('Highest score: ', max(test_balanced_accuracy_score_all))
    print(params[test_balanced_accuracy_score_all.index(max(test_balanced_accuracy_score_all))])
    
    max_index = test_balanced_accuracy_score_all.index(max(test_balanced_accuracy_score_all))
    
    train_scores = []
    for i in training: 
        i = i[max_index]
        if type(i) == tuple: 
            prec = i[0]
            recall = i[1]
            f1 = i[2]
            train_scores.append(prec)
            train_scores.append(recall)
            train_scores.append(f1)
        else:  
            train_scores.append(i)
    
    test_scores = []
    for i in testing: 
        i = i[max_index]
        if type(i) == tuple: 
            prec = i[0]
            recall = i[1]
            f1 = i[2]
            test_scores.append(prec)
            test_scores.append(recall)
            test_scores.append(f1)
        else:  
            test_scores.append(i)
    
    score_names =  ['balanced_accuracy', 'matthews_corrcoef', 'fbeta_score_micro', 'fbeta_score_macro', 'fbeta_score_weighted', 'precision_micro', 'recall_micro', 'f1_micro', 'precision_macro', 'precision_macro', 'f1_macro', 'precision_weighted', 'recall_weighted', 'f1_weighted']
       
    assert len(train_scores) == len(score_names)
    assert len(test_scores) == len(score_names)
          
    testdot1_13 = pd.DataFrame({'classifier': clf.__class__.__name__
                             , 'training': train_scores
                            , 'test': test_scores
                            , 'score': score_names})
    return testdot1_13
    

   
def cross_validation_lineardiscriminantanalysis(X, y, cv=5): 
    params = []
    
    score_names_all = []
    test_scores_all = []
    train_scores_all = []
    solver = ['lsqr', 'eigen']
    shrinkage = [None, 'auto']
    
    print('Number of combinations: ', len(solver)*len(shrinkage))
    
    for so in solver:
        for sh in shrinkage: 
            clf = LinearDiscriminantAnalysis(solver=so, shrinkage = sh)
            scoring = ['balanced_accuracy', 'f1_micro', 'f1_macro', 'f1_weighted', 'precision_micro', 'precision_macro', 'precision_weighted','recall_micro', 'recall_macro', 'recall_weighted', 'roc_auc_ovr', 'roc_auc_ovo', 'roc_auc_ovr_weighted', 'roc_auc_ovo_weighted']
            # train on training set  
            scores = cross_validate(clf, X, y, cv=5, scoring=scoring, return_train_score=True)
            score_names = []
            test_scores = [] 
            train_scores = []        
            for el in sorted(scores.keys()): 
                if el.startswith('test') and not el.endswith('matrix'): 
                    test_scores.append(scores[el].mean())
                    score_names.append(el[5:])
                elif el.startswith('train') and not el.endswith('matrix'): 
                    train_scores.append(scores[el].mean())
            assert len(train_scores) == len(test_scores)
            assert len(train_scores) == len(score_names)
        
            score_names_all.append(score_names)
            test_scores_all.append(test_scores)
            train_scores_all.append(train_scores)
            params.append(f'solver: {so}, shrinkage: {sh}')
            
            
    # get max score's parameters 
    for el in test_scores_all: 
        if el[score_names.index('balanced_accuracy')] == max([i[score_names.index('balanced_accuracy')] for i in test_scores_all]): 
            windi = test_scores_all.index(el)
    
    print('Highest score: ', max([i[score_names.index('balanced_accuracy')] for i in test_scores_all]))
    print(params[windi])
    
    nb =  max([i[score_names.index('balanced_accuracy')] for i in test_scores_all])
    for el in test_scores_all: 
        if el[score_names.index('balanced_accuracy')] == nb: 
            test_scores = el
    nb =  max([i[score_names.index('balanced_accuracy')] for i in train_scores_all])
    for el in train_scores_all: 
        if el[score_names.index('balanced_accuracy')] == nb: 
            train_scores = el
            
    score_names =  score_names_all[score_names.index('balanced_accuracy')]
    
    cv5_14 = pd.DataFrame({'classifier': clf.__class__.__name__
                             , 'training': train_scores
                            , 'test': test_scores
                            , 'score': score_names})
    return cv5_14


def train_test_split_lineardiscriminantanalysis(X_train, y_train): 
    params = []

    solver = ['lsqr', 'eigen']
    shrinkage = [None, 'auto']
    print('Number of combinations: ', len(solver)*len(shrinkage))
        
    # train-test split
    train_balanced_accuracy_score_all = []
    train_confusion_matrix_all = []
    train_multilabel_confusion_matrix_all = []
    train_matthews_corrcoef_all = []
    train_fbeta_score_micro_all = []
    train_fbeta_score_macro_all = []
    train_fbeta_score_weighted_all = []
    train_precision_recall_fscore_support_micro_all =[]
    train_precision_recall_fscore_support_macro_all = []
    train_precision_recall_fscore_support_weighted_all = []
    test_balanced_accuracy_score_all= []
    test_confusion_matrix_all= []
    test_multilabel_confusion_matrix_all= []
    test_matthews_corrcoef_all= []
    test_fbeta_score_micro_all= []
    test_fbeta_score_macro_all= []
    test_fbeta_score_weighted_all= []
    test_precision_recall_fscore_support_micro_all= []
    test_precision_recall_fscore_support_macro_all= []
    test_precision_recall_fscore_support_weighted_all= []
    training = [train_balanced_accuracy_score_all, train_matthews_corrcoef_all, train_fbeta_score_micro_all, train_fbeta_score_macro_all, train_fbeta_score_weighted_all, train_precision_recall_fscore_support_micro_all, train_precision_recall_fscore_support_macro_all, train_precision_recall_fscore_support_weighted_all]
    testing = [test_balanced_accuracy_score_all, test_matthews_corrcoef_all, test_fbeta_score_micro_all, test_fbeta_score_macro_all, test_fbeta_score_weighted_all, test_precision_recall_fscore_support_micro_all, test_precision_recall_fscore_support_macro_all, test_precision_recall_fscore_support_weighted_all]
            
            
    for so in solver:
        for sh in shrinkage: 
            clf = LinearDiscriminantAnalysis(solver=so, shrinkage = sh)
            # train on training set  
            clf.fit(X_train, y_train)
            # training set 
            y_pred = clf.predict(X_train)    
          # multiclass evaluation scores 
            train_balanced_accuracy_score_all.append(balanced_accuracy_score(y_train, y_pred))
            train_confusion_matrix_all.append(confusion_matrix(y_train, y_pred))
            train_multilabel_confusion_matrix_all.append(multilabel_confusion_matrix(y_train, y_pred))
            # d = hinge_loss(y_train, y_pred, labels)
            train_matthews_corrcoef_all.append(matthews_corrcoef(y_train, y_pred))
            # f= roc_auc_score(y_train, y_pred, average='micro', multi_class='ovo')
            train_fbeta_score_micro_all.append(fbeta_score(y_train, y_pred, average='micro', beta=1))
            train_fbeta_score_macro_all.append(fbeta_score(y_train, y_pred, average='macro', beta=1))
            train_fbeta_score_weighted_all.append(fbeta_score(y_train, y_pred, average='weighted', beta=1))
            train_precision_recall_fscore_support_micro_all.append(precision_recall_fscore_support(y_train, y_pred, beta=1, average='micro'))
            train_precision_recall_fscore_support_macro_all.append(precision_recall_fscore_support(y_train, y_pred, beta=1, average='macro'))
            train_precision_recall_fscore_support_weighted_all.append(precision_recall_fscore_support(y_train, y_pred, beta=1, average='weighted'))
            
            # test set 
            y_pred2 = clf.predict(X_test)  
            # multiclass evaluation scores
            test_balanced_accuracy_score_all.append(balanced_accuracy_score(y_test, y_pred2))
            test_confusion_matrix_all.append(confusion_matrix(y_test, y_pred2))
            test_multilabel_confusion_matrix_all.append(multilabel_confusion_matrix(y_test, y_pred2))
            # d2 = hinge_loss(y_test, labels=y_pred)
            test_matthews_corrcoef_all.append(matthews_corrcoef(y_test, y_pred2))
            # f2= roc_auc_score(y_test, y_pred, average='micro', multi_class='ovo')
            test_fbeta_score_micro_all.append(fbeta_score(y_test, y_pred2, average='micro', beta=1))
            test_fbeta_score_macro_all.append(fbeta_score(y_test, y_pred2, average='macro', beta=1))
            test_fbeta_score_weighted_all.append(fbeta_score(y_test, y_pred2, average='weighted', beta=1))
            test_precision_recall_fscore_support_micro_all.append(precision_recall_fscore_support(y_test, y_pred2, beta=1, average='micro'))
            test_precision_recall_fscore_support_macro_all.append(precision_recall_fscore_support(y_test, y_pred2, beta=1, average='macro'))
            test_precision_recall_fscore_support_weighted_all.append(precision_recall_fscore_support(y_test, y_pred2, beta=1, average='weighted'))    
      
            params.append(f'solver: {so}, shrinkage: {sh}')
            
    
    # get max score's parameters 
    print('Highest score: ', max(test_balanced_accuracy_score_all))
    print(params[test_balanced_accuracy_score_all.index(max(test_balanced_accuracy_score_all))])
    # print('Number of combinations: ', len(solver)*len(multi_class))
    
    max_index = test_balanced_accuracy_score_all.index(max(test_balanced_accuracy_score_all))
    
    train_scores = []
    for i in training: 
        i = i[max_index]
        if type(i) == tuple: 
            prec = i[0]
            recall = i[1]
            f1 = i[2]
            train_scores.append(prec)
            train_scores.append(recall)
            train_scores.append(f1)
        else:  
            train_scores.append(i)
    
    test_scores = []
    for i in testing: 
        i = i[max_index]
        if type(i) == tuple: 
            prec = i[0]
            recall = i[1]
            f1 = i[2]
            test_scores.append(prec)
            test_scores.append(recall)
            test_scores.append(f1)
        else:  
            test_scores.append(i)
    
    score_names =  ['balanced_accuracy', 'matthews_corrcoef', 'fbeta_score_micro', 'fbeta_score_macro', 'fbeta_score_weighted', 'precision_micro', 'recall_micro', 'f1_micro', 'precision_macro', 'precision_macro', 'f1_macro', 'precision_weighted', 'recall_weighted', 'f1_weighted']
    
    
    assert len(train_scores) == len(score_names)
    assert len(test_scores) == len(score_names)
          
    testdot1_14 = pd.DataFrame({'classifier': clf.__class__.__name__
                             , 'training': train_scores
                            , 'test': test_scores
                            , 'score': score_names})
    return testdot1_14

def cross_validation_quadraticdiscriminantanalysis(X, y, cv=5): 

    clf = QuadraticDiscriminantAnalysis()
    
    scoring = ['balanced_accuracy', 'f1_micro', 'f1_macro', 'f1_weighted', 'precision_micro', 'precision_macro', 'precision_weighted','recall_micro', 'recall_macro', 'recall_weighted', 'roc_auc_ovr', 'roc_auc_ovo', 'roc_auc_ovr_weighted', 'roc_auc_ovo_weighted']
    scores = cross_validate(clf, X, y, cv=5, scoring=scoring, return_train_score=True)
    key_scores = sorted(scores.keys())[2:]
    for i, score in enumerate(key_scores): 
         print(f"{key_scores[i]}: %0.2f (+/- %0.2f)" % (scores[key_scores[i]].mean(), scores[key_scores[i]].std() * 2))
    
    score_names = []
    test_scores = [] 
    train_scores = []        
    for el in sorted(scores.keys()): 
        if el.startswith('test') and not el.endswith('matrix'): 
            test_scores.append(scores[el].mean())
            score_names.append(el[5:])
        elif el.startswith('train') and not el.endswith('matrix'): 
            train_scores.append(scores[el].mean())
                
    
    assert len(train_scores) == len(test_scores)
    assert len(train_scores) == len(score_names)
    
    cv5_15 = pd.DataFrame({'classifier': clf.__class__.__name__
                             , 'training': train_scores
                            , 'test': test_scores
                            , 'score': score_names})
    return cv5_15, clf


def train_test_split_quadraticdiscriminantanalysis(X_train, y_train): 
    
    clf = QuadraticDiscriminantAnalysis()
    # train on training set  
    clf.fit(X_train, y_train)
    # training set 
    y_pred = clf.predict(X_train)   
    # multiclass evaluation scores 
    a = balanced_accuracy_score(y_train, y_pred)
    b= confusion_matrix(y_train, y_pred)
    c= multilabel_confusion_matrix(y_train, y_pred)
    # d = hinge_loss(y_train, y_pred, labels)
    e= matthews_corrcoef(y_train, y_pred)
    # f= roc_auc_score(y_train, y_pred, average='micro', multi_class='ovo')
    g= fbeta_score(y_train, y_pred, average='micro', beta=1)
    h= fbeta_score(y_train, y_pred, average='macro', beta=1)
    i = fbeta_score(y_train, y_pred, average='weighted', beta=1)
    j= precision_recall_fscore_support(y_train, y_pred, beta=1, average='micro')
    k= precision_recall_fscore_support(y_train, y_pred, beta=1, average='macro')
    l= precision_recall_fscore_support(y_train, y_pred, beta=1, average='weighted')
    
    multiclass_scores['train_balanced_accuracy_score'] = a
    multiclass_scores['train_confusion_matrix'] = b
    multiclass_scores['train_multilabel_confusion_matrix'] = c
    # multiclass_scores['train_hinge_loss'] = d
    multiclass_scores['train_matthews_corrcoef'] = e
    # multiclass_scores['train_roc_auc_score'] = f
    multiclass_scores['train_fbeta_score_micro'] = g
    multiclass_scores['train_fbeta_score_macro'] = h
    multiclass_scores['train_fbeta_score_weighted'] = i
    multiclass_scores['train_precision_recall_fscore_support_micro'] = j
    multiclass_scores['train_precision_recall_fscore_support_macro'] = k
    multiclass_scores['train_precision_recall_fscore_support_weighted'] = l
    
    # test set 
    y_pred2 = clf.predict(X_test)  
    # multiclass evaluation scores
    a2 = balanced_accuracy_score(y_test, y_pred2)
    b2= confusion_matrix(y_test, y_pred2)
    c2= multilabel_confusion_matrix(y_test, y_pred2)
    # d2 = hinge_loss(y_test, labels=y_pred)
    e2= matthews_corrcoef(y_test, y_pred2)
    # f2= roc_auc_score(y_test, y_pred, average='micro', multi_class='ovo')
    g2= fbeta_score(y_test, y_pred2, average='micro', beta=1)
    h2= fbeta_score(y_test, y_pred2, average='macro', beta=1)
    i2 = fbeta_score(y_test, y_pred2, average='weighted', beta=1)
    j2= precision_recall_fscore_support(y_test, y_pred2, beta=1, average='micro')
    k2= precision_recall_fscore_support(y_test, y_pred2, beta=1, average='macro')
    l2= precision_recall_fscore_support(y_test, y_pred2, beta=1, average='weighted')
    
    multiclass_scores['test_balanced_accuracy_score'] = a2
    multiclass_scores['test_confusion_matrix'] = b2
    multiclass_scores['test_multilabel_confusion_matrix'] = c2
    # multiclass_scores['test_hinge_loss'] = d2
    multiclass_scores['test_matthews_corrcoef'] = e2
    # multiclass_scores['test_roc_auc_score'] = f2
    multiclass_scores['test_fbeta_score_micro'] = g2
    multiclass_scores['test_fbeta_score_macro'] = h2
    multiclass_scores['test_fbeta_score_weighted'] = i2
    multiclass_scores['test_precision_recall_fscore_support_micro'] = j2
    multiclass_scores['test_precision_recall_fscore_support_macro'] = k2
    multiclass_scores['test_precision_recall_fscore_support_weighted'] = l2
    
    score_names = []
    test_scores = [] 
    train_scores = []        
    for el in sorted(multiclass_scores.keys()): 
        if el.startswith('test') and not el.endswith('matrix'): 
            if el.startswith('test_precision_recall_fscore'): 
                prec, rec, fscore, nan = multiclass_scores[el]
                test_scores.append(prec)
                test_scores.append(rec)
                test_scores.append(fscore)
                if el.endswith('weighted'): 
                    score_names.append(str(el[5:14])+'_'+str(el[-8:]))
                    score_names.append(str(el[15:21])+'_'+str(el[-8:]))
                    score_names.append(str(el[22:28])+'_'+str(el[-8:]))
                else: 
                    score_names.append(str(el[5:14])+'_'+str(el[-5:]))
                    score_names.append(str(el[15:21])+'_'+str(el[-5:]))
                    score_names.append(str(el[22:28])+'_'+str(el[-5:]))                
            else: 
                test_scores.append(multiclass_scores[el])
                score_names.append(el[5:])
        elif el.startswith('train') and not el.endswith('matrix'): 
            if el.startswith('train_precision_recall_fscore'): 
                prec, rec, fscore, nan = multiclass_scores[el]
                train_scores.append(prec)
                train_scores.append(rec)
                train_scores.append(fscore)
            else: 
                train_scores.append(multiclass_scores[el])
                
    assert len(train_scores) == len(test_scores)
    assert len(train_scores) == len(score_names)
            
    testdot1_15 = pd.DataFrame({'classifier': clf.__class__.__name__
                             , 'training': train_scores
                            , 'test': test_scores
                            , 'score': score_names})
    return testdot1_15

   
def cross_validation_labelpropagation(X, y, cv=5): 
    params = []
 
    score_names_all = []
    test_scores_all = []
    train_scores_all = []
    
    n_neighbors = np.arange(1, 21).tolist()
    print('Number of combinations: ', len(n_neighbors))
     
    for n in n_neighbors: 
        clf = LabelPropagation(kernel='knn', gamma=0, n_neighbors=n, n_jobs=-1)
        scoring = ['balanced_accuracy', 'f1_micro', 'f1_macro', 'f1_weighted', 'precision_micro', 'precision_macro', 'precision_weighted','recall_micro', 'recall_macro', 'recall_weighted', 'roc_auc_ovr', 'roc_auc_ovo', 'roc_auc_ovr_weighted', 'roc_auc_ovo_weighted']
        # train on training set  
        scores = cross_validate(clf, X, y, cv=5, scoring=scoring, return_train_score=True)
        score_names = []
        test_scores = [] 
        train_scores = []        
        for el in sorted(scores.keys()): 
            if el.startswith('test') and not el.endswith('matrix'): 
                test_scores.append(scores[el].mean())
                score_names.append(el[5:])
            elif el.startswith('train') and not el.endswith('matrix'): 
                train_scores.append(scores[el].mean())
        assert len(train_scores) == len(test_scores)
        assert len(train_scores) == len(score_names)
    
        score_names_all.append(score_names)
        test_scores_all.append(test_scores)
        train_scores_all.append(train_scores)
        params.append(f'n_neighbors: {n}')
        
    # get max score's parameters 
    for el in test_scores_all: 
        if el[score_names.index('balanced_accuracy')] == max([i[score_names.index('balanced_accuracy')] for i in test_scores_all]): 
            windi = test_scores_all.index(el)
    
    print('Highest score: ', max([i[score_names.index('balanced_accuracy')] for i in test_scores_all]))
    print(params[windi])

    nb =  max([i[score_names.index('balanced_accuracy')] for i in test_scores_all])
    for el in test_scores_all: 
        if el[score_names.index('balanced_accuracy')] == nb: 
            test_scores = el
    nb =  max([i[score_names.index('balanced_accuracy')] for i in train_scores_all])
    for el in train_scores_all: 
        if el[score_names.index('balanced_accuracy')] == nb: 
            train_scores = el
    score_names =  score_names_all[score_names.index('balanced_accuracy')]
    
    cv5_16 = pd.DataFrame({'classifier': clf.__class__.__name__
                             , 'training': train_scores
                            , 'test': test_scores
                            , 'score': score_names})
    return cv5_16


def train_test_split_labelpropagation(X_train, y_train): 
    params = []
    
    n_neighbors = np.arange(1, 21).tolist()
    print('Number of combinations: ', len(n_neighbors))
         
    # train-test split
    train_balanced_accuracy_score_all = []
    train_confusion_matrix_all = []
    train_multilabel_confusion_matrix_all = []
    train_matthews_corrcoef_all = []
    train_fbeta_score_micro_all = []
    train_fbeta_score_macro_all = []
    train_fbeta_score_weighted_all = []
    train_precision_recall_fscore_support_micro_all =[]
    train_precision_recall_fscore_support_macro_all = []
    train_precision_recall_fscore_support_weighted_all = []
    test_balanced_accuracy_score_all= []
    test_confusion_matrix_all= []
    test_multilabel_confusion_matrix_all= []
    test_matthews_corrcoef_all= []
    test_fbeta_score_micro_all= []
    test_fbeta_score_macro_all= []
    test_fbeta_score_weighted_all= []
    test_precision_recall_fscore_support_micro_all= []
    test_precision_recall_fscore_support_macro_all= []
    test_precision_recall_fscore_support_weighted_all= []
    training = [train_balanced_accuracy_score_all, train_matthews_corrcoef_all, train_fbeta_score_micro_all, train_fbeta_score_macro_all, train_fbeta_score_weighted_all, train_precision_recall_fscore_support_micro_all, train_precision_recall_fscore_support_macro_all, train_precision_recall_fscore_support_weighted_all]
    testing = [test_balanced_accuracy_score_all, test_matthews_corrcoef_all, test_fbeta_score_micro_all, test_fbeta_score_macro_all, test_fbeta_score_weighted_all, test_precision_recall_fscore_support_micro_all, test_precision_recall_fscore_support_macro_all, test_precision_recall_fscore_support_weighted_all]
        
        
    for n in n_neighbors: 
        clf = LabelPropagation(kernel='knn', gamma=0, n_neighbors=n, n_jobs=-1)
        # train on training set  
        clf.fit(X_train, y_train)
        # training set 
        y_pred = clf.predict(X_train)    
        # multiclass evaluation scores 
        train_balanced_accuracy_score_all.append(balanced_accuracy_score(y_train, y_pred))
        train_confusion_matrix_all.append(confusion_matrix(y_train, y_pred))
        train_multilabel_confusion_matrix_all.append(multilabel_confusion_matrix(y_train, y_pred))
        # d = hinge_loss(y_train, y_pred, labels)
        train_matthews_corrcoef_all.append(matthews_corrcoef(y_train, y_pred))
        # f= roc_auc_score(y_train, y_pred, average='micro', multi_class='ovo')
        train_fbeta_score_micro_all.append(fbeta_score(y_train, y_pred, average='micro', beta=1))
        train_fbeta_score_macro_all.append(fbeta_score(y_train, y_pred, average='macro', beta=1))
        train_fbeta_score_weighted_all.append(fbeta_score(y_train, y_pred, average='weighted', beta=1))
        train_precision_recall_fscore_support_micro_all.append(precision_recall_fscore_support(y_train, y_pred, beta=1, average='micro'))
        train_precision_recall_fscore_support_macro_all.append(precision_recall_fscore_support(y_train, y_pred, beta=1, average='macro'))
        train_precision_recall_fscore_support_weighted_all.append(precision_recall_fscore_support(y_train, y_pred, beta=1, average='weighted'))
        
        # test set 
        y_pred2 = clf.predict(X_test)  
        # multiclass evaluation scores
        test_balanced_accuracy_score_all.append(balanced_accuracy_score(y_test, y_pred2))
        test_confusion_matrix_all.append(confusion_matrix(y_test, y_pred2))
        test_multilabel_confusion_matrix_all.append(multilabel_confusion_matrix(y_test, y_pred2))
        # d2 = hinge_loss(y_test, labels=y_pred)
        test_matthews_corrcoef_all.append(matthews_corrcoef(y_test, y_pred2))
        # f2= roc_auc_score(y_test, y_pred, average='micro', multi_class='ovo')
        test_fbeta_score_micro_all.append(fbeta_score(y_test, y_pred2, average='micro', beta=1))
        test_fbeta_score_macro_all.append(fbeta_score(y_test, y_pred2, average='macro', beta=1))
        test_fbeta_score_weighted_all.append(fbeta_score(y_test, y_pred2, average='weighted', beta=1))
        test_precision_recall_fscore_support_micro_all.append(precision_recall_fscore_support(y_test, y_pred2, beta=1, average='micro'))
        test_precision_recall_fscore_support_macro_all.append(precision_recall_fscore_support(y_test, y_pred2, beta=1, average='macro'))
        test_precision_recall_fscore_support_weighted_all.append(precision_recall_fscore_support(y_test, y_pred2, beta=1, average='weighted'))    
    
        params.append(f'n_neighbors: {n}')
                
    
    # get max score's parameters 
    print('Highest score: ', max(test_balanced_accuracy_score_all))
    print(params[test_balanced_accuracy_score_all.index(max(test_balanced_accuracy_score_all))])
    # print('Number of combinations: ', len(solver)*len(multi_class))
    
    max_index = test_balanced_accuracy_score_all.index(max(test_balanced_accuracy_score_all))
    
    train_scores = []
    for i in training: 
        i = i[max_index]
        if type(i) == tuple: 
            prec = i[0]
            recall = i[1]
            f1 = i[2]
            train_scores.append(prec)
            train_scores.append(recall)
            train_scores.append(f1)
        else:  
            train_scores.append(i)
    
    test_scores = []
    for i in testing: 
        i = i[max_index]
        if type(i) == tuple: 
            prec = i[0]
            recall = i[1]
            f1 = i[2]
            test_scores.append(prec)
            test_scores.append(recall)
            test_scores.append(f1)
        else:  
            test_scores.append(i)
    
    score_names =  ['balanced_accuracy', 'matthews_corrcoef', 'fbeta_score_micro', 'fbeta_score_macro', 'fbeta_score_weighted', 'precision_micro', 'recall_micro', 'f1_micro', 'precision_macro', 'precision_macro', 'f1_macro', 'precision_weighted', 'recall_weighted', 'f1_weighted']
    
    
    assert len(train_scores) == len(score_names)
    assert len(test_scores) == len(score_names)
          
    testdot1_16 = pd.DataFrame({'classifier': clf.__class__.__name__
                             , 'training': train_scores
                            , 'test': test_scores
                            , 'score': score_names})
    return testdot1_16

  
def cross_validation_labelspreading(X, y, cv=5): 
    params = []
    
    score_names_all = []
    test_scores_all = []
    train_scores_all = []
        
    n_neighbors = np.arange(1, 21).tolist()
    alphas = [.2, .4, .6, .8]
    print('Number of combinations: ', len(n_neighbors)*len(alphas))
       
    for n in n_neighbors: 
        for a in alphas: 
            clf = LabelSpreading(kernel='knn', gamma=0, n_neighbors=n, alpha=a, n_jobs=-1)
            scoring = ['balanced_accuracy', 'f1_micro', 'f1_macro', 'f1_weighted', 'precision_micro', 'precision_macro', 'precision_weighted','recall_micro', 'recall_macro', 'recall_weighted', 'roc_auc_ovr', 'roc_auc_ovo', 'roc_auc_ovr_weighted', 'roc_auc_ovo_weighted']
            # train on training set  
            scores = cross_validate(clf, X, y, cv=5, scoring=scoring, return_train_score=True)
            score_names = []
            test_scores = [] 
            train_scores = []        
            for el in sorted(scores.keys()): 
                if el.startswith('test') and not el.endswith('matrix'): 
                    test_scores.append(scores[el].mean())
                    score_names.append(el[5:])
                elif el.startswith('train') and not el.endswith('matrix'): 
                    train_scores.append(scores[el].mean())
            assert len(train_scores) == len(test_scores)
            assert len(train_scores) == len(score_names)
        
            score_names_all.append(score_names)
            test_scores_all.append(test_scores)
            train_scores_all.append(train_scores)
            params.append(f'n_neighbors: {n}, alphas: {a}')
            
    # get max score's parameters 
    for el in test_scores_all: 
        if el[score_names.index('balanced_accuracy')] == max([i[score_names.index('balanced_accuracy')] for i in test_scores_all]): 
            windi = test_scores_all.index(el)
    
    print('Highest score: ', max([i[score_names.index('balanced_accuracy')] for i in test_scores_all]))
    print(params[windi])

    nb =  max([i[score_names.index('balanced_accuracy')] for i in test_scores_all])
    for el in test_scores_all: 
        if el[score_names.index('balanced_accuracy')] == nb: 
            test_scores = el
    nb =  max([i[score_names.index('balanced_accuracy')] for i in train_scores_all])
    for el in train_scores_all: 
        if el[score_names.index('balanced_accuracy')] == nb: 
            train_scores = el
            
    score_names =  score_names_all[score_names.index('balanced_accuracy')]
    
    cv5_17 = pd.DataFrame({'classifier': clf.__class__.__name__
                             , 'training': train_scores
                            , 'test': test_scores
                            , 'score': score_names})
    return cv5_17


def train_test_split_labelspreading(X_train, y_train): 
    
    params = []
        
    n_neighbors = np.arange(1, 21).tolist()
    alphas = [.2, .4, .6, .8]
    print('Number of combinations: ', len(n_neighbors)*len(alphas))
        
    # train-test split
    train_balanced_accuracy_score_all = []
    train_confusion_matrix_all = []
    train_multilabel_confusion_matrix_all = []
    train_matthews_corrcoef_all = []
    train_fbeta_score_micro_all = []
    train_fbeta_score_macro_all = []
    train_fbeta_score_weighted_all = []
    train_precision_recall_fscore_support_micro_all =[]
    train_precision_recall_fscore_support_macro_all = []
    train_precision_recall_fscore_support_weighted_all = []
    test_balanced_accuracy_score_all= []
    test_confusion_matrix_all= []
    test_multilabel_confusion_matrix_all= []
    test_matthews_corrcoef_all= []
    test_fbeta_score_micro_all= []
    test_fbeta_score_macro_all= []
    test_fbeta_score_weighted_all= []
    test_precision_recall_fscore_support_micro_all= []
    test_precision_recall_fscore_support_macro_all= []
    test_precision_recall_fscore_support_weighted_all= []
    training = [train_balanced_accuracy_score_all, train_matthews_corrcoef_all, train_fbeta_score_micro_all, train_fbeta_score_macro_all, train_fbeta_score_weighted_all, train_precision_recall_fscore_support_micro_all, train_precision_recall_fscore_support_macro_all, train_precision_recall_fscore_support_weighted_all]
    testing = [test_balanced_accuracy_score_all, test_matthews_corrcoef_all, test_fbeta_score_micro_all, test_fbeta_score_macro_all, test_fbeta_score_weighted_all, test_precision_recall_fscore_support_micro_all, test_precision_recall_fscore_support_macro_all, test_precision_recall_fscore_support_weighted_all]
    
    for n in n_neighbors: 
        for a in alphas: 
            clf = LabelSpreading(kernel='knn', gamma=0, n_neighbors=n, alpha=a, n_jobs=-1)
            # train on training set  
            clf.fit(X_train, y_train)
            # training set 
            y_pred = clf.predict(X_train)    
           # multiclass evaluation scores 
            train_balanced_accuracy_score_all.append(balanced_accuracy_score(y_train, y_pred))
            train_confusion_matrix_all.append(confusion_matrix(y_train, y_pred))
            train_multilabel_confusion_matrix_all.append(multilabel_confusion_matrix(y_train, y_pred))
            # d = hinge_loss(y_train, y_pred, labels)
            train_matthews_corrcoef_all.append(matthews_corrcoef(y_train, y_pred))
            # f= roc_auc_score(y_train, y_pred, average='micro', multi_class='ovo')
            train_fbeta_score_micro_all.append(fbeta_score(y_train, y_pred, average='micro', beta=1))
            train_fbeta_score_macro_all.append(fbeta_score(y_train, y_pred, average='macro', beta=1))
            train_fbeta_score_weighted_all.append(fbeta_score(y_train, y_pred, average='weighted', beta=1))
            train_precision_recall_fscore_support_micro_all.append(precision_recall_fscore_support(y_train, y_pred, beta=1, average='micro'))
            train_precision_recall_fscore_support_macro_all.append(precision_recall_fscore_support(y_train, y_pred, beta=1, average='macro'))
            train_precision_recall_fscore_support_weighted_all.append(precision_recall_fscore_support(y_train, y_pred, beta=1, average='weighted'))
            
            # test set 
            y_pred2 = clf.predict(X_test)  
            # multiclass evaluation scores
            test_balanced_accuracy_score_all.append(balanced_accuracy_score(y_test, y_pred2))
            test_confusion_matrix_all.append(confusion_matrix(y_test, y_pred2))
            test_multilabel_confusion_matrix_all.append(multilabel_confusion_matrix(y_test, y_pred2))
            # d2 = hinge_loss(y_test, labels=y_pred)
            test_matthews_corrcoef_all.append(matthews_corrcoef(y_test, y_pred2))
            # f2= roc_auc_score(y_test, y_pred, average='micro', multi_class='ovo')
            test_fbeta_score_micro_all.append(fbeta_score(y_test, y_pred2, average='micro', beta=1))
            test_fbeta_score_macro_all.append(fbeta_score(y_test, y_pred2, average='macro', beta=1))
            test_fbeta_score_weighted_all.append(fbeta_score(y_test, y_pred2, average='weighted', beta=1))
            test_precision_recall_fscore_support_micro_all.append(precision_recall_fscore_support(y_test, y_pred2, beta=1, average='micro'))
            test_precision_recall_fscore_support_macro_all.append(precision_recall_fscore_support(y_test, y_pred2, beta=1, average='macro'))
            test_precision_recall_fscore_support_weighted_all.append(precision_recall_fscore_support(y_test, y_pred2, beta=1, average='weighted'))    
     
            params.append(f'n_neighbors: {n}, alphas: {a}')
                    
    
    # get max score's parameters 
    print('Highest score: ', max(test_balanced_accuracy_score_all))
    print(params[test_balanced_accuracy_score_all.index(max(test_balanced_accuracy_score_all))])

    max_index = test_balanced_accuracy_score_all.index(max(test_balanced_accuracy_score_all))
    
    train_scores = []
    for i in training: 
        i = i[max_index]
        if type(i) == tuple: 
            prec = i[0]
            recall = i[1]
            f1 = i[2]
            train_scores.append(prec)
            train_scores.append(recall)
            train_scores.append(f1)
        else:  
            train_scores.append(i)
    
    test_scores = []
    for i in testing: 
        i = i[max_index]
        if type(i) == tuple: 
            prec = i[0]
            recall = i[1]
            f1 = i[2]
            test_scores.append(prec)
            test_scores.append(recall)
            test_scores.append(f1)
        else:  
            test_scores.append(i)
    
    score_names =  ['balanced_accuracy', 'matthews_corrcoef', 'fbeta_score_micro', 'fbeta_score_macro', 'fbeta_score_weighted', 'precision_micro', 'recall_micro', 'f1_micro', 'precision_macro', 'precision_macro', 'f1_macro', 'precision_weighted', 'recall_weighted', 'f1_weighted']
    
    
    assert len(train_scores) == len(score_names)
    assert len(test_scores) == len(score_names)
          
    testdot1_17 = pd.DataFrame({'classifier': clf.__class__.__name__
                             , 'training': train_scores
                            , 'test': test_scores
                            , 'score': score_names})
    return testdot1_17



def cross_validation_nusvc(X, y, cv=5):  

    params = []
    
    score_names_all = []
    test_scores_all = []
    train_scores_all = []
    
    nu = [0.9999999]
    kernel = ['linear']
    gamma = ['scale', 'auto']
    shrinking = [True, False]
    class_weight = [None, 'balanced']
    decision_function_shape = ['ovo', 'ovr'] 
    
    for n in nu: 
        for k in kernel:
            for g in gamma: 
                for sh in shrinking: 
                    for w in class_weight: 
                        for d in decision_function_shape:                   
                            clf = NuSVC(nu=n, kernel=k, gamma=g, shrinking=sh, class_weight=w, decision_function_shape=d)
                            scoring = ['balanced_accuracy', 'f1_micro', 'f1_macro', 'f1_weighted', 'precision_micro', 'precision_macro', 'precision_weighted','recall_micro', 'recall_macro', 'recall_weighted', 'roc_auc_ovr', 'roc_auc_ovo', 'roc_auc_ovr_weighted', 'roc_auc_ovo_weighted']
                            # train on training set  
                            scores = cross_validate(clf, X, y, cv=5, scoring=scoring, return_train_score=True)
                            score_names = []
                            test_scores = [] 
                            train_scores = []        
                            for el in sorted(scores.keys()): 
                                if el.startswith('test') and not el.endswith('matrix'): 
                                    test_scores.append(scores[el].mean())
                                    score_names.append(el[5:])
                                elif el.startswith('train') and not el.endswith('matrix'): 
                                    train_scores.append(scores[el].mean())
                            assert len(train_scores) == len(test_scores)
                            assert len(train_scores) == len(score_names)
                        
                            score_names_all.append(score_names)
                            test_scores_all.append(test_scores)
                            train_scores_all.append(train_scores)
                            params.append(f'kernel: {k}, gamma: {g}, shrinking: {sh}, class_weight: {w}, decision_function_shape: {d}')
         
            
    # get max score's parameters 
    for el in test_scores_all: 
        if el[score_names.index('balanced_accuracy')] == max([i[score_names.index('balanced_accuracy')] for i in test_scores_all]): 
            windi = test_scores_all.index(el)
    
    print('Highest score: ', max([i[score_names.index('balanced_accuracy')] for i in test_scores_all]))
    print(params[windi])
 
    train_scores = train_scores_all[score_names.index('balanced_accuracy')]
    test_scores = test_scores_all[score_names.index('balanced_accuracy')]
    score_names =  score_names_all[score_names.index('balanced_accuracy')]
    
    cv5_18 = pd.DataFrame({'classifier': clf.__class__.__name__
                             , 'training': train_scores
                            , 'test': test_scores
                            , 'score': score_names})
    return cv5_18

def train_test_split_nusvc(X_train, y_train): 
    params = []
    
    nu = [0.9999999]
    kernel = ['linear']
    gamma = ['scale', 'auto']
    shrinking = [True, False]
    class_weight = [None, 'balanced']
    decision_function_shape = ['ovo', 'ovr'] 
    
    
    # train-test split
    train_balanced_accuracy_score_all = []
    train_confusion_matrix_all = []
    train_multilabel_confusion_matrix_all = []
    train_matthews_corrcoef_all = []
    train_fbeta_score_micro_all = []
    train_fbeta_score_macro_all = []
    train_fbeta_score_weighted_all = []
    train_precision_recall_fscore_support_micro_all =[]
    train_precision_recall_fscore_support_macro_all = []
    train_precision_recall_fscore_support_weighted_all = []
    test_balanced_accuracy_score_all= []
    test_confusion_matrix_all= []
    test_multilabel_confusion_matrix_all= []
    test_matthews_corrcoef_all= []
    test_fbeta_score_micro_all= []
    test_fbeta_score_macro_all= []
    test_fbeta_score_weighted_all= []
    test_precision_recall_fscore_support_micro_all= []
    test_precision_recall_fscore_support_macro_all= []
    test_precision_recall_fscore_support_weighted_all= []
    training = [train_balanced_accuracy_score_all, train_matthews_corrcoef_all, train_fbeta_score_micro_all, train_fbeta_score_macro_all, train_fbeta_score_weighted_all, train_precision_recall_fscore_support_micro_all, train_precision_recall_fscore_support_macro_all, train_precision_recall_fscore_support_weighted_all]
    testing = [test_balanced_accuracy_score_all, test_matthews_corrcoef_all, test_fbeta_score_micro_all, test_fbeta_score_macro_all, test_fbeta_score_weighted_all, test_precision_recall_fscore_support_micro_all, test_precision_recall_fscore_support_macro_all, test_precision_recall_fscore_support_weighted_all]
    
         
            
    for n in nu: 
        for k in kernel:
            for g in gamma: 
                for sh in shrinking: 
                    for w in class_weight: 
                        for d in decision_function_shape:                   
                            clf = NuSVC(nu=n, kernel=k, gamma=g, shrinking=sh, class_weight=w, decision_function_shape=d)
                            # train on training set  
                            clf.fit(X_train, y_train)
                            # training set 
                            y_pred = clf.predict(X_train)    
                                    # multiclass evaluation scores 
                            train_balanced_accuracy_score_all.append(balanced_accuracy_score(y_train, y_pred))
                            train_confusion_matrix_all.append(confusion_matrix(y_train, y_pred))
                            train_multilabel_confusion_matrix_all.append(multilabel_confusion_matrix(y_train, y_pred))
                            # d = hinge_loss(y_train, y_pred, labels)
                            train_matthews_corrcoef_all.append(matthews_corrcoef(y_train, y_pred))
                            # f= roc_auc_score(y_train, y_pred, average='micro', multi_class='ovo')
                            train_fbeta_score_micro_all.append(fbeta_score(y_train, y_pred, average='micro', beta=1))
                            train_fbeta_score_macro_all.append(fbeta_score(y_train, y_pred, average='macro', beta=1))
                            train_fbeta_score_weighted_all.append(fbeta_score(y_train, y_pred, average='weighted', beta=1))
                            train_precision_recall_fscore_support_micro_all.append(precision_recall_fscore_support(y_train, y_pred, beta=1, average='micro'))
                            train_precision_recall_fscore_support_macro_all.append(precision_recall_fscore_support(y_train, y_pred, beta=1, average='macro'))
                            train_precision_recall_fscore_support_weighted_all.append(precision_recall_fscore_support(y_train, y_pred, beta=1, average='weighted'))
                            
                            # test set 
                            y_pred2 = clf.predict(X_test)  
                            # multiclass evaluation scores
                            test_balanced_accuracy_score_all.append(balanced_accuracy_score(y_test, y_pred2))
                            test_confusion_matrix_all.append(confusion_matrix(y_test, y_pred2))
                            test_multilabel_confusion_matrix_all.append(multilabel_confusion_matrix(y_test, y_pred2))
                            # d2 = hinge_loss(y_test, labels=y_pred)
                            test_matthews_corrcoef_all.append(matthews_corrcoef(y_test, y_pred2))
                            # f2= roc_auc_score(y_test, y_pred, average='micro', multi_class='ovo')
                            test_fbeta_score_micro_all.append(fbeta_score(y_test, y_pred2, average='micro', beta=1))
                            test_fbeta_score_macro_all.append(fbeta_score(y_test, y_pred2, average='macro', beta=1))
                            test_fbeta_score_weighted_all.append(fbeta_score(y_test, y_pred2, average='weighted', beta=1))
                            test_precision_recall_fscore_support_micro_all.append(precision_recall_fscore_support(y_test, y_pred2, beta=1, average='micro'))
                            test_precision_recall_fscore_support_macro_all.append(precision_recall_fscore_support(y_test, y_pred2, beta=1, average='macro'))
                            test_precision_recall_fscore_support_weighted_all.append(precision_recall_fscore_support(y_test, y_pred2, beta=1, average='weighted'))    
                           
                            params.append(f'kernel: {k}, gamma: {g}, shrinking: {sh}, class_weight: {w}, decision_function_shape: {d}')
                                    
    
    # get max score's parameters 
    print('Highest score: ', max(test_balanced_accuracy_score_all))
    print(params[test_balanced_accuracy_score_all.index(max(test_balanced_accuracy_score_all))])
    # print('Number of combinations: ', len(solver)*len(multi_class))
    
    max_index = test_balanced_accuracy_score_all.index(max(test_balanced_accuracy_score_all))
    
    train_scores = []
    for i in training: 
        i = i[max_index]
        if type(i) == tuple: 
            prec = i[0]
            recall = i[1]
            f1 = i[2]
            train_scores.append(prec)
            train_scores.append(recall)
            train_scores.append(f1)
        else:  
            train_scores.append(i)
    
    test_scores = []
    for i in testing: 
        i = i[max_index]
        if type(i) == tuple: 
            prec = i[0]
            recall = i[1]
            f1 = i[2]
            test_scores.append(prec)
            test_scores.append(recall)
            test_scores.append(f1)
        else:  
            test_scores.append(i)
    
    score_names =  ['balanced_accuracy', 'matthews_corrcoef', 'fbeta_score_micro', 'fbeta_score_macro', 'fbeta_score_weighted', 'precision_micro', 'recall_micro', 'f1_micro', 'precision_macro', 'precision_macro', 'f1_macro', 'precision_weighted', 'recall_weighted', 'f1_weighted']
       
    assert len(train_scores) == len(score_names)
    assert len(test_scores) == len(score_names)
          
    testdot1_18 = pd.DataFrame({'classifier': clf.__class__.__name__
                             , 'training': train_scores
                            , 'test': test_scores
                            , 'score': score_names})
    return testdot1_18

def cross_validation_svc(X, y, cv=5): 
    params = []
    
    score_names_all = []
    test_scores_all = []
    train_scores_all = []
    
    C = [1, 10, 20]
    kernel = ['linear', 'poly', 'rbf', 'sigmoid']
    gamma = ['scale', 'auto']
    shrinking = [True, False]
    class_weight = [None, 'balanced']
    decision_function_shape = ['ovo', 'ovr']
    
    for c in C: 
        for k in kernel:
            for g in gamma: 
                for sh in shrinking: 
                    for w in class_weight: 
                        for d in decision_function_shape:                   
                            clf = SVC(C=c, kernel=k, gamma=g, shrinking=sh,  probability=True, class_weight=w, decision_function_shape=d, random_state=24)
                            scoring = ['balanced_accuracy', 'f1_micro', 'f1_macro', 'f1_weighted', 'precision_micro', 'precision_macro', 'precision_weighted','recall_micro', 'recall_macro', 'recall_weighted', 'roc_auc_ovr', 'roc_auc_ovo', 'roc_auc_ovr_weighted', 'roc_auc_ovo_weighted']
                            # train on training set  
                            scores = cross_validate(clf, X, y, cv=5, scoring=scoring, return_train_score=True)
                            score_names = []
                            test_scores = [] 
                            train_scores = []        
                            for el in sorted(scores.keys()): 
                                if el.startswith('test') and not el.endswith('matrix'): 
                                    test_scores.append(scores[el].mean())
                                    score_names.append(el[5:])
                                elif el.startswith('train') and not el.endswith('matrix'): 
                                    train_scores.append(scores[el].mean())
                            assert len(train_scores) == len(test_scores)
                            assert len(train_scores) == len(score_names)
                        
                            score_names_all.append(score_names)
                            test_scores_all.append(test_scores)
                            train_scores_all.append(train_scores)
                            params.append(f'C: {c}, gamma: {g}, kernel: {k}, shrinking: {sh}, class_weight: {w}, decision_function_shape: {d}')
                                    
    
    # get max score's parameters 
    for el in test_scores_all: 
        if el[score_names.index('balanced_accuracy')] == max([i[score_names.index('balanced_accuracy')] for i in test_scores_all]): 
            windi = test_scores_all.index(el)
    
    print('Highest score: ', max([i[score_names.index('balanced_accuracy')] for i in test_scores_all]))
    print(params[windi])
    # print('Number of combinations: ', len(solver)*len(multi_class))
    
    nb =  max([i[score_names.index('balanced_accuracy')] for i in test_scores_all])
    for el in test_scores_all: 
        if el[score_names.index('balanced_accuracy')] == nb: 
            test_scores = el
    nb =  max([i[score_names.index('balanced_accuracy')] for i in train_scores_all])
    for el in train_scores_all: 
        if el[score_names.index('balanced_accuracy')] == nb: 
            train_scores = el
    score_names =  score_names_all[score_names.index('balanced_accuracy')]
    
    cv5_19 = pd.DataFrame({'classifier': clf.__class__.__name__
                             , 'training': train_scores
                            , 'test': test_scores
                            , 'score': score_names})
    return cv5_19


    

def train_test_split_svc(X_train, y_train): 
    params = []
    
    C = [1, 10, 20]
    kernel = ['linear', 'poly', 'rbf', 'sigmoid']
    gamma = ['scale', 'auto']
    shrinking = [True, False]
    class_weight = [None, 'balanced']
    decision_function_shape = ['ovo', 'ovr']
    print('Number of combinations: ', len(C)*len(kernel)*len(gamma)*len(shrinking)*len(class_weight)*len(decision_function_shape))
    
    # train-test split
    train_balanced_accuracy_score_all = []
    train_confusion_matrix_all = []
    train_multilabel_confusion_matrix_all = []
    train_matthews_corrcoef_all = []
    train_fbeta_score_micro_all = []
    train_fbeta_score_macro_all = []
    train_fbeta_score_weighted_all = []
    train_precision_recall_fscore_support_micro_all =[]
    train_precision_recall_fscore_support_macro_all = []
    train_precision_recall_fscore_support_weighted_all = []
    test_balanced_accuracy_score_all= []
    test_confusion_matrix_all= []
    test_multilabel_confusion_matrix_all= []
    test_matthews_corrcoef_all= []
    test_fbeta_score_micro_all= []
    test_fbeta_score_macro_all= []
    test_fbeta_score_weighted_all= []
    test_precision_recall_fscore_support_micro_all= []
    test_precision_recall_fscore_support_macro_all= []
    test_precision_recall_fscore_support_weighted_all= []
    training = [train_balanced_accuracy_score_all, train_matthews_corrcoef_all, train_fbeta_score_micro_all, train_fbeta_score_macro_all, train_fbeta_score_weighted_all, train_precision_recall_fscore_support_micro_all, train_precision_recall_fscore_support_macro_all, train_precision_recall_fscore_support_weighted_all]
    testing = [test_balanced_accuracy_score_all, test_matthews_corrcoef_all, test_fbeta_score_micro_all, test_fbeta_score_macro_all, test_fbeta_score_weighted_all, test_precision_recall_fscore_support_micro_all, test_precision_recall_fscore_support_macro_all, test_precision_recall_fscore_support_weighted_all]
       
    for c in C: 
        for k in kernel:
            for g in gamma: 
                for sh in shrinking: 
                    for w in class_weight: 
                        for d in decision_function_shape:                   
                            clf = SVC(C=c, kernel=k, gamma=g, shrinking=sh, probability=True, class_weight=w, decision_function_shape=d, random_state=24)
                            # train on training set  
                            clf.fit(X_train, y_train)
                            # training set 
                            y_pred = clf.predict(X_train)    
                            # multiclass evaluation scores 
                            train_balanced_accuracy_score_all.append(balanced_accuracy_score(y_train, y_pred))
                            train_confusion_matrix_all.append(confusion_matrix(y_train, y_pred))
                            train_multilabel_confusion_matrix_all.append(multilabel_confusion_matrix(y_train, y_pred))
                            # d = hinge_loss(y_train, y_pred, labels)
                            train_matthews_corrcoef_all.append(matthews_corrcoef(y_train, y_pred))
                            # f= roc_auc_score(y_train, y_pred, average='micro', multi_class='ovo')
                            train_fbeta_score_micro_all.append(fbeta_score(y_train, y_pred, average='micro', beta=1))
                            train_fbeta_score_macro_all.append(fbeta_score(y_train, y_pred, average='macro', beta=1))
                            train_fbeta_score_weighted_all.append(fbeta_score(y_train, y_pred, average='weighted', beta=1))
                            train_precision_recall_fscore_support_micro_all.append(precision_recall_fscore_support(y_train, y_pred, beta=1, average='micro'))
                            train_precision_recall_fscore_support_macro_all.append(precision_recall_fscore_support(y_train, y_pred, beta=1, average='macro'))
                            train_precision_recall_fscore_support_weighted_all.append(precision_recall_fscore_support(y_train, y_pred, beta=1, average='weighted'))
                            
                            # test set 
                            y_pred2 = clf.predict(X_test)  
                            # multiclass evaluation scores
                            test_balanced_accuracy_score_all.append(balanced_accuracy_score(y_test, y_pred2))
                            test_confusion_matrix_all.append(confusion_matrix(y_test, y_pred2))
                            test_multilabel_confusion_matrix_all.append(multilabel_confusion_matrix(y_test, y_pred2))
                            # d2 = hinge_loss(y_test, labels=y_pred)
                            test_matthews_corrcoef_all.append(matthews_corrcoef(y_test, y_pred2))
                            # f2= roc_auc_score(y_test, y_pred, average='micro', multi_class='ovo')
                            test_fbeta_score_micro_all.append(fbeta_score(y_test, y_pred2, average='micro', beta=1))
                            test_fbeta_score_macro_all.append(fbeta_score(y_test, y_pred2, average='macro', beta=1))
                            test_fbeta_score_weighted_all.append(fbeta_score(y_test, y_pred2, average='weighted', beta=1))
                            test_precision_recall_fscore_support_micro_all.append(precision_recall_fscore_support(y_test, y_pred2, beta=1, average='micro'))
                            test_precision_recall_fscore_support_macro_all.append(precision_recall_fscore_support(y_test, y_pred2, beta=1, average='macro'))
                            test_precision_recall_fscore_support_weighted_all.append(precision_recall_fscore_support(y_test, y_pred2, beta=1, average='weighted'))    
                       
                            params.append(f'C: {c}, gamma: {g}, shrinking: {sh}, class_weight: {w}, decision_function_shape: {d}')
                                    
    # get max score's parameters 
    print('Highest score: ', max(test_balanced_accuracy_score_all))
    print(params[test_balanced_accuracy_score_all.index(max(test_balanced_accuracy_score_all))])

    max_index = test_balanced_accuracy_score_all.index(max(test_balanced_accuracy_score_all))
    
    train_scores = []
    for i in training: 
        i = i[max_index]
        if type(i) == tuple: 
            prec = i[0]
            recall = i[1]
            f1 = i[2]
            train_scores.append(prec)
            train_scores.append(recall)
            train_scores.append(f1)
        else:  
            train_scores.append(i)
    
    test_scores = []
    for i in testing: 
        i = i[max_index]
        if type(i) == tuple: 
            prec = i[0]
            recall = i[1]
            f1 = i[2]
            test_scores.append(prec)
            test_scores.append(recall)
            test_scores.append(f1)
        else:  
            test_scores.append(i)
    
    score_names =  ['balanced_accuracy', 'matthews_corrcoef', 'fbeta_score_micro', 'fbeta_score_macro', 'fbeta_score_weighted', 'precision_micro', 'recall_micro', 'f1_micro', 'precision_macro', 'precision_macro', 'f1_macro', 'precision_weighted', 'recall_weighted', 'f1_weighted']
    
    
    assert len(train_scores) == len(score_names)
    assert len(test_scores) == len(score_names)
          
    testdot1_19 = pd.DataFrame({'classifier': clf.__class__.__name__
                             , 'training': train_scores
                            , 'test': test_scores
                            , 'score': score_names})
    return testdot1_19

def cross_validation_gaussianprocessclassifier(X, y, cv=5): 
  
    score_names_all = []
    test_scores_all = []
    train_scores_all = []
    
    clf = GaussianProcessClassifier(copy_X_train=False, random_state=24, multi_class='one_vs_rest',  n_jobs=-1)
    
    scoring = ['balanced_accuracy', 'f1_micro', 'f1_macro', 'f1_weighted', 'precision_micro', 'precision_macro', 'precision_weighted','recall_micro', 'recall_macro', 'recall_weighted', 'roc_auc_ovr', 'roc_auc_ovo', 'roc_auc_ovr_weighted', 'roc_auc_ovo_weighted']
    # train on training set  
    scores = cross_validate(clf, X, y, cv=5, scoring=scoring, return_train_score=True)
    score_names = []
    test_scores = [] 
    train_scores = []        
    for el in sorted(scores.keys()): 
        if el.startswith('test') and not el.endswith('matrix'): 
            test_scores.append(scores[el].mean())
            score_names.append(el[5:])
        elif el.startswith('train') and not el.endswith('matrix'): 
            train_scores.append(scores[el].mean())
    assert len(train_scores) == len(test_scores)
    assert len(train_scores) == len(score_names)
    
    score_names_all.append(score_names)
    test_scores_all.append(test_scores)
    train_scores_all.append(train_scores)
    # params.append()
        
    # get max score's parameters 
    for el in test_scores_all: 
        if el[score_names.index('balanced_accuracy')] == max([i[score_names.index('balanced_accuracy')] for i in test_scores_all]): 
            windi = test_scores_all.index(el)
    
    print('Highest score: ', max([i[score_names.index('balanced_accuracy')] for i in test_scores_all]))
    # print(params[windi])
  
    nb =  max([i[score_names.index('balanced_accuracy')] for i in test_scores_all])
    for el in test_scores_all: 
        if el[score_names.index('balanced_accuracy')] == nb: 
            test_scores = el
    nb =  max([i[score_names.index('balanced_accuracy')] for i in train_scores_all])
    for el in train_scores_all: 
        if el[score_names.index('balanced_accuracy')] == nb: 
            train_scores = el
    score_names =  score_names_all[score_names.index('balanced_accuracy')]
    
    cv5_20 = pd.DataFrame({'classifier': clf.__class__.__name__
                             , 'training': train_scores
                            , 'test': test_scores
                            , 'score': score_names})
    return cv5_20

    
def train_test_split_gaussianprocessclassifier(X_train, y_train): 
 
    clf = GaussianProcessClassifier(copy_X_train=False, random_state=24, multi_class='one_vs_rest', n_jobs=-1) 
    clf.fit(X_train, y_train) 
    y_pred = clf.predict(X_train)    
    y_pred2 = clf.predict(X_test)  
    training, testing, test_balanced_accuracy_score_all = multi_class_scorer_train_test_split(y_train, y_pred, y_pred2)
    print('Highest score: ', max(test_balanced_accuracy_score_all))

    max_index = test_balanced_accuracy_score_all.index(max(test_balanced_accuracy_score_all))
    
    train_scores = scores(training, max_index)
    test_scores = scores(testing, max_index)
    
    score_names =  ['balanced_accuracy', 'matthews_corrcoef', 'fbeta_score_micro', 'fbeta_score_macro', 'fbeta_score_weighted', 'precision_micro', 'recall_micro', 'f1_micro', 'precision_macro', 'precision_macro', 'f1_macro', 'precision_weighted', 'recall_weighted', 'f1_weighted']
       
    assert len(train_scores) == len(score_names)
    assert len(test_scores) == len(score_names)
          
    testdot1_20 = pd.DataFrame({'classifier': clf.__class__.__name__
                             , 'training': train_scores
                            , 'test': test_scores
                            , 'score': score_names})
    return testdot1_20


def cross_validation_sgdclassifier(X, y, cv=5): 
    params = []
    
    score_names_all = []
    test_scores_all = []
    train_scores_all = []
    
    penalty = ['l2', 'elasticnet']
    learning_rate = ['optimal', 'adaptive']
    class_weight = [None, 'balanced']
    eta0 = [0.8, .9, 1, 1.1, 1.2, 1.4]
    print('Number of combinations: ', len(penalty)*len(learning_rate)*len(class_weight)*len(eta0))
 
    for p in penalty:
        for lr in learning_rate: 
            for c in class_weight: 
                for eta in eta0: 
                    clf = SGDClassifier(penalty=p, n_jobs=-1, random_state=24, learning_rate=lr, eta0=eta, class_weight=c)
                    scoring = ['balanced_accuracy', 'f1_micro', 'f1_macro', 'f1_weighted', 'precision_micro', 'precision_macro', 'precision_weighted','recall_micro', 'recall_macro', 'recall_weighted', 'roc_auc_ovr', 'roc_auc_ovo', 'roc_auc_ovr_weighted', 'roc_auc_ovo_weighted']
                    # train on training set  
                    scores = cross_validate(clf, X, y, cv=5, scoring=scoring, return_train_score=True)
                    score_names = []
                    test_scores = [] 
                    train_scores = []        
                    for el in sorted(scores.keys()): 
                        if el.startswith('test') and not el.endswith('matrix'): 
                            test_scores.append(scores[el].mean())
                            score_names.append(el[5:])
                        elif el.startswith('train') and not el.endswith('matrix'): 
                            train_scores.append(scores[el].mean())
                    assert len(train_scores) == len(test_scores)
                    assert len(train_scores) == len(score_names)
                
                    score_names_all.append(score_names)
                    test_scores_all.append(test_scores)
                    train_scores_all.append(train_scores)
                    params.append(f'penalty: {p}, learning_rate: {lr}, class_weight: {c}, eta0: {eta}')
                    
    # get max score's parameters 
    for el in test_scores_all: 
        if el[score_names.index('balanced_accuracy')] == max([i[score_names.index('balanced_accuracy')] for i in test_scores_all]): 
            windi = test_scores_all.index(el)
    
    print('Highest score: ', max([i[score_names.index('balanced_accuracy')] for i in test_scores_all]))
    print(params[windi])
   
    train_scores = train_scores_all[score_names.index('balanced_accuracy')]
    test_scores = test_scores_all[score_names.index('balanced_accuracy')]
    score_names =  score_names_all[score_names.index('balanced_accuracy')]
    
    cv5_21 = pd.DataFrame({'classifier': clf.__class__.__name__
                             , 'training': train_scores
                            , 'test': test_scores
                            , 'score': score_names})
    return cv5_21

def train_test_split_sgdclassifier(X_train, y_train): 
    
    params = []
    
    # train-test split
    train_balanced_accuracy_score_all = []
    train_confusion_matrix_all = []
    train_multilabel_confusion_matrix_all = []
    train_matthews_corrcoef_all = []
    train_fbeta_score_micro_all = []
    train_fbeta_score_macro_all = []
    train_fbeta_score_weighted_all = []
    train_precision_recall_fscore_support_micro_all =[]
    train_precision_recall_fscore_support_macro_all = []
    train_precision_recall_fscore_support_weighted_all = []
    test_balanced_accuracy_score_all= []
    test_confusion_matrix_all= []
    test_multilabel_confusion_matrix_all= []
    test_matthews_corrcoef_all= []
    test_fbeta_score_micro_all= []
    test_fbeta_score_macro_all= []
    test_fbeta_score_weighted_all= []
    test_precision_recall_fscore_support_micro_all= []
    test_precision_recall_fscore_support_macro_all= []
    test_precision_recall_fscore_support_weighted_all= []
    training = [train_balanced_accuracy_score_all, train_matthews_corrcoef_all, train_fbeta_score_micro_all, train_fbeta_score_macro_all, train_fbeta_score_weighted_all, train_precision_recall_fscore_support_micro_all, train_precision_recall_fscore_support_macro_all, train_precision_recall_fscore_support_weighted_all]
    testing = [test_balanced_accuracy_score_all, test_matthews_corrcoef_all, test_fbeta_score_micro_all, test_fbeta_score_macro_all, test_fbeta_score_weighted_all, test_precision_recall_fscore_support_micro_all, test_precision_recall_fscore_support_macro_all, test_precision_recall_fscore_support_weighted_all]
                    
                    
    for p in penalty:
        for lr in learning_rate: 
            for c in class_weight: 
                for eta in eta0: 
                    clf = SGDClassifier(penalty=p, n_jobs=-1, random_state=24, learning_rate=lr, eta0=eta, class_weight=c)
                    # train on training set  
                    clf.fit(X_train, y_train)
                    # training set 
                    y_pred = clf.predict(X_train)    
                    # multiclass evaluation scores 
                    train_balanced_accuracy_score_all.append(balanced_accuracy_score(y_train, y_pred))
                    train_confusion_matrix_all.append(confusion_matrix(y_train, y_pred))
                    train_multilabel_confusion_matrix_all.append(multilabel_confusion_matrix(y_train, y_pred))
                    # d = hinge_loss(y_train, y_pred, labels)
                    train_matthews_corrcoef_all.append(matthews_corrcoef(y_train, y_pred))
                    # f= roc_auc_score(y_train, y_pred, average='micro', multi_class='ovo')
                    train_fbeta_score_micro_all.append(fbeta_score(y_train, y_pred, average='micro', beta=1))
                    train_fbeta_score_macro_all.append(fbeta_score(y_train, y_pred, average='macro', beta=1))
                    train_fbeta_score_weighted_all.append(fbeta_score(y_train, y_pred, average='weighted', beta=1))
                    train_precision_recall_fscore_support_micro_all.append(precision_recall_fscore_support(y_train, y_pred, beta=1, average='micro'))
                    train_precision_recall_fscore_support_macro_all.append(precision_recall_fscore_support(y_train, y_pred, beta=1, average='macro'))
                    train_precision_recall_fscore_support_weighted_all.append(precision_recall_fscore_support(y_train, y_pred, beta=1, average='weighted'))
                    
                    # test set 
                    y_pred2 = clf.predict(X_test)  
                    # multiclass evaluation scores
                    test_balanced_accuracy_score_all.append(balanced_accuracy_score(y_test, y_pred2))
                    test_confusion_matrix_all.append(confusion_matrix(y_test, y_pred2))
                    test_multilabel_confusion_matrix_all.append(multilabel_confusion_matrix(y_test, y_pred2))
                    # d2 = hinge_loss(y_test, labels=y_pred)
                    test_matthews_corrcoef_all.append(matthews_corrcoef(y_test, y_pred2))
                    # f2= roc_auc_score(y_test, y_pred, average='micro', multi_class='ovo')
                    test_fbeta_score_micro_all.append(fbeta_score(y_test, y_pred2, average='micro', beta=1))
                    test_fbeta_score_macro_all.append(fbeta_score(y_test, y_pred2, average='macro', beta=1))
                    test_fbeta_score_weighted_all.append(fbeta_score(y_test, y_pred2, average='weighted', beta=1))
                    test_precision_recall_fscore_support_micro_all.append(precision_recall_fscore_support(y_test, y_pred2, beta=1, average='micro'))
                    test_precision_recall_fscore_support_macro_all.append(precision_recall_fscore_support(y_test, y_pred2, beta=1, average='macro'))
                    test_precision_recall_fscore_support_weighted_all.append(precision_recall_fscore_support(y_test, y_pred2, beta=1, average='weighted'))    
                  
                    params.append(f'penalty: {p}, learning_rate: {lr}, class_weight: {c}, eta0: {eta}')
                            
    
    # get max score's parameters 
    print('Highest score: ', max(test_balanced_accuracy_score_all))
    print(params[test_balanced_accuracy_score_all.index(max(test_balanced_accuracy_score_all))])
    # print('Number of combinations: ', len(solver)*len(multi_class))
    
    max_index = test_balanced_accuracy_score_all.index(max(test_balanced_accuracy_score_all))
    
    train_scores = []
    for i in training: 
        i = i[max_index]
        if type(i) == tuple: 
            prec = i[0]
            recall = i[1]
            f1 = i[2]
            train_scores.append(prec)
            train_scores.append(recall)
            train_scores.append(f1)
        else:  
            train_scores.append(i)
    
    test_scores = []
    for i in testing: 
        i = i[max_index]
        if type(i) == tuple: 
            prec = i[0]
            recall = i[1]
            f1 = i[2]
            test_scores.append(prec)
            test_scores.append(recall)
            test_scores.append(f1)
        else:  
            test_scores.append(i)
    
    score_names =  ['balanced_accuracy', 'matthews_corrcoef', 'fbeta_score_micro', 'fbeta_score_macro', 'fbeta_score_weighted', 'precision_micro', 'recall_micro', 'f1_micro', 'precision_macro', 'precision_macro', 'f1_macro', 'precision_weighted', 'recall_weighted', 'f1_weighted']
       
    assert len(train_scores) == len(score_names)
    assert len(test_scores) == len(score_names)
          
    testdot1_21 = pd.DataFrame({'classifier': clf.__class__.__name__
                             , 'training': train_scores
                            , 'test': test_scores
                            , 'score': score_names})
    return testdot1_21


def cross_validation_perceptron(X, y, cv=5): 
    multiclass_scores = {}
    params = []
     
    score_names_all = []
    test_scores_all = []
    train_scores_all = []
    
    penalty = ['l2', 'elasticnet']
    class_weight = [None, 'balanced']
    
    for p in penalty:
        for c in class_weight:  
            clf = Perceptron(penalty=p, n_jobs=-1, random_state=24, class_weight=c)
            scoring = ['balanced_accuracy', 'f1_micro', 'f1_macro', 'f1_weighted', 'precision_micro', 'precision_macro', 'precision_weighted','recall_micro', 'recall_macro', 'recall_weighted', 'roc_auc_ovr', 'roc_auc_ovo', 'roc_auc_ovr_weighted', 'roc_auc_ovo_weighted']
            # train on training set  
            scores = cross_validate(clf, X, y, cv=5, scoring=scoring, return_train_score=True)
            score_names = []
            test_scores = [] 
            train_scores = []        
            for el in sorted(scores.keys()): 
                if el.startswith('test') and not el.endswith('matrix'): 
                    test_scores.append(scores[el].mean())
                    score_names.append(el[5:])
                elif el.startswith('train') and not el.endswith('matrix'): 
                    train_scores.append(scores[el].mean())
            assert len(train_scores) == len(test_scores)
            assert len(train_scores) == len(score_names)
        
            score_names_all.append(score_names)
            test_scores_all.append(test_scores)
            train_scores_all.append(train_scores)
            params.append(f'penalty: {p}, class_weight: {c}')
            
    # get max score's parameters 
    for el in test_scores_all: 
        if el[score_names.index('balanced_accuracy')] == max([i[score_names.index('balanced_accuracy')] for i in test_scores_all]): 
            windi = test_scores_all.index(el)
    
    print('Highest score: ', max([i[score_names.index('balanced_accuracy')] for i in test_scores_all]))
    print(params[windi])
    # print('Number of combinations: ', len(solver)*len(multi_class))
    
    train_scores = train_scores_all[score_names.index('balanced_accuracy')]
    test_scores = test_scores_all[score_names.index('balanced_accuracy')]
    score_names =  score_names_all[score_names.index('balanced_accuracy')]
    
    cv5_22 = pd.DataFrame({'classifier': clf.__class__.__name__
                             , 'training': train_scores
                            , 'test': test_scores
                            , 'score': score_names})

    return cv5_22


def train_test_split_perceptron(X_train, y_train): 
    
    params = []
    
    # train-test split
    train_balanced_accuracy_score_all = []
    train_confusion_matrix_all = []
    train_multilabel_confusion_matrix_all = []
    train_matthews_corrcoef_all = []
    train_fbeta_score_micro_all = []
    train_fbeta_score_macro_all = []
    train_fbeta_score_weighted_all = []
    train_precision_recall_fscore_support_micro_all =[]
    train_precision_recall_fscore_support_macro_all = []
    train_precision_recall_fscore_support_weighted_all = []
    test_balanced_accuracy_score_all= []
    test_confusion_matrix_all= []
    test_multilabel_confusion_matrix_all= []
    test_matthews_corrcoef_all= []
    test_fbeta_score_micro_all= []
    test_fbeta_score_macro_all= []
    test_fbeta_score_weighted_all= []
    test_precision_recall_fscore_support_micro_all= []
    test_precision_recall_fscore_support_macro_all= []
    test_precision_recall_fscore_support_weighted_all= []
    training = [train_balanced_accuracy_score_all, train_matthews_corrcoef_all, train_fbeta_score_micro_all, train_fbeta_score_macro_all, train_fbeta_score_weighted_all, train_precision_recall_fscore_support_micro_all, train_precision_recall_fscore_support_macro_all, train_precision_recall_fscore_support_weighted_all]
    testing = [test_balanced_accuracy_score_all, test_matthews_corrcoef_all, test_fbeta_score_micro_all, test_fbeta_score_macro_all, test_fbeta_score_weighted_all, test_precision_recall_fscore_support_micro_all, test_precision_recall_fscore_support_macro_all, test_precision_recall_fscore_support_weighted_all]
    
            
            
    
    for p in penalty:
        for c in class_weight:  
            clf = Perceptron(penalty=p, n_jobs=-1, random_state=24, class_weight=c)
            # train on training set  
            clf.fit(X_train, y_train)
            # training set 
            y_pred = clf.predict(X_train)    
    
            # multiclass evaluation scores 
            train_balanced_accuracy_score_all.append(balanced_accuracy_score(y_train, y_pred))
            train_confusion_matrix_all.append(confusion_matrix(y_train, y_pred))
            train_multilabel_confusion_matrix_all.append(multilabel_confusion_matrix(y_train, y_pred))
            # d = hinge_loss(y_train, y_pred, labels)
            train_matthews_corrcoef_all.append(matthews_corrcoef(y_train, y_pred))
            # f= roc_auc_score(y_train, y_pred, average='micro', multi_class='ovo')
            train_fbeta_score_micro_all.append(fbeta_score(y_train, y_pred, average='micro', beta=1))
            train_fbeta_score_macro_all.append(fbeta_score(y_train, y_pred, average='macro', beta=1))
            train_fbeta_score_weighted_all.append(fbeta_score(y_train, y_pred, average='weighted', beta=1))
            train_precision_recall_fscore_support_micro_all.append(precision_recall_fscore_support(y_train, y_pred, beta=1, average='micro'))
            train_precision_recall_fscore_support_macro_all.append(precision_recall_fscore_support(y_train, y_pred, beta=1, average='macro'))
            train_precision_recall_fscore_support_weighted_all.append(precision_recall_fscore_support(y_train, y_pred, beta=1, average='weighted'))
            
            # test set 
            y_pred2 = clf.predict(X_test)  
            # multiclass evaluation scores
            test_balanced_accuracy_score_all.append(balanced_accuracy_score(y_test, y_pred2))
            test_confusion_matrix_all.append(confusion_matrix(y_test, y_pred2))
            test_multilabel_confusion_matrix_all.append(multilabel_confusion_matrix(y_test, y_pred2))
            # d2 = hinge_loss(y_test, labels=y_pred)
            test_matthews_corrcoef_all.append(matthews_corrcoef(y_test, y_pred2))
            # f2= roc_auc_score(y_test, y_pred, average='micro', multi_class='ovo')
            test_fbeta_score_micro_all.append(fbeta_score(y_test, y_pred2, average='micro', beta=1))
            test_fbeta_score_macro_all.append(fbeta_score(y_test, y_pred2, average='macro', beta=1))
            test_fbeta_score_weighted_all.append(fbeta_score(y_test, y_pred2, average='weighted', beta=1))
            test_precision_recall_fscore_support_micro_all.append(precision_recall_fscore_support(y_test, y_pred2, beta=1, average='micro'))
            test_precision_recall_fscore_support_macro_all.append(precision_recall_fscore_support(y_test, y_pred2, beta=1, average='macro'))
            test_precision_recall_fscore_support_weighted_all.append(precision_recall_fscore_support(y_test, y_pred2, beta=1, average='weighted'))    
           
            params.append(f'penalty: {p}, class_weight: {c}')
                    
    
    # get max score's parameters 
    print('Highest score: ', max(test_balanced_accuracy_score_all))
    print(params[test_balanced_accuracy_score_all.index(max(test_balanced_accuracy_score_all))])
    # print('Number of combinations: ', len(solver)*len(multi_class))
    
    max_index = test_balanced_accuracy_score_all.index(max(test_balanced_accuracy_score_all))
    
    train_scores = []
    for i in training: 
        i = i[max_index]
        if type(i) == tuple: 
            prec = i[0]
            recall = i[1]
            f1 = i[2]
            train_scores.append(prec)
            train_scores.append(recall)
            train_scores.append(f1)
        else:  
            train_scores.append(i)
    
    test_scores = []
    for i in testing: 
        i = i[max_index]
        if type(i) == tuple: 
            prec = i[0]
            recall = i[1]
            f1 = i[2]
            test_scores.append(prec)
            test_scores.append(recall)
            test_scores.append(f1)
        else:  
            test_scores.append(i)
    
    score_names =  ['balanced_accuracy', 'matthews_corrcoef', 'fbeta_score_micro', 'fbeta_score_macro', 'fbeta_score_weighted', 'precision_micro', 'recall_micro', 'f1_micro', 'precision_macro', 'precision_macro', 'f1_macro', 'precision_weighted', 'recall_weighted', 'f1_weighted']
    
    
    assert len(train_scores) == len(score_names)
    assert len(test_scores) == len(score_names)
          
    testdot1_22 = pd.DataFrame({'classifier': clf.__class__.__name__
                             , 'training': train_scores
                            , 'test': test_scores
                            , 'score': score_names})
    return testdot1_22


def cross_validation_passiveaggressiveclassifier(X, y, cv=5): 
        
    multiclass_scores = {}
    params = []
    
    score_names_all = []
    test_scores_all = []
    train_scores_all = []
    
    C = [.001, .01, .1, .2, .4, .6, .8, 1, 1.2]
    class_weight = [None, 'balanced']
    
    for c in C:
        for cl in class_weight:  
            clf = PassiveAggressiveClassifier(C=c, n_jobs=-1, random_state=24, class_weight=cl, average=True)
            scoring = ['balanced_accuracy', 'f1_micro', 'f1_macro', 'f1_weighted', 'precision_micro', 'precision_macro', 'precision_weighted','recall_micro', 'recall_macro', 'recall_weighted', 'roc_auc_ovr', 'roc_auc_ovo', 'roc_auc_ovr_weighted', 'roc_auc_ovo_weighted']
            # train on training set  
            scores = cross_validate(clf, X, y, cv=5, scoring=scoring, return_train_score=True)
            score_names = []
            test_scores = [] 
            train_scores = []        
            for el in sorted(scores.keys()): 
                if el.startswith('test') and not el.endswith('matrix'): 
                    test_scores.append(scores[el].mean())
                    score_names.append(el[5:])
                elif el.startswith('train') and not el.endswith('matrix'): 
                    train_scores.append(scores[el].mean())
            assert len(train_scores) == len(test_scores)
            assert len(train_scores) == len(score_names)
        
            score_names_all.append(score_names)
            test_scores_all.append(test_scores)
            train_scores_all.append(train_scores)
            params.append(f'C: {c}, class_weight: {cl}')
            
    # get max score's parameters 
    for el in test_scores_all: 
        if el[score_names.index('balanced_accuracy')] == max([i[score_names.index('balanced_accuracy')] for i in test_scores_all]): 
            windi = test_scores_all.index(el)
    
    print('Highest score: ', max([i[score_names.index('balanced_accuracy')] for i in test_scores_all]))
    print(params[windi])
    # print('Number of combinations: ', len(solver)*len(multi_class))
    
    train_scores = train_scores_all[score_names.index('balanced_accuracy')]
    test_scores = test_scores_all[score_names.index('balanced_accuracy')]
    score_names =  score_names_all[score_names.index('balanced_accuracy')]
    
    cv5_23 = pd.DataFrame({'classifier': clf.__class__.__name__
                             , 'training': train_scores
                            , 'test': test_scores
                            , 'score': score_names})
    return cv5_23

def train_test_split_passiveaggressiveclassifier(X_train, y_train): 
    
    params = []
    
    # train-test split
    train_balanced_accuracy_score_all = []
    train_confusion_matrix_all = []
    train_multilabel_confusion_matrix_all = []
    train_matthews_corrcoef_all = []
    train_fbeta_score_micro_all = []
    train_fbeta_score_macro_all = []
    train_fbeta_score_weighted_all = []
    train_precision_recall_fscore_support_micro_all =[]
    train_precision_recall_fscore_support_macro_all = []
    train_precision_recall_fscore_support_weighted_all = []
    test_balanced_accuracy_score_all= []
    test_confusion_matrix_all= []
    test_multilabel_confusion_matrix_all= []
    test_matthews_corrcoef_all= []
    test_fbeta_score_micro_all= []
    test_fbeta_score_macro_all= []
    test_fbeta_score_weighted_all= []
    test_precision_recall_fscore_support_micro_all= []
    test_precision_recall_fscore_support_macro_all= []
    test_precision_recall_fscore_support_weighted_all= []
    training = [train_balanced_accuracy_score_all, train_matthews_corrcoef_all, train_fbeta_score_micro_all, train_fbeta_score_macro_all, train_fbeta_score_weighted_all, train_precision_recall_fscore_support_micro_all, train_precision_recall_fscore_support_macro_all, train_precision_recall_fscore_support_weighted_all]
    testing = [test_balanced_accuracy_score_all, test_matthews_corrcoef_all, test_fbeta_score_micro_all, test_fbeta_score_macro_all, test_fbeta_score_weighted_all, test_precision_recall_fscore_support_micro_all, test_precision_recall_fscore_support_macro_all, test_precision_recall_fscore_support_weighted_all]
    
            
            
    for c in C:
        for cl in class_weight:  
            clf = PassiveAggressiveClassifier(C=c, n_jobs=-1, random_state=24, class_weight=cl, average=True)
            # train on training set  
            clf.fit(X_train, y_train)
            # training set 
            y_pred = clf.predict(X_train)    
            # multiclass evaluation scores 
            train_balanced_accuracy_score_all.append(balanced_accuracy_score(y_train, y_pred))
            train_confusion_matrix_all.append(confusion_matrix(y_train, y_pred))
            train_multilabel_confusion_matrix_all.append(multilabel_confusion_matrix(y_train, y_pred))
            # d = hinge_loss(y_train, y_pred, labels)
            train_matthews_corrcoef_all.append(matthews_corrcoef(y_train, y_pred))
            # f= roc_auc_score(y_train, y_pred, average='micro', multi_class='ovo')
            train_fbeta_score_micro_all.append(fbeta_score(y_train, y_pred, average='micro', beta=1))
            train_fbeta_score_macro_all.append(fbeta_score(y_train, y_pred, average='macro', beta=1))
            train_fbeta_score_weighted_all.append(fbeta_score(y_train, y_pred, average='weighted', beta=1))
            train_precision_recall_fscore_support_micro_all.append(precision_recall_fscore_support(y_train, y_pred, beta=1, average='micro'))
            train_precision_recall_fscore_support_macro_all.append(precision_recall_fscore_support(y_train, y_pred, beta=1, average='macro'))
            train_precision_recall_fscore_support_weighted_all.append(precision_recall_fscore_support(y_train, y_pred, beta=1, average='weighted'))
            
            # test set 
            y_pred2 = clf.predict(X_test)  
            # multiclass evaluation scores
            test_balanced_accuracy_score_all.append(balanced_accuracy_score(y_test, y_pred2))
            test_confusion_matrix_all.append(confusion_matrix(y_test, y_pred2))
            test_multilabel_confusion_matrix_all.append(multilabel_confusion_matrix(y_test, y_pred2))
            # d2 = hinge_loss(y_test, labels=y_pred)
            test_matthews_corrcoef_all.append(matthews_corrcoef(y_test, y_pred2))
            # f2= roc_auc_score(y_test, y_pred, average='micro', multi_class='ovo')
            test_fbeta_score_micro_all.append(fbeta_score(y_test, y_pred2, average='micro', beta=1))
            test_fbeta_score_macro_all.append(fbeta_score(y_test, y_pred2, average='macro', beta=1))
            test_fbeta_score_weighted_all.append(fbeta_score(y_test, y_pred2, average='weighted', beta=1))
            test_precision_recall_fscore_support_micro_all.append(precision_recall_fscore_support(y_test, y_pred2, beta=1, average='micro'))
            test_precision_recall_fscore_support_macro_all.append(precision_recall_fscore_support(y_test, y_pred2, beta=1, average='macro'))
            test_precision_recall_fscore_support_weighted_all.append(precision_recall_fscore_support(y_test, y_pred2, beta=1, average='weighted'))    
           
    
            params.append(f'C: {c}, class_weight: {cl}')
                    
    
    # get max score's parameters 
    print('Highest score: ', max(test_balanced_accuracy_score_all))
    print(params[test_balanced_accuracy_score_all.index(max(test_balanced_accuracy_score_all))])
    # print('Number of combinations: ', len(solver)*len(multi_class))
    
    max_index = test_balanced_accuracy_score_all.index(max(test_balanced_accuracy_score_all))
    
    train_scores = []
    for i in training: 
        i = i[max_index]
        if type(i) == tuple: 
            prec = i[0]
            recall = i[1]
            f1 = i[2]
            train_scores.append(prec)
            train_scores.append(recall)
            train_scores.append(f1)
        else:  
            train_scores.append(i)
    
    test_scores = []
    for i in testing: 
        i = i[max_index]
        if type(i) == tuple: 
            prec = i[0]
            recall = i[1]
            f1 = i[2]
            test_scores.append(prec)
            test_scores.append(recall)
            test_scores.append(f1)
        else:  
            test_scores.append(i)
    
    score_names =  ['balanced_accuracy', 'matthews_corrcoef', 'fbeta_score_micro', 'fbeta_score_macro', 'fbeta_score_weighted', 'precision_micro', 'recall_micro', 'f1_micro', 'precision_macro', 'precision_macro', 'f1_macro', 'precision_weighted', 'recall_weighted', 'f1_weighted']
    
    
    assert len(train_scores) == len(score_names)
    assert len(test_scores) == len(score_names)
          
    testdot1_23 = pd.DataFrame({'classifier': clf.__class__.__name__
                             , 'training': train_scores
                            , 'test': test_scores
                            , 'score': score_names})
    return testdot1_23


def cross_validation_gradientboostingclassifier(X, y, cv=5): 
    
    params = []
    
    score_names_all = []
    test_scores_all = []
    train_scores_all = []
    
    
    learning_rate = [.1, .109, .11, .12]
    n_estimators = [180, 190]
    max_depth = [2, 4, 6, 8]
    init = [None, 'zero']
    
    for lr in learning_rate:  
        for n in n_estimators: 
                for d in max_depth: 
                    for ini in init:                         
                        clf = GradientBoostingClassifier(learning_rate=lr, n_estimators=n, max_depth=d, init=ini, random_state=24)
                        #scoring = ['balanced_accuracy', 'f1_micro', 'f1_macro', 'f1_weighted', 'precision_micro', 'precision_macro', 'precision_weighted','recall_micro', 'recall_macro', 'recall_weighted', 'roc_auc_ovr', 'roc_auc_ovo', 'roc_auc_ovr_weighted', 'roc_auc_ovo_weighted']
                        # train on training set  
                        scoring = ['balanced_accuracy', 'f1_micro', 'f1_macro', 'f1_weighted', 'precision_micro', 'precision_macro', 'precision_weighted','recall_micro', 'recall_macro', 'recall_weighted']
                        scores = cross_validate(clf, X, y, cv=5, scoring=scoring, return_train_score=True)
                        score_names = []
                        test_scores = [] 
                        train_scores = []        
                        for el in sorted(scores.keys()): 
                            if el.startswith('test') and not el.endswith('matrix'): 
                                test_scores.append(scores[el].mean())
                                score_names.append(el[5:])
                            elif el.startswith('train') and not el.endswith('matrix'): 
                                train_scores.append(scores[el].mean())
                        assert len(train_scores) == len(test_scores)
                        assert len(train_scores) == len(score_names)
                    
                        score_names_all.append(score_names)
                        test_scores_all.append(test_scores)
                        train_scores_all.append(train_scores)
                        params.append(f'learning_rate: {lr}, n_estimators: {n}, max_depth: {d}, init: {ini}')
                        
    # get max score's parameters 
    for el in test_scores_all: 
        if el[score_names.index('balanced_accuracy')] == max([i[score_names.index('balanced_accuracy')] for i in test_scores_all]): 
            windi = test_scores_all.index(el)
    
    print('Highest score: ', max([i[score_names.index('balanced_accuracy')] for i in test_scores_all]))
    print(params[windi])
    # print('Number of combinations: ', len(solver)*len(multi_class))
    
    nb =  max([i[score_names.index('balanced_accuracy')] for i in test_scores_all])
    for el in test_scores_all: 
        if el[score_names.index('balanced_accuracy')] == nb: 
            test_scores = el
    nb =  max([i[score_names.index('balanced_accuracy')] for i in train_scores_all])
    for el in train_scores_all: 
        if el[score_names.index('balanced_accuracy')] == nb: 
            train_scores = el
    score_names =  score_names_all[score_names.index('balanced_accuracy')]
    
    cv5_24 = pd.DataFrame({'classifier': clf.__class__.__name__
                             , 'training': train_scores
                            , 'test': test_scores
                            , 'score': score_names})
    return cv5_24


def train_test_split_gradientboostingclassifier(X_train, y_train): 
    
    params = []
        
    learning_rate = [.1, .109, .11, .12]
    n_estimators = [180, 190]
    max_depth = [2, 4, 6, 8]
    init = [None, 'zero']
    
    # train-test split
    train_balanced_accuracy_score_all = []
    train_confusion_matrix_all = []
    train_multilabel_confusion_matrix_all = []
    train_matthews_corrcoef_all = []
    train_fbeta_score_micro_all = []
    train_fbeta_score_macro_all = []
    train_fbeta_score_weighted_all = []
    train_precision_recall_fscore_support_micro_all =[]
    train_precision_recall_fscore_support_macro_all = []
    train_precision_recall_fscore_support_weighted_all = []
    test_balanced_accuracy_score_all= []
    test_confusion_matrix_all= []
    test_multilabel_confusion_matrix_all= []
    test_matthews_corrcoef_all= []
    test_fbeta_score_micro_all= []
    test_fbeta_score_macro_all= []
    test_fbeta_score_weighted_all= []
    test_precision_recall_fscore_support_micro_all= []
    test_precision_recall_fscore_support_macro_all= []
    test_precision_recall_fscore_support_weighted_all= []
    training = [train_balanced_accuracy_score_all, train_matthews_corrcoef_all, train_fbeta_score_micro_all, train_fbeta_score_macro_all, train_fbeta_score_weighted_all, train_precision_recall_fscore_support_micro_all, train_precision_recall_fscore_support_macro_all, train_precision_recall_fscore_support_weighted_all]
    testing = [test_balanced_accuracy_score_all, test_matthews_corrcoef_all, test_fbeta_score_micro_all, test_fbeta_score_macro_all, test_fbeta_score_weighted_all, test_precision_recall_fscore_support_micro_all, test_precision_recall_fscore_support_macro_all, test_precision_recall_fscore_support_weighted_all]
    
    
    
    for lr in learning_rate:  
        for n in n_estimators: 
                for d in max_depth: 
                    for ini in init:                         
                        clf = GradientBoostingClassifier(learning_rate=lr, n_estimators=n, max_depth=d, init=ini, random_state=24)
                        # train on training set  
                        clf.fit(X_train, y_train)
                        # training set 
                        y_pred = clf.predict(X_train)    
                        # multiclass evaluation scores 
                        train_balanced_accuracy_score_all.append(balanced_accuracy_score(y_train, y_pred))
                        train_confusion_matrix_all.append(confusion_matrix(y_train, y_pred))
                        train_multilabel_confusion_matrix_all.append(multilabel_confusion_matrix(y_train, y_pred))
                        # d = hinge_loss(y_train, y_pred, labels)
                        train_matthews_corrcoef_all.append(matthews_corrcoef(y_train, y_pred))
                        # f= roc_auc_score(y_train, y_pred, average='micro', multi_class='ovo')
                        train_fbeta_score_micro_all.append(fbeta_score(y_train, y_pred, average='micro', beta=1))
                        train_fbeta_score_macro_all.append(fbeta_score(y_train, y_pred, average='macro', beta=1))
                        train_fbeta_score_weighted_all.append(fbeta_score(y_train, y_pred, average='weighted', beta=1))
                        train_precision_recall_fscore_support_micro_all.append(precision_recall_fscore_support(y_train, y_pred, beta=1, average='micro'))
                        train_precision_recall_fscore_support_macro_all.append(precision_recall_fscore_support(y_train, y_pred, beta=1, average='macro'))
                        train_precision_recall_fscore_support_weighted_all.append(precision_recall_fscore_support(y_train, y_pred, beta=1, average='weighted'))
                        
                        # test set 
                        y_pred2 = clf.predict(X_test)  
                        # multiclass evaluation scores
                        test_balanced_accuracy_score_all.append(balanced_accuracy_score(y_test, y_pred2))
                        test_confusion_matrix_all.append(confusion_matrix(y_test, y_pred2))
                        test_multilabel_confusion_matrix_all.append(multilabel_confusion_matrix(y_test, y_pred2))
                        # d2 = hinge_loss(y_test, labels=y_pred)
                        test_matthews_corrcoef_all.append(matthews_corrcoef(y_test, y_pred2))
                        # f2= roc_auc_score(y_test, y_pred, average='micro', multi_class='ovo')
                        test_fbeta_score_micro_all.append(fbeta_score(y_test, y_pred2, average='micro', beta=1))
                        test_fbeta_score_macro_all.append(fbeta_score(y_test, y_pred2, average='macro', beta=1))
                        test_fbeta_score_weighted_all.append(fbeta_score(y_test, y_pred2, average='weighted', beta=1))
                        test_precision_recall_fscore_support_micro_all.append(precision_recall_fscore_support(y_test, y_pred2, beta=1, average='micro'))
                        test_precision_recall_fscore_support_macro_all.append(precision_recall_fscore_support(y_test, y_pred2, beta=1, average='macro'))
                        test_precision_recall_fscore_support_weighted_all.append(precision_recall_fscore_support(y_test, y_pred2, beta=1, average='weighted'))    
                      
                        params.append(f'learning_rate: {lr}, n_estimators: {n},  max_depth: {d}, init: {ini}')
                                
    
    # get max score's parameters 
    print('Highest score: ', max(test_balanced_accuracy_score_all))
    print(params[test_balanced_accuracy_score_all.index(max(test_balanced_accuracy_score_all))])
    # print('Number of combinations: ', len(solver)*len(multi_class))
    
    max_index = test_balanced_accuracy_score_all.index(max(test_balanced_accuracy_score_all))
    
    train_scores = []
    for i in training: 
        i = i[max_index]
        if type(i) == tuple: 
            prec = i[0]
            recall = i[1]
            f1 = i[2]
            train_scores.append(prec)
            train_scores.append(recall)
            train_scores.append(f1)
        else:  
            train_scores.append(i)
    
    test_scores = []
    for i in testing: 
        i = i[max_index]
        if type(i) == tuple: 
            prec = i[0]
            recall = i[1]
            f1 = i[2]
            test_scores.append(prec)
            test_scores.append(recall)
            test_scores.append(f1)
        else:  
            test_scores.append(i)
    
    score_names =  ['balanced_accuracy', 'matthews_corrcoef', 'fbeta_score_micro', 'fbeta_score_macro', 'fbeta_score_weighted', 'precision_micro', 'recall_micro', 'f1_micro', 'precision_macro', 'precision_macro', 'f1_macro', 'precision_weighted', 'recall_weighted', 'f1_weighted']
    
    
    assert len(train_scores) == len(score_names)
    assert len(test_scores) == len(score_names)
          
    testdot1_24 = pd.DataFrame({'classifier': clf.__class__.__name__
                             , 'training': train_scores
                            , 'test': test_scores
                            , 'score': score_names})
    return testdot1_24




#%%    
if __name__ == '__main__':
    
    # declare variables 
    TEST_SIZE=0.1
    
     # set directory 
    os.chdir(PATH)
    
    # load data 
    df = pd.read_excel(FILE, sep=" ", index_col=0)
    df.head()
 
    # analyze data   
    analyze_dataset(df, LABEL, LABEL2)   
    # preprocessing
    X, y = get_X_y(df, FEATURE, LABEL)
    # split data
    X_train, X_test, y_train, y_test = test_train_split(X, y, TEST_SIZE)
    
#%%
    # processing 
    cv5_1 = cross_validation_logisticregression(X, y, cv=5)
    testdot1_1 = train_test_split_logisticregression(X_train, y_train)
    cv5_2 = cross_validation_logisticregressioncv(X, y, cv=5)
    testdot1_2 = train_test_split_logisticregressioncv(X_train, y_train)
    testdot1_3 = train_test_split_ridgeclassifier(X_train, y_train)
    testdot1_4 = train_test_split_ridgeclassifiercv(X_train, y_train)
    cv5_5 = cross_validation_gaussiannb(X, y, cv=5)
    testdot1_5 = train_test_split_gaussiannb(X_train, y_train)
    cv5_6 = cross_validation_knn(X, y, cv=5)
    testdot1_6 = train_test_split_knn(X_train, y_train)
    cv5_7 = cross_validation_nearestcentroid(X, y, cv=5)
    testdot1_7 = train_test_split_nearestcentroid(X_train, y_train)
    cv5_8 = cross_validation_radiusneighborsclassifier(X, y, cv=5)
    testdot1_8 = train_test_split_radiusneighborsclassifier(X_train, y_train)
    cv5_9 = cross_validation_linearsvc(X, y, cv=5)
    testdot1_9 = train_test_split_linearsvc(X_train, y_train)
    cv5_10 = cross_validation_decisiontreeclassifier(X, y, cv=5)
    testdot1_10 = train_test_split_decisiontreeclassifier(X_train, y_train)
    cv5_11 = cross_validation_extratreeclassifier(X, y, cv=5)
    testdot1_11 = train_test_split_extratreeclassifier(X_train, y_train)
    cv5_12 = cross_validation_extratreesclassifier(X, y, cv=5)
    testdot1_12 = train_test_split_extratreesclassifier(X_train, y_train)
    cv5_13 = cross_validation_randomforestclassifier(X, y, cv=5)
    testdot1_13 = train_test_split_randomforestclassifier(X_train, y_train)
    cv5_14 = cross_validation_lineardiscriminantanalysis(X, y, cv=5)
    testdot1_14 = train_test_split_lineardiscriminantanalysis(X_train, y_train)
    cv5_15 = cross_validation_quadraticdiscriminantanalysis(X, y, cv=5)
    testdot1_15 = train_test_split_quadraticdiscriminantanalysis(X_train, y_train)
    cv5_16 = cross_validation_labelpropagation(X, y, cv=5)
    testdot1_16 = train_test_split_labelpropagation(X_train, y_train)
    cv5_17 = cross_validation_labelspreading(X, y, cv=5)
    testdot1_17 = train_test_split_labelspreading(X_train, y_train)
    cv5_18 = cross_validation_nusvc(X, y, cv=5)
    testdot1_18 = train_test_split_nusvc(X_train, y_train)
    cv5_19 = cross_validation_svc(X, y, cv=5)
    testdot1_19 = train_test_split_svc(X_train, y_train)
    cv5_20 = cross_validation_gaussianprocessclassifier(X, y, cv=5)
    testdot1_20 = train_test_split_gaussianprocessclassifier(X_train, y_train)
    cv5_21 = cross_validation_sgdclassifier(X, y, cv=5)
    testdot1_21 = train_test_split_sgdclassifier(X_train, y_train)
    cv5_22 = cross_validation_perceptron(X, y, cv=5)
    testdot1_22 = train_test_split_perceptron(X_train, y_train)
    cv5_23 = cross_validation_passiveaggressiveclassifier(X, y, cv=5)
    testdot1_23 = train_test_split_passiveaggressiveclassifier(X_train, y_train)
    cv5_24 = cross_validation_gradientboostingclassifier(X, y, cv=5)
    testdot1_24 = train_test_split_gradientboostingclassifier(X_train, y_train)
    
    # best model class 
    start = timer()
    cv5_15, clf = cross_validation_quadraticdiscriminantanalysis(X, y, cv=5)
    end = timer()
    duration = end - start
    print('duration: ', duration)
    
    start = timer()
    testdot1_5, clf = train_test_split_gaussiannb(X_train, y_train)
    end = timer()
    duration = end - start
    print('duration: ', duration)
    


#%%
    # save best model
    os.chdir(SAVE_PATH)
    testscore = testdot1_5['test'][testdot1_5['score'] == 'balanced_accuracy_score'].iloc[0]

    filename = f'model_{SOURCE}_{SYSTEM}_{METHOD}_{clf.__class__.__name__}_cat{df[LABEL].nunique()}_testacc{np.round(testscore,2)}.sav'
    pickle.dump(clf, open(filename, 'wb'))


