# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 20:34:09 2020

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
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings
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
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifierCV
# sklearn scorer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
# sklearn metric
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, roc_auc_score
from sklearn.metrics import fbeta_score, precision_recall_fscore_support, f1_score, precision_score, recall_score
from sklearn.metrics import log_loss, jaccard_score,zero_one_loss,hamming_loss, average_precision_score
from timeit import default_timer as timer

#%%

# declare variables 
PATH = r'D:\thesis\input_color_name_dictionaries\thesaurus\datasets'
FILE = 'eeffcnd_thesaurus_basicitten_upinterval.xlsx'
SOURCE = 'THESAURUS'
SYSTEM = 'ITTEN'
METHOD = 'INTERVAL' 

FEATURE = 'cielab'
LABEL =  'cat1'
LABEL2 =  'cat2'

SAVE_PATH = r'D:\thesis\machine_learning\models'

# ignore warning
warnings.filterwarnings('ignore')



#%%

# functions
 
def analyze_dataset(df, label, label2): 
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
    # y: convert multiclass-multioutput to multilabel 
    lab2pt2 = df[LABEL2].tolist() #list(df.index)
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



        
def test_train_split(X, y, test_size): 
    """ splits data set into test and training set"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    print(f"You have {len(X_train)} training colors and {len(X_test)} test colors - test_size: {test_size*100}.")
    return X_train, X_test, y_train, y_test  



def cross_validation_knn(X, y, cv=5):
    params = []
    
    n_neighbors = np.arange(1,21).tolist()
    leaf_sizes = np.arange(1,11).tolist()
    combinations = len(n_neighbors) *len(leaf_sizes)
    print(combinations)
    
    score_names_all = []
    test_scores_all = []
    train_scores_all = []
      
    for n in n_neighbors: 
        for l in leaf_sizes:        
            clf = KNeighborsClassifier(n_neighbors=n, leaf_size=l, weights='distance', n_jobs=-1)
            # long exec time 
            scoring = ['accuracy', 'average_precision',  'f1_micro', 'f1_macro', 'f1_weighted', 'f1_samples', 'precision_micro', 'precision_macro', 'precision_weighted', 'precision_samples', 'recall_micro', 'recall_macro', 'recall_weighted', 'recall_samples']
            #scoring = ['accuracy']
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
            
            print(n, l)
            score_names_all.append(score_names)
            test_scores_all.append(test_scores)
            train_scores_all.append(train_scores)      
            params.append(f'n_neighbors: {n}, leaf_size: {l}')
    
    # get max score's parameters 
    for el in test_scores_all: 
        if el[score_names.index(scoring[0])] == max([i[score_names.index(scoring[0])] for i in test_scores_all]): 
            windi = test_scores_all.index(el)
    
    print('Highest score: ', max([i[score_names.index(scoring[0])] for i in test_scores_all]))
    print(params[windi])
    #print('Number of combinations: ', len(solver)*len(multi_class))
    
    train_scores = train_scores_all[score_names.index(scoring[0])]
    test_scores = test_scores_all[score_names.index(scoring[0])]
    score_names =  score_names_all[score_names.index(scoring[0])]
    
    cv5_1 = pd.DataFrame({'classifier': clf.__class__.__name__
                             , 'training': train_scores
                            , 'test': test_scores
                            , 'score': score_names})
    return cv5_1


def train_test_split_knn(X_train, y_train): 
    params = []
    
    n_neighbors = np.arange(1,21).tolist()
    leaf_sizes = np.arange(1,11).tolist()
    combinations = len(n_neighbors) *len(leaf_sizes)
    print(combinations)
    
    # train-test split
    train_accuracy_score_all = []
    train_fbeta_score_micro_all = []
    train_fbeta_score_macro_all = []
    train_fbeta_score_weighted_all = []
    train_fbeta_score_samples_all = []
    train_hamming_loss_all = []
    train_jaccard_score_micro_all = []
    train_jaccard_score_macro_all = []
    train_jaccard_score_weighted_all = []
    train_jaccard_score_samples_all = []
    train_log_loss_all = []
    train_multilabel_confusion_matrix_all = []
    train_precision_recall_fscore_support_micro_all =[]
    train_precision_recall_fscore_support_macro_all = []
    train_precision_recall_fscore_support_weighted_all = []
    train_precision_recall_fscore_support_samples_all = []
    train_roc_auc_score_micro_all =[]
    train_roc_auc_score_macro_all = []
    train_roc_auc_score_weighted_all = []
    train_roc_auc_score_samples_all = []
    train_zero_one_loss_all = []
    train_average_precision_score_micro_all = []
    train_average_precision_score_macro_all = []
    train_average_precision_score_weighted_all = []
    train_average_precision_score_samples_all = []
    
    test_accuracy_score_all = []
    test_fbeta_score_micro_all = []
    test_fbeta_score_macro_all =  [] 
    test_fbeta_score_weighted_all = []
    test_fbeta_score_samples_all = []
    test_hamming_loss_all = []
    test_jaccard_score_micro_all = []
    test_jaccard_score_macro_all = []
    test_jaccard_score_weighted_all = []
    test_jaccard_score_samples_all = []
    test_log_loss_all = []
    test_multilabel_confusion_matrix_all = []
    test_precision_recall_fscore_support_micro_all =[]
    test_precision_recall_fscore_support_macro_all = []
    test_precision_recall_fscore_support_weighted_all = []
    test_precision_recall_fscore_support_samples_all = []
    test_roc_auc_score_micro_all = []
    test_roc_auc_score_macro_all = []
    test_roc_auc_score_weighted_all = []
    test_roc_auc_score_samples_all = []
    test_zero_one_loss_all = []  
    test_average_precision_score_micro_all = []
    test_average_precision_score_macro_all = []
    test_average_precision_score_weighted_all = []
    test_average_precision_score_samples_all = []
        
    training = [train_accuracy_score_all, train_fbeta_score_micro_all, train_fbeta_score_macro_all, train_fbeta_score_weighted_all, train_fbeta_score_samples_all,train_hamming_loss_all,train_jaccard_score_micro_all,train_jaccard_score_macro_all,train_jaccard_score_weighted_all,train_jaccard_score_samples_all,train_log_loss_all,train_multilabel_confusion_matrix_all,train_precision_recall_fscore_support_micro_all,train_precision_recall_fscore_support_macro_all,train_precision_recall_fscore_support_weighted_all,train_precision_recall_fscore_support_samples_all,train_roc_auc_score_micro_all,train_roc_auc_score_macro_all,train_roc_auc_score_weighted_all,train_roc_auc_score_samples_all,train_zero_one_loss_all,train_average_precision_score_micro_all,train_average_precision_score_macro_all,train_average_precision_score_weighted_all,train_average_precision_score_samples_all]
    testing = [test_accuracy_score_all, test_fbeta_score_micro_all, test_fbeta_score_macro_all, test_fbeta_score_weighted_all, test_fbeta_score_samples_all,test_hamming_loss_all,test_jaccard_score_micro_all,test_jaccard_score_macro_all,test_jaccard_score_weighted_all,test_jaccard_score_samples_all,test_log_loss_all,test_multilabel_confusion_matrix_all,test_precision_recall_fscore_support_micro_all,test_precision_recall_fscore_support_macro_all,test_precision_recall_fscore_support_weighted_all,test_precision_recall_fscore_support_samples_all,test_roc_auc_score_micro_all,test_roc_auc_score_macro_all,test_roc_auc_score_weighted_all,test_roc_auc_score_samples_all,test_zero_one_loss_all,test_average_precision_score_micro_all,test_average_precision_score_macro_all,test_average_precision_score_weighted_all,test_average_precision_score_samples_all]
     
    
    for n in n_neighbors: 
        for l in leaf_sizes: 
            clf = KNeighborsClassifier(n_neighbors=n, weights='distance', leaf_size=l, n_jobs=-1)
            clf.fit(X_train, y_train)
            # training set
            y_pred = clf.predict(X_train)     
            train_accuracy_score_all.append(accuracy_score(y_train, y_pred))
            train_fbeta_score_micro_all.append(fbeta_score(y_train, y_pred, average='micro', beta=1))
            train_fbeta_score_macro_all.append(fbeta_score(y_train, y_pred, average='macro', beta=1))
            train_fbeta_score_weighted_all.append(fbeta_score(y_train, y_pred, average='weighted', beta=1))
            train_fbeta_score_samples_all.append(fbeta_score(y_train, y_pred, average='samples', beta=1))
            train_hamming_loss_all.append(hamming_loss(y_train, y_pred))
            train_jaccard_score_micro_all.append(jaccard_score(y_train, y_pred, average='micro'))
            train_jaccard_score_macro_all.append(jaccard_score(y_train, y_pred, average='macro'))
            train_jaccard_score_weighted_all.append(jaccard_score(y_train, y_pred, average='weighted'))
            train_jaccard_score_samples_all.append(jaccard_score(y_train, y_pred, average='samples'))
            train_log_loss_all.append(log_loss(y_train, y_pred))
            train_multilabel_confusion_matrix_all.append(multilabel_confusion_matrix(y_train, y_pred))
            train_precision_recall_fscore_support_micro_all.append(precision_recall_fscore_support(y_train, y_pred, beta=1, average='micro'))
            train_precision_recall_fscore_support_macro_all.append(precision_recall_fscore_support(y_train, y_pred, beta=1, average='macro'))
            train_precision_recall_fscore_support_weighted_all.append(precision_recall_fscore_support(y_train, y_pred, beta=1, average='weighted'))
            train_precision_recall_fscore_support_samples_all.append(precision_recall_fscore_support(y_train, y_pred, beta=1, average='samples'))
            train_roc_auc_score_micro_all.append(roc_auc_score(y_train, y_pred, average='micro'))
            train_roc_auc_score_macro_all.append(roc_auc_score(y_train, y_pred, average='macro'))
            train_roc_auc_score_weighted_all.append(roc_auc_score(y_train, y_pred, average='weighted'))
            train_roc_auc_score_samples_all.append(roc_auc_score(y_train, y_pred, average='samples'))
            train_zero_one_loss_all.append(zero_one_loss(y_train, y_pred, normalize=False))    
               
            # test set 
            y_pred2 = clf.predict(X_test)  
            test_accuracy_score_all.append(accuracy_score(y_test, y_pred2))
            test_fbeta_score_micro_all.append(fbeta_score(y_test, y_pred2, average='micro', beta=1))
            test_fbeta_score_macro_all.append(fbeta_score(y_test, y_pred2, average='macro', beta=1))
            test_fbeta_score_weighted_all.append(fbeta_score(y_test, y_pred2, average='weighted', beta=1))
            test_fbeta_score_samples_all.append(fbeta_score(y_test, y_pred2, average='samples', beta=1))
            test_hamming_loss_all.append(hamming_loss(y_test, y_pred2))
            test_jaccard_score_micro_all.append(jaccard_score(y_test, y_pred2, average='micro'))
            test_jaccard_score_macro_all.append(jaccard_score(y_test, y_pred2, average='macro'))
            test_jaccard_score_weighted_all.append(jaccard_score(y_test, y_pred2, average='weighted'))
            test_jaccard_score_samples_all.append(jaccard_score(y_test, y_pred2, average='samples'))
            test_log_loss_all.append(log_loss(y_test, y_pred2))
            test_multilabel_confusion_matrix_all.append(multilabel_confusion_matrix(y_test, y_pred2))
            test_precision_recall_fscore_support_micro_all.append(precision_recall_fscore_support(y_test, y_pred2, beta=1, average='micro'))
            test_precision_recall_fscore_support_macro_all.append(precision_recall_fscore_support(y_test, y_pred2, beta=1, average='macro'))
            test_precision_recall_fscore_support_weighted_all.append(precision_recall_fscore_support(y_test, y_pred2, beta=1, average='weighted'))
            test_precision_recall_fscore_support_samples_all.append(precision_recall_fscore_support(y_test, y_pred2, beta=1, average='samples'))
            test_roc_auc_score_micro_all.append(roc_auc_score(y_test, y_pred2, average='micro'))
            test_roc_auc_score_macro_all.append(roc_auc_score(y_test, y_pred2, average='macro'))
            test_roc_auc_score_weighted_all.append(roc_auc_score(y_test, y_pred2, average='weighted'))
            test_roc_auc_score_samples_all.append(roc_auc_score(y_test, y_pred2, average='samples'))
            test_zero_one_loss_all.append(zero_one_loss(y_test, y_pred2, normalize=False))    
            print(n,l)
            params.append(f'n_neighbors: {n}, leaf_sizes: {l}')
        
    
    # get max score's parameters 
    print('Highest score: ', max(test_accuracy_score_all))
    print(params[test_accuracy_score_all.index(max(test_accuracy_score_all))])
    print('Number of combinations: ', len(n_neighbors)*len(leaf_sizes))
    
    max_index = test_accuracy_score_all.index(max(test_accuracy_score_all))
    
    train_scores = []
    for train in training[:11]: 
        i = train[max_index]
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
    for i in testing[:11]: 
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
    
    score_names =  ['accuracy', 'fbeta_micro', 'fbeta_macro', 'fbeta_weighted', 'fbeta_samples', 'hamming_loss', 'jaccard_micro', 'jaccard_macro', 'jaccard_weighted', 'jaccard_samples', 'log_loss', 'multilabel_confusion_matrix', 'precision_recall_fsupport_micro', 'precision_recall_fsupport_macro', 'precision_recall_fsupport_weighted', 'precision_recall_fsupport_samples', 'roc_auc_micro', 'roc_auc_macro','roc_auc_weighted', 'roc_auc_samples', 'zero_one_loss', 'average_precision_micro', 'average_precision_macro', 'average_precision_weighted', 'average_precision_samples']
    score_names =  score_names[:11]
    
    assert len(train_scores) == len(score_names)
    assert len(test_scores) == len(score_names)
          
    testdot1_1 = pd.DataFrame({'classifier': clf.__class__.__name__
                             , 'training': train_scores
                            , 'test': test_scores
                            , 'score': score_names})
    return testdot1_1


def train_test_split_radiusneighborsclassifier(X_train, y_train): 
    params = []
    
    # varying parameters
    radius = [300, 200, 100]
    leaf_sizes = np.arange(1,31).tolist()
    combinations = len(radius) *len(leaf_sizes)
    print(combinations)
    
    # train-test split
    train_accuracy_score_all = []
    train_fbeta_score_micro_all = []
    train_fbeta_score_macro_all = []
    train_fbeta_score_weighted_all = []
    train_fbeta_score_samples_all = []
    train_hamming_loss_all = []
    train_jaccard_score_micro_all = []
    train_jaccard_score_macro_all = []
    train_jaccard_score_weighted_all = []
    train_jaccard_score_samples_all = []
    train_log_loss_all = []
    train_multilabel_confusion_matrix_all = []
    train_precision_recall_fscore_support_micro_all =[]
    train_precision_recall_fscore_support_macro_all = []
    train_precision_recall_fscore_support_weighted_all = []
    train_precision_recall_fscore_support_samples_all = []
    train_roc_auc_score_micro_all =[]
    train_roc_auc_score_macro_all = []
    train_roc_auc_score_weighted_all = []
    train_roc_auc_score_samples_all = []
    train_zero_one_loss_all = []
    train_average_precision_score_micro_all = []
    train_average_precision_score_macro_all = []
    train_average_precision_score_weighted_all = []
    train_average_precision_score_samples_all = []
    
    test_accuracy_score_all = []
    test_fbeta_score_micro_all = []
    test_fbeta_score_macro_all =  [] 
    test_fbeta_score_weighted_all = []
    test_fbeta_score_samples_all = []
    test_hamming_loss_all = []
    test_jaccard_score_micro_all = []
    test_jaccard_score_macro_all = []
    test_jaccard_score_weighted_all = []
    test_jaccard_score_samples_all = []
    test_log_loss_all = []
    test_multilabel_confusion_matrix_all = []
    test_precision_recall_fscore_support_micro_all =[]
    test_precision_recall_fscore_support_macro_all = []
    test_precision_recall_fscore_support_weighted_all = []
    test_precision_recall_fscore_support_samples_all = []
    test_roc_auc_score_micro_all = []
    test_roc_auc_score_macro_all = []
    test_roc_auc_score_weighted_all = []
    test_roc_auc_score_samples_all = []
    test_zero_one_loss_all = []  
    test_average_precision_score_micro_all = []
    test_average_precision_score_macro_all = []
    test_average_precision_score_weighted_all = []
    test_average_precision_score_samples_all = []
        
    training = [train_accuracy_score_all, train_fbeta_score_micro_all, train_fbeta_score_macro_all, train_fbeta_score_weighted_all, train_fbeta_score_samples_all,train_hamming_loss_all,train_jaccard_score_micro_all,train_jaccard_score_macro_all,train_jaccard_score_weighted_all,train_jaccard_score_samples_all,train_log_loss_all,train_multilabel_confusion_matrix_all,train_precision_recall_fscore_support_micro_all,train_precision_recall_fscore_support_macro_all,train_precision_recall_fscore_support_weighted_all,train_precision_recall_fscore_support_samples_all,train_roc_auc_score_micro_all,train_roc_auc_score_macro_all,train_roc_auc_score_weighted_all,train_roc_auc_score_samples_all,train_zero_one_loss_all,train_average_precision_score_micro_all,train_average_precision_score_macro_all,train_average_precision_score_weighted_all,train_average_precision_score_samples_all]
    testing = [test_accuracy_score_all, test_fbeta_score_micro_all, test_fbeta_score_macro_all, test_fbeta_score_weighted_all, test_fbeta_score_samples_all,test_hamming_loss_all,test_jaccard_score_micro_all,test_jaccard_score_macro_all,test_jaccard_score_weighted_all,test_jaccard_score_samples_all,test_log_loss_all,test_multilabel_confusion_matrix_all,test_precision_recall_fscore_support_micro_all,test_precision_recall_fscore_support_macro_all,test_precision_recall_fscore_support_weighted_all,test_precision_recall_fscore_support_samples_all,test_roc_auc_score_micro_all,test_roc_auc_score_macro_all,test_roc_auc_score_weighted_all,test_roc_auc_score_samples_all,test_zero_one_loss_all,test_average_precision_score_micro_all,test_average_precision_score_macro_all,test_average_precision_score_weighted_all,test_average_precision_score_samples_all]
        
    for r in radius: 
        for l in leaf_sizes: 
            clf = RadiusNeighborsClassifier(radius=r, leaf_size=l, weights='distance', n_jobs=-1)  
            clf.fit(X_train, y_train)
            # training set 
            y_pred = clf.predict(X_train)    
            train_accuracy_score_all.append(accuracy_score(y_train, y_pred))
            train_fbeta_score_micro_all.append(fbeta_score(y_train, y_pred, average='micro', beta=1))
            train_fbeta_score_macro_all.append(fbeta_score(y_train, y_pred, average='macro', beta=1))
            train_fbeta_score_weighted_all.append(fbeta_score(y_train, y_pred, average='weighted', beta=1))
            train_fbeta_score_samples_all.append(fbeta_score(y_train, y_pred, average='samples', beta=1))
            train_hamming_loss_all.append(hamming_loss(y_train, y_pred))
            train_jaccard_score_micro_all.append(jaccard_score(y_train, y_pred, average='micro'))
            train_jaccard_score_macro_all.append(jaccard_score(y_train, y_pred, average='macro'))
            train_jaccard_score_weighted_all.append(jaccard_score(y_train, y_pred, average='weighted'))
            train_jaccard_score_samples_all.append(jaccard_score(y_train, y_pred, average='samples'))
            train_log_loss_all.append(log_loss(y_train, y_pred))
            train_multilabel_confusion_matrix_all.append(multilabel_confusion_matrix(y_train, y_pred))
            train_precision_recall_fscore_support_micro_all.append(precision_recall_fscore_support(y_train, y_pred, beta=1, average='micro'))
            train_precision_recall_fscore_support_macro_all.append(precision_recall_fscore_support(y_train, y_pred, beta=1, average='macro'))
            train_precision_recall_fscore_support_weighted_all.append(precision_recall_fscore_support(y_train, y_pred, beta=1, average='weighted'))
            train_precision_recall_fscore_support_samples_all.append(precision_recall_fscore_support(y_train, y_pred, beta=1, average='samples'))
            train_roc_auc_score_micro_all.append(roc_auc_score(y_train, y_pred, average='micro'))
            train_roc_auc_score_macro_all.append(roc_auc_score(y_train, y_pred, average='macro'))
            train_roc_auc_score_weighted_all.append(roc_auc_score(y_train, y_pred, average='weighted'))
            train_roc_auc_score_samples_all.append(roc_auc_score(y_train, y_pred, average='samples'))
            train_zero_one_loss_all.append(zero_one_loss(y_train, y_pred, normalize=False))    
    
            # test set 
            y_pred2 = clf.predict(X_test)  
            test_accuracy_score_all.append(accuracy_score(y_test, y_pred2))
            test_fbeta_score_micro_all.append(fbeta_score(y_test, y_pred2, average='micro', beta=1))
            test_fbeta_score_macro_all.append(fbeta_score(y_test, y_pred2, average='macro', beta=1))
            test_fbeta_score_weighted_all.append(fbeta_score(y_test, y_pred2, average='weighted', beta=1))
            test_fbeta_score_samples_all.append(fbeta_score(y_test, y_pred2, average='samples', beta=1))
            test_hamming_loss_all.append(hamming_loss(y_test, y_pred2))
            test_jaccard_score_micro_all.append(jaccard_score(y_test, y_pred2, average='micro'))
            test_jaccard_score_macro_all.append(jaccard_score(y_test, y_pred2, average='macro'))
            test_jaccard_score_weighted_all.append(jaccard_score(y_test, y_pred2, average='weighted'))
            test_jaccard_score_samples_all.append(jaccard_score(y_test, y_pred2, average='samples'))
            test_log_loss_all.append(log_loss(y_test, y_pred2))
            test_multilabel_confusion_matrix_all.append(multilabel_confusion_matrix(y_test, y_pred2))
            test_precision_recall_fscore_support_micro_all.append(precision_recall_fscore_support(y_test, y_pred2, beta=1, average='micro'))
            test_precision_recall_fscore_support_macro_all.append(precision_recall_fscore_support(y_test, y_pred2, beta=1, average='macro'))
            test_precision_recall_fscore_support_weighted_all.append(precision_recall_fscore_support(y_test, y_pred2, beta=1, average='weighted'))
            test_precision_recall_fscore_support_samples_all.append(precision_recall_fscore_support(y_test, y_pred2, beta=1, average='samples'))
            test_roc_auc_score_micro_all.append(roc_auc_score(y_test, y_pred2, average='micro'))
            test_roc_auc_score_macro_all.append(roc_auc_score(y_test, y_pred2, average='macro'))
            test_roc_auc_score_weighted_all.append(roc_auc_score(y_test, y_pred2, average='weighted'))
            test_roc_auc_score_samples_all.append(roc_auc_score(y_test, y_pred2, average='samples'))
            test_zero_one_loss_all.append(zero_one_loss(y_test, y_pred2, normalize=False))    
            print(r,l)
            params.append(f'radius: {r}, leaf_size: {l}')
    
    # get max score's parameters 
    print('Highest score: ', max(test_accuracy_score_all))
    print(params[test_accuracy_score_all.index(max(test_accuracy_score_all))])
    max_index = test_accuracy_score_all.index(max(test_accuracy_score_all))
    
    train_scores = []
    for i in training[:11]: 
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
    for i in testing[:11]: 
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
    
    score_names =  ['accuracy', 'fbeta_micro', 'fbeta_macro', 'fbeta_weighted', 'fbeta_samples', 'hamming_loss', 'jaccard_micro', 'jaccard_macro', 'jaccard_weighted', 'jaccard_samples', 'log_loss', 'multilabel_confusion_matrix', 'precision_recall_fsupport_micro', 'precision_recall_fsupport_macro', 'precision_recall_fsupport_weighted', 'precision_recall_fsupport_samples', 'roc_auc_micro', 'roc_auc_macro','roc_auc_weighted', 'roc_auc_samples', 'zero_one_loss', 'average_precision_micro', 'average_precision_macro', 'average_precision_weighted', 'average_precision_samples']
    score_names = score_names[:11]
    
    assert len(train_scores) == len(score_names)
    assert len(test_scores) == len(score_names)
          
    testdot1_2 = pd.DataFrame({'classifier': clf.__class__.__name__
                             , 'training': train_scores
                            , 'test': test_scores
                            , 'score': score_names})
    
    return testdot1_2



def cross_validation_decisiontreeclassifier(X, y, cv=5): 
    params = []
    
    # varying parameters
    max_depth = [1, 5, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    combinations = len(max_depth)
    print(combinations)
    
    score_names_all = []
    test_scores_all = []
    train_scores_all = []
    
    for d in max_depth: 
        clf = DecisionTreeClassifier(max_depth=d, random_state=24, class_weight='balanced')
        # train on training set  
        scoring = ['accuracy', 'average_precision',  'f1_micro', 'f1_macro', 'f1_weighted', 'f1_samples', 'precision_micro', 'precision_macro', 'precision_weighted', 'precision_samples', 'recall_micro', 'recall_macro', 'recall_weighted', 'recall_samples'] 
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
        print(d)
        params.append(f'max_depth: {d}')
    
    # get max score's parameters 
    for el in test_scores_all: 
        if el[score_names.index(scoring[0])] == max([i[score_names.index(scoring[0])] for i in test_scores_all]): 
            windi = test_scores_all.index(el)
    
    print('Highest score: ', max([i[score_names.index(scoring[0])] for i in test_scores_all]))
    print(params[windi])

    train_scores = train_scores_all[windi]
    test_scores = test_scores_all[windi]
    score_names =  score_names_all[windi]
    
    cv5_3 = pd.DataFrame({'classifier': clf.__class__.__name__
                             , 'training': train_scores
                            , 'test': test_scores
                            , 'score': score_names})
    return cv5_3


def  train_test_split_decisiontreeclassifier(X_train, y_train): 
    params = []
    
    # varying parameters
    max_depth = [1, 5, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    combinations = len(max_depth)
    print(combinations)
    
    train_accuracy_score_all = []
    train_fbeta_score_micro_all = []
    train_fbeta_score_macro_all = []
    train_fbeta_score_weighted_all = []
    train_fbeta_score_samples_all = []
    train_hamming_loss_all = []
    train_jaccard_score_micro_all = []
    train_jaccard_score_macro_all = []
    train_jaccard_score_weighted_all = []
    train_jaccard_score_samples_all = []
    train_log_loss_all = []
    train_multilabel_confusion_matrix_all = []
    train_precision_recall_fscore_support_micro_all =[]
    train_precision_recall_fscore_support_macro_all = []
    train_precision_recall_fscore_support_weighted_all = []
    train_precision_recall_fscore_support_samples_all = []
    train_roc_auc_score_micro_all =[]
    train_roc_auc_score_macro_all = []
    train_roc_auc_score_weighted_all = []
    train_roc_auc_score_samples_all = []
    train_zero_one_loss_all = []
    train_average_precision_score_micro_all = []
    train_average_precision_score_macro_all = []
    train_average_precision_score_weighted_all = []
    train_average_precision_score_samples_all = []
    
    test_accuracy_score_all = []
    test_fbeta_score_micro_all = []
    test_fbeta_score_macro_all =  [] 
    test_fbeta_score_weighted_all = []
    test_fbeta_score_samples_all = []
    test_hamming_loss_all = []
    test_jaccard_score_micro_all = []
    test_jaccard_score_macro_all = []
    test_jaccard_score_weighted_all = []
    test_jaccard_score_samples_all = []
    test_log_loss_all = []
    test_multilabel_confusion_matrix_all = []
    test_precision_recall_fscore_support_micro_all =[]
    test_precision_recall_fscore_support_macro_all = []
    test_precision_recall_fscore_support_weighted_all = []
    test_precision_recall_fscore_support_samples_all = []
    test_roc_auc_score_micro_all = []
    test_roc_auc_score_macro_all = []
    test_roc_auc_score_weighted_all = []
    test_roc_auc_score_samples_all = []
    test_zero_one_loss_all = []  
    test_average_precision_score_micro_all = []
    test_average_precision_score_macro_all = []
    test_average_precision_score_weighted_all = []
    test_average_precision_score_samples_all = []
        
    training = [train_accuracy_score_all, train_fbeta_score_micro_all, train_fbeta_score_macro_all, train_fbeta_score_weighted_all, train_fbeta_score_samples_all,train_hamming_loss_all,train_jaccard_score_micro_all,train_jaccard_score_macro_all,train_jaccard_score_weighted_all,train_jaccard_score_samples_all,train_log_loss_all,train_multilabel_confusion_matrix_all,train_precision_recall_fscore_support_micro_all,train_precision_recall_fscore_support_macro_all,train_precision_recall_fscore_support_weighted_all,train_precision_recall_fscore_support_samples_all,train_roc_auc_score_micro_all,train_roc_auc_score_macro_all,train_roc_auc_score_weighted_all,train_roc_auc_score_samples_all,train_zero_one_loss_all,train_average_precision_score_micro_all,train_average_precision_score_macro_all,train_average_precision_score_weighted_all,train_average_precision_score_samples_all]
    testing = [test_accuracy_score_all, test_fbeta_score_micro_all, test_fbeta_score_macro_all, test_fbeta_score_weighted_all, test_fbeta_score_samples_all,test_hamming_loss_all,test_jaccard_score_micro_all,test_jaccard_score_macro_all,test_jaccard_score_weighted_all,test_jaccard_score_samples_all,test_log_loss_all,test_multilabel_confusion_matrix_all,test_precision_recall_fscore_support_micro_all,test_precision_recall_fscore_support_macro_all,test_precision_recall_fscore_support_weighted_all,test_precision_recall_fscore_support_samples_all,test_roc_auc_score_micro_all,test_roc_auc_score_macro_all,test_roc_auc_score_weighted_all,test_roc_auc_score_samples_all,test_zero_one_loss_all,test_average_precision_score_micro_all,test_average_precision_score_macro_all,test_average_precision_score_weighted_all,test_average_precision_score_samples_all]
         
    for d in max_depth: 
        clf = DecisionTreeClassifier(max_depth=d, random_state=24, class_weight='balanced') 
        clf.fit(X_train, y_train)
        # training set 
        y_pred = clf.predict(X_train)    
        train_accuracy_score_all.append(accuracy_score(y_train, y_pred))
        train_fbeta_score_micro_all.append(fbeta_score(y_train, y_pred, average='micro', beta=1))
        train_fbeta_score_macro_all.append(fbeta_score(y_train, y_pred, average='macro', beta=1))
        train_fbeta_score_weighted_all.append(fbeta_score(y_train, y_pred, average='weighted', beta=1))
        train_fbeta_score_samples_all.append(fbeta_score(y_train, y_pred, average='samples', beta=1))
        train_hamming_loss_all.append(hamming_loss(y_train, y_pred))
        train_jaccard_score_micro_all.append(jaccard_score(y_train, y_pred, average='micro'))
        train_jaccard_score_macro_all.append(jaccard_score(y_train, y_pred, average='macro'))
        train_jaccard_score_weighted_all.append(jaccard_score(y_train, y_pred, average='weighted'))
        train_jaccard_score_samples_all.append(jaccard_score(y_train, y_pred, average='samples'))
        train_log_loss_all.append(log_loss(y_train, y_pred))
        train_multilabel_confusion_matrix_all.append(multilabel_confusion_matrix(y_train, y_pred))
        train_precision_recall_fscore_support_micro_all.append(precision_recall_fscore_support(y_train, y_pred, beta=1, average='micro'))
        train_precision_recall_fscore_support_macro_all.append(precision_recall_fscore_support(y_train, y_pred, beta=1, average='macro'))
        train_precision_recall_fscore_support_weighted_all.append(precision_recall_fscore_support(y_train, y_pred, beta=1, average='weighted'))
        train_precision_recall_fscore_support_samples_all.append(precision_recall_fscore_support(y_train, y_pred, beta=1, average='samples'))
        train_roc_auc_score_micro_all.append(roc_auc_score(y_train, y_pred, average='micro'))
        train_roc_auc_score_macro_all.append(roc_auc_score(y_train, y_pred, average='macro'))
        train_roc_auc_score_weighted_all.append(roc_auc_score(y_train, y_pred, average='weighted'))
        train_roc_auc_score_samples_all.append(roc_auc_score(y_train, y_pred, average='samples'))
        train_zero_one_loss_all.append(zero_one_loss(y_train, y_pred, normalize=False))     
        
        # test set 
        y_pred2 = clf.predict(X_test)  
        test_accuracy_score_all.append(accuracy_score(y_test, y_pred2))
        test_fbeta_score_micro_all.append(fbeta_score(y_test, y_pred2, average='micro', beta=1))
        test_fbeta_score_macro_all.append(fbeta_score(y_test, y_pred2, average='macro', beta=1))
        test_fbeta_score_weighted_all.append(fbeta_score(y_test, y_pred2, average='weighted', beta=1))
        test_fbeta_score_samples_all.append(fbeta_score(y_test, y_pred2, average='samples', beta=1))
        test_hamming_loss_all.append(hamming_loss(y_test, y_pred2))
        test_jaccard_score_micro_all.append(jaccard_score(y_test, y_pred2, average='micro'))
        test_jaccard_score_macro_all.append(jaccard_score(y_test, y_pred2, average='macro'))
        test_jaccard_score_weighted_all.append(jaccard_score(y_test, y_pred2, average='weighted'))
        test_jaccard_score_samples_all.append(jaccard_score(y_test, y_pred2, average='samples'))
        test_log_loss_all.append(log_loss(y_test, y_pred2))
        test_multilabel_confusion_matrix_all.append(multilabel_confusion_matrix(y_test, y_pred2))
        test_precision_recall_fscore_support_micro_all.append(precision_recall_fscore_support(y_test, y_pred2, beta=1, average='micro'))
        test_precision_recall_fscore_support_macro_all.append(precision_recall_fscore_support(y_test, y_pred2, beta=1, average='macro'))
        test_precision_recall_fscore_support_weighted_all.append(precision_recall_fscore_support(y_test, y_pred2, beta=1, average='weighted'))
        test_precision_recall_fscore_support_samples_all.append(precision_recall_fscore_support(y_test, y_pred2, beta=1, average='samples'))
        test_roc_auc_score_micro_all.append(roc_auc_score(y_test, y_pred2, average='micro'))
        test_roc_auc_score_macro_all.append(roc_auc_score(y_test, y_pred2, average='macro'))
        test_roc_auc_score_weighted_all.append(roc_auc_score(y_test, y_pred2, average='weighted'))
        test_roc_auc_score_samples_all.append(roc_auc_score(y_test, y_pred2, average='samples'))
        test_zero_one_loss_all.append(zero_one_loss(y_test, y_pred2, normalize=False))    
        print(d)
        params.append(f'max_depth: {d}')
    
    # get max score's parameters 
    print('Highest score: ', max(test_accuracy_score_all))
    print(params[test_accuracy_score_all.index(max(test_accuracy_score_all))])
    max_index = test_accuracy_score_all.index(max(test_accuracy_score_all))
    
    train_scores = []
    for i in training[:11]: 
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
    for i in testing[:11]: 
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
    
    score_names =  ['accuracy', 'fbeta_micro', 'fbeta_macro', 'fbeta_weighted', 'fbeta_samples', 'hamming_loss', 'jaccard_micro', 'jaccard_macro', 'jaccard_weighted', 'jaccard_samples', 'log_loss', 'multilabel_confusion_matrix', 'precision_recall_fsupport_micro', 'precision_recall_fsupport_macro', 'precision_recall_fsupport_weighted', 'precision_recall_fsupport_samples', 'roc_auc_micro', 'roc_auc_macro','roc_auc_weighted', 'roc_auc_samples', 'zero_one_loss', 'average_precision_micro', 'average_precision_macro', 'average_precision_weighted', 'average_precision_samples']
    score_names = score_names[:11]
    
    assert len(train_scores) == len(score_names)
    assert len(test_scores) == len(score_names)
          
    testdot1_3 = pd.DataFrame({'classifier': clf.__class__.__name__
                             , 'training': train_scores
                            , 'test': test_scores
                            , 'score': score_names})
    return testdot1_3



def cross_validation_extratreeclassifier(X, y, cv=5): 
    params = []
    
    # varying parameters
    max_depth = [10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 25, 50, 75, 100]
    combinations = len(max_depth)
    print(combinations)
     
    score_names_all = []
    test_scores_all = []
    train_scores_all = []
    
    for d in max_depth: 
        clf = ExtraTreeClassifier(max_depth=d, max_features=None, random_state=24, class_weight='balanced')
        # train on training set  
        scoring = ['accuracy', 'average_precision',  'f1_micro', 'f1_macro', 'f1_weighted', 'f1_samples', 'precision_micro', 'precision_macro', 'precision_weighted', 'precision_samples', 'recall_micro', 'recall_macro', 'recall_weighted', 'recall_samples'] 
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
        print(d)
        params.append(f'max_depth: {d}')
    
    # get max score's parameters 
    for el in test_scores_all: 
        if el[score_names.index(scoring[0])] == max([i[score_names.index(scoring[0])] for i in test_scores_all]): 
            windi = test_scores_all.index(el)
    
    print('Highest score: ', max([i[score_names.index(scoring[0])] for i in test_scores_all]))
    print(params[windi])

    train_scores = train_scores_all[windi]
    test_scores = test_scores_all[windi]
    score_names =  score_names_all[windi]
    
    cv5_4 = pd.DataFrame({'classifier': clf.__class__.__name__
                             , 'training': train_scores
                            , 'test': test_scores
                            , 'score': score_names})
    return cv5_4


def train_test_split_extratreeclassifier(X_train, y_train): 
    params = []
    
    # varying parameters
    max_depth = [10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 25, 50, 75, 100]
    combinations = len(max_depth)
    print(combinations)
    
    # train-test split
    train_accuracy_score_all = []
    train_fbeta_score_micro_all = []
    train_fbeta_score_macro_all = []
    train_fbeta_score_weighted_all = []
    train_fbeta_score_samples_all = []
    train_hamming_loss_all = []
    train_jaccard_score_micro_all = []
    train_jaccard_score_macro_all = []
    train_jaccard_score_weighted_all = []
    train_jaccard_score_samples_all = []
    train_log_loss_all = []
    train_multilabel_confusion_matrix_all = []
    train_precision_recall_fscore_support_micro_all =[]
    train_precision_recall_fscore_support_macro_all = []
    train_precision_recall_fscore_support_weighted_all = []
    train_precision_recall_fscore_support_samples_all = []
    train_roc_auc_score_micro_all =[]
    train_roc_auc_score_macro_all = []
    train_roc_auc_score_weighted_all = []
    train_roc_auc_score_samples_all = []
    train_zero_one_loss_all = []
    train_average_precision_score_micro_all = []
    train_average_precision_score_macro_all = []
    train_average_precision_score_weighted_all = []
    train_average_precision_score_samples_all = []
    
    test_accuracy_score_all = []
    test_fbeta_score_micro_all = []
    test_fbeta_score_macro_all =  [] 
    test_fbeta_score_weighted_all = []
    test_fbeta_score_samples_all = []
    test_hamming_loss_all = []
    test_jaccard_score_micro_all = []
    test_jaccard_score_macro_all = []
    test_jaccard_score_weighted_all = []
    test_jaccard_score_samples_all = []
    test_log_loss_all = []
    test_multilabel_confusion_matrix_all = []
    test_precision_recall_fscore_support_micro_all =[]
    test_precision_recall_fscore_support_macro_all = []
    test_precision_recall_fscore_support_weighted_all = []
    test_precision_recall_fscore_support_samples_all = []
    test_roc_auc_score_micro_all = []
    test_roc_auc_score_macro_all = []
    test_roc_auc_score_weighted_all = []
    test_roc_auc_score_samples_all = []
    test_zero_one_loss_all = []  
    test_average_precision_score_micro_all = []
    test_average_precision_score_macro_all = []
    test_average_precision_score_weighted_all = []
    test_average_precision_score_samples_all = []
        
    training = [train_accuracy_score_all, train_fbeta_score_micro_all, train_fbeta_score_macro_all, train_fbeta_score_weighted_all, train_fbeta_score_samples_all,train_hamming_loss_all,train_jaccard_score_micro_all,train_jaccard_score_macro_all,train_jaccard_score_weighted_all,train_jaccard_score_samples_all,train_log_loss_all,train_multilabel_confusion_matrix_all,train_precision_recall_fscore_support_micro_all,train_precision_recall_fscore_support_macro_all,train_precision_recall_fscore_support_weighted_all,train_precision_recall_fscore_support_samples_all,train_roc_auc_score_micro_all,train_roc_auc_score_macro_all,train_roc_auc_score_weighted_all,train_roc_auc_score_samples_all,train_zero_one_loss_all,train_average_precision_score_micro_all,train_average_precision_score_macro_all,train_average_precision_score_weighted_all,train_average_precision_score_samples_all]
    testing = [test_accuracy_score_all, test_fbeta_score_micro_all, test_fbeta_score_macro_all, test_fbeta_score_weighted_all, test_fbeta_score_samples_all,test_hamming_loss_all,test_jaccard_score_micro_all,test_jaccard_score_macro_all,test_jaccard_score_weighted_all,test_jaccard_score_samples_all,test_log_loss_all,test_multilabel_confusion_matrix_all,test_precision_recall_fscore_support_micro_all,test_precision_recall_fscore_support_macro_all,test_precision_recall_fscore_support_weighted_all,test_precision_recall_fscore_support_samples_all,test_roc_auc_score_micro_all,test_roc_auc_score_macro_all,test_roc_auc_score_weighted_all,test_roc_auc_score_samples_all,test_zero_one_loss_all,test_average_precision_score_micro_all,test_average_precision_score_macro_all,test_average_precision_score_weighted_all,test_average_precision_score_samples_all]
    
    for d in max_depth: 
        clf = ExtraTreeClassifier(max_depth=d, max_features=None, random_state=24, class_weight='balanced')
        clf.fit(X_train, y_train)
       # training set 
        y_pred = clf.predict(X_train)    
        train_accuracy_score_all.append(accuracy_score(y_train, y_pred))
        train_fbeta_score_micro_all.append(fbeta_score(y_train, y_pred, average='micro', beta=1))
        train_fbeta_score_macro_all.append(fbeta_score(y_train, y_pred, average='macro', beta=1))
        train_fbeta_score_weighted_all.append(fbeta_score(y_train, y_pred, average='weighted', beta=1))
        train_fbeta_score_samples_all.append(fbeta_score(y_train, y_pred, average='samples', beta=1))
        train_hamming_loss_all.append(hamming_loss(y_train, y_pred))
        train_jaccard_score_micro_all.append(jaccard_score(y_train, y_pred, average='micro'))
        train_jaccard_score_macro_all.append(jaccard_score(y_train, y_pred, average='macro'))
        train_jaccard_score_weighted_all.append(jaccard_score(y_train, y_pred, average='weighted'))
        train_jaccard_score_samples_all.append(jaccard_score(y_train, y_pred, average='samples'))
        train_log_loss_all.append(log_loss(y_train, y_pred))
        train_multilabel_confusion_matrix_all.append(multilabel_confusion_matrix(y_train, y_pred))
        train_precision_recall_fscore_support_micro_all.append(precision_recall_fscore_support(y_train, y_pred, beta=1, average='micro'))
        train_precision_recall_fscore_support_macro_all.append(precision_recall_fscore_support(y_train, y_pred, beta=1, average='macro'))
        train_precision_recall_fscore_support_weighted_all.append(precision_recall_fscore_support(y_train, y_pred, beta=1, average='weighted'))
        train_precision_recall_fscore_support_samples_all.append(precision_recall_fscore_support(y_train, y_pred, beta=1, average='samples'))
        train_roc_auc_score_micro_all.append(roc_auc_score(y_train, y_pred, average='micro'))
        train_roc_auc_score_macro_all.append(roc_auc_score(y_train, y_pred, average='macro'))
        train_roc_auc_score_weighted_all.append(roc_auc_score(y_train, y_pred, average='weighted'))
        train_roc_auc_score_samples_all.append(roc_auc_score(y_train, y_pred, average='samples'))
        train_zero_one_loss_all.append(zero_one_loss(y_train, y_pred, normalize=False))      
        
        # test set 
        y_pred2 = clf.predict(X_test)  
        test_accuracy_score_all.append(accuracy_score(y_test, y_pred2))
        test_fbeta_score_micro_all.append(fbeta_score(y_test, y_pred2, average='micro', beta=1))
        test_fbeta_score_macro_all.append(fbeta_score(y_test, y_pred2, average='macro', beta=1))
        test_fbeta_score_weighted_all.append(fbeta_score(y_test, y_pred2, average='weighted', beta=1))
        test_fbeta_score_samples_all.append(fbeta_score(y_test, y_pred2, average='samples', beta=1))
        test_hamming_loss_all.append(hamming_loss(y_test, y_pred2))
        test_jaccard_score_micro_all.append(jaccard_score(y_test, y_pred2, average='micro'))
        test_jaccard_score_macro_all.append(jaccard_score(y_test, y_pred2, average='macro'))
        test_jaccard_score_weighted_all.append(jaccard_score(y_test, y_pred2, average='weighted'))
        test_jaccard_score_samples_all.append(jaccard_score(y_test, y_pred2, average='samples'))
        test_log_loss_all.append(log_loss(y_test, y_pred2))
        test_multilabel_confusion_matrix_all.append(multilabel_confusion_matrix(y_test, y_pred2))
        test_precision_recall_fscore_support_micro_all.append(precision_recall_fscore_support(y_test, y_pred2, beta=1, average='micro'))
        test_precision_recall_fscore_support_macro_all.append(precision_recall_fscore_support(y_test, y_pred2, beta=1, average='macro'))
        test_precision_recall_fscore_support_weighted_all.append(precision_recall_fscore_support(y_test, y_pred2, beta=1, average='weighted'))
        test_precision_recall_fscore_support_samples_all.append(precision_recall_fscore_support(y_test, y_pred2, beta=1, average='samples'))
        test_roc_auc_score_micro_all.append(roc_auc_score(y_test, y_pred2, average='micro'))
        test_roc_auc_score_macro_all.append(roc_auc_score(y_test, y_pred2, average='macro'))
        test_roc_auc_score_weighted_all.append(roc_auc_score(y_test, y_pred2, average='weighted'))
        test_roc_auc_score_samples_all.append(roc_auc_score(y_test, y_pred2, average='samples'))
        test_zero_one_loss_all.append(zero_one_loss(y_test, y_pred2, normalize=False))    
        print(d)
        params.append(f'max_depth: {d}')
    
    # get max score's parameters 
    print('Highest score: ', max(test_accuracy_score_all))
    print(params[test_accuracy_score_all.index(max(test_accuracy_score_all))])
    max_index = test_accuracy_score_all.index(max(test_accuracy_score_all))
    
    train_scores = []
    for i in training[:11]: 
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
    for i in testing[:11]: 
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
    
    score_names =  ['accuracy', 'fbeta_micro', 'fbeta_macro', 'fbeta_weighted', 'fbeta_samples', 'hamming_loss', 'jaccard_micro', 'jaccard_macro', 'jaccard_weighted', 'jaccard_samples', 'log_loss', 'multilabel_confusion_matrix', 'precision_recall_fsupport_micro', 'precision_recall_fsupport_macro', 'precision_recall_fsupport_weighted', 'precision_recall_fsupport_samples', 'roc_auc_micro', 'roc_auc_macro','roc_auc_weighted', 'roc_auc_samples', 'zero_one_loss', 'average_precision_micro', 'average_precision_macro', 'average_precision_weighted', 'average_precision_samples']
    score_names = score_names[:11]
    
    assert len(train_scores) == len(score_names)
    assert len(test_scores) == len(score_names)
          
    testdot1_4 = pd.DataFrame({'classifier': clf.__class__.__name__
                             , 'training': train_scores
                            , 'test': test_scores
                            , 'score': score_names})
    return testdot1_4


def cross_validation_extratreesclassifier(X, y, cv=5): 
    params = []
    
    # varying parameters
    n_estimators = [1, 2, 5, 7, 10, 15, 20]
    max_depths = [10, 12, 15, 25, 30]
    min_samples_split=[2, 5, 10, 15]
    min_samples_leaf=[1, 2, 5, 10]
    combinations = len(n_estimators) *len(max_depths) *len(min_samples_split) *len(min_samples_leaf) 
    print(combinations)
    
    # cross-validation 
    score_names_all = []
    test_scores_all = []
    train_scores_all = []
    
    for n in n_estimators: 
        for d in max_depths: 
            for ss in min_samples_split: 
                for sl in min_samples_leaf:  
                    clf = ExtraTreesClassifier(n_estimators=n, max_depth=d, min_samples_split=ss, min_samples_leaf=sl, n_jobs=-1, random_state=24, class_weight='balanced')
                    # train on training set  
                    scoring = ['accuracy', 'average_precision',  'f1_micro', 'f1_macro', 'f1_weighted', 'f1_samples', 'precision_micro', 'precision_macro', 'precision_weighted', 'precision_samples', 'recall_micro', 'recall_macro', 'recall_weighted', 'recall_samples'] 
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
                    print(n,d, ss, sl)
                    params.append(f'n_estimators: {n}, max_depth: {d}, min_samples_split: {ss}, min_samples_leaf: {sl}')
    
    
    # get max score's parameters 
    for el in test_scores_all: 
        if el[score_names.index(scoring[0])] == max([i[score_names.index(scoring[0])] for i in test_scores_all]): 
            windi = test_scores_all.index(el)
    
    print('Highest score: ', max([i[score_names.index(scoring[0])] for i in test_scores_all]))
    print(params[windi])

    train_scores = train_scores_all[windi]
    test_scores = test_scores_all[windi]
    score_names =  score_names_all[windi]
    
    cv5_5 = pd.DataFrame({'classifier': clf.__class__.__name__
                             , 'training': train_scores
                            , 'test': test_scores
                            , 'score': score_names})
    return cv5_5
    


def  train_test_split_extratreesclassifier(X_train, y_train): 
    params = []
    
    # varying parameters
    n_estimators = [1, 2, 5, 7, 10, 15, 20]
    max_depths = [10, 12, 15, 25, 30]
    min_samples_split=[2, 5, 10, 15]
    min_samples_leaf=[1, 2, 5, 10]
    combinations = len(n_estimators) *len(max_depths) *len(min_samples_split) *len(min_samples_leaf) 
    print(combinations)
    
    # train-test split
    train_accuracy_score_all = []
    train_fbeta_score_micro_all = []
    train_fbeta_score_macro_all = []
    train_fbeta_score_weighted_all = []
    train_fbeta_score_samples_all = []
    train_hamming_loss_all = []
    train_jaccard_score_micro_all = []
    train_jaccard_score_macro_all = []
    train_jaccard_score_weighted_all = []
    train_jaccard_score_samples_all = []
    train_log_loss_all = []
    train_multilabel_confusion_matrix_all = []
    train_precision_recall_fscore_support_micro_all =[]
    train_precision_recall_fscore_support_macro_all = []
    train_precision_recall_fscore_support_weighted_all = []
    train_precision_recall_fscore_support_samples_all = []
    train_roc_auc_score_micro_all =[]
    train_roc_auc_score_macro_all = []
    train_roc_auc_score_weighted_all = []
    train_roc_auc_score_samples_all = []
    train_zero_one_loss_all = []
    train_average_precision_score_micro_all = []
    train_average_precision_score_macro_all = []
    train_average_precision_score_weighted_all = []
    train_average_precision_score_samples_all = []
    
    test_accuracy_score_all = []
    test_fbeta_score_micro_all = []
    test_fbeta_score_macro_all =  [] 
    test_fbeta_score_weighted_all = []
    test_fbeta_score_samples_all = []
    test_hamming_loss_all = []
    test_jaccard_score_micro_all = []
    test_jaccard_score_macro_all = []
    test_jaccard_score_weighted_all = []
    test_jaccard_score_samples_all = []
    test_log_loss_all = []
    test_multilabel_confusion_matrix_all = []
    test_precision_recall_fscore_support_micro_all =[]
    test_precision_recall_fscore_support_macro_all = []
    test_precision_recall_fscore_support_weighted_all = []
    test_precision_recall_fscore_support_samples_all = []
    test_roc_auc_score_micro_all = []
    test_roc_auc_score_macro_all = []
    test_roc_auc_score_weighted_all = []
    test_roc_auc_score_samples_all = []
    test_zero_one_loss_all = []  
    test_average_precision_score_micro_all = []
    test_average_precision_score_macro_all = []
    test_average_precision_score_weighted_all = []
    test_average_precision_score_samples_all = []
        
    training = [train_accuracy_score_all, train_fbeta_score_micro_all, train_fbeta_score_macro_all, train_fbeta_score_weighted_all, train_fbeta_score_samples_all,train_hamming_loss_all,train_jaccard_score_micro_all,train_jaccard_score_macro_all,train_jaccard_score_weighted_all,train_jaccard_score_samples_all,train_log_loss_all,train_multilabel_confusion_matrix_all,train_precision_recall_fscore_support_micro_all,train_precision_recall_fscore_support_macro_all,train_precision_recall_fscore_support_weighted_all,train_precision_recall_fscore_support_samples_all,train_roc_auc_score_micro_all,train_roc_auc_score_macro_all,train_roc_auc_score_weighted_all,train_roc_auc_score_samples_all,train_zero_one_loss_all,train_average_precision_score_micro_all,train_average_precision_score_macro_all,train_average_precision_score_weighted_all,train_average_precision_score_samples_all]
    testing = [test_accuracy_score_all, test_fbeta_score_micro_all, test_fbeta_score_macro_all, test_fbeta_score_weighted_all, test_fbeta_score_samples_all,test_hamming_loss_all,test_jaccard_score_micro_all,test_jaccard_score_macro_all,test_jaccard_score_weighted_all,test_jaccard_score_samples_all,test_log_loss_all,test_multilabel_confusion_matrix_all,test_precision_recall_fscore_support_micro_all,test_precision_recall_fscore_support_macro_all,test_precision_recall_fscore_support_weighted_all,test_precision_recall_fscore_support_samples_all,test_roc_auc_score_micro_all,test_roc_auc_score_macro_all,test_roc_auc_score_weighted_all,test_roc_auc_score_samples_all,test_zero_one_loss_all,test_average_precision_score_micro_all,test_average_precision_score_macro_all,test_average_precision_score_weighted_all,test_average_precision_score_samples_all]
    
    for n in n_estimators: 
        for d in max_depths: 
            for ss in min_samples_split: 
                for sl in min_samples_leaf: 
                    clf = ExtraTreesClassifier(n_estimators=n, max_depth=d, min_samples_split=ss, min_samples_leaf=sl, n_jobs=-1, random_state=24, class_weight='balanced')
                    # train on training set  
                    clf.fit(X_train, y_train)
                    # training set 
                    y_pred = clf.predict(X_train)    
                    train_accuracy_score_all.append(accuracy_score(y_train, y_pred))
                    train_fbeta_score_micro_all.append(fbeta_score(y_train, y_pred, average='micro', beta=1))
                    train_fbeta_score_macro_all.append(fbeta_score(y_train, y_pred, average='macro', beta=1))
                    train_fbeta_score_weighted_all.append(fbeta_score(y_train, y_pred, average='weighted', beta=1))
                    train_fbeta_score_samples_all.append(fbeta_score(y_train, y_pred, average='samples', beta=1))
                    train_hamming_loss_all.append(hamming_loss(y_train, y_pred))
                    train_jaccard_score_micro_all.append(jaccard_score(y_train, y_pred, average='micro'))
                    train_jaccard_score_macro_all.append(jaccard_score(y_train, y_pred, average='macro'))
                    train_jaccard_score_weighted_all.append(jaccard_score(y_train, y_pred, average='weighted'))
                    train_jaccard_score_samples_all.append(jaccard_score(y_train, y_pred, average='samples'))
                    train_log_loss_all.append(log_loss(y_train, y_pred))
                    train_multilabel_confusion_matrix_all.append(multilabel_confusion_matrix(y_train, y_pred))
                    train_precision_recall_fscore_support_micro_all.append(precision_recall_fscore_support(y_train, y_pred, beta=1, average='micro'))
                    train_precision_recall_fscore_support_macro_all.append(precision_recall_fscore_support(y_train, y_pred, beta=1, average='macro'))
                    train_precision_recall_fscore_support_weighted_all.append(precision_recall_fscore_support(y_train, y_pred, beta=1, average='weighted'))
                    train_precision_recall_fscore_support_samples_all.append(precision_recall_fscore_support(y_train, y_pred, beta=1, average='samples'))
                    train_roc_auc_score_micro_all.append(roc_auc_score(y_train, y_pred, average='micro'))
                    train_roc_auc_score_macro_all.append(roc_auc_score(y_train, y_pred, average='macro'))
                    train_roc_auc_score_weighted_all.append(roc_auc_score(y_train, y_pred, average='weighted'))
                    train_roc_auc_score_samples_all.append(roc_auc_score(y_train, y_pred, average='samples'))
                    train_zero_one_loss_all.append(zero_one_loss(y_train, y_pred, normalize=False))                  
                    
                    # test set 
                    y_pred2 = clf.predict(X_test)  
                    test_accuracy_score_all.append(accuracy_score(y_test, y_pred2))
                    test_fbeta_score_micro_all.append(fbeta_score(y_test, y_pred2, average='micro', beta=1))
                    test_fbeta_score_macro_all.append(fbeta_score(y_test, y_pred2, average='macro', beta=1))
                    test_fbeta_score_weighted_all.append(fbeta_score(y_test, y_pred2, average='weighted', beta=1))
                    test_fbeta_score_samples_all.append(fbeta_score(y_test, y_pred2, average='samples', beta=1))
                    test_hamming_loss_all.append(hamming_loss(y_test, y_pred2))
                    test_jaccard_score_micro_all.append(jaccard_score(y_test, y_pred2, average='micro'))
                    test_jaccard_score_macro_all.append(jaccard_score(y_test, y_pred2, average='macro'))
                    test_jaccard_score_weighted_all.append(jaccard_score(y_test, y_pred2, average='weighted'))
                    test_jaccard_score_samples_all.append(jaccard_score(y_test, y_pred2, average='samples'))
                    test_log_loss_all.append(log_loss(y_test, y_pred2))
                    test_multilabel_confusion_matrix_all.append(multilabel_confusion_matrix(y_test, y_pred2))
                    test_precision_recall_fscore_support_micro_all.append(precision_recall_fscore_support(y_test, y_pred2, beta=1, average='micro'))
                    test_precision_recall_fscore_support_macro_all.append(precision_recall_fscore_support(y_test, y_pred2, beta=1, average='macro'))
                    test_precision_recall_fscore_support_weighted_all.append(precision_recall_fscore_support(y_test, y_pred2, beta=1, average='weighted'))
                    test_precision_recall_fscore_support_samples_all.append(precision_recall_fscore_support(y_test, y_pred2, beta=1, average='samples'))
                    test_roc_auc_score_micro_all.append(roc_auc_score(y_test, y_pred2, average='micro'))
                    test_roc_auc_score_macro_all.append(roc_auc_score(y_test, y_pred2, average='macro'))
                    test_roc_auc_score_weighted_all.append(roc_auc_score(y_test, y_pred2, average='weighted'))
                    test_roc_auc_score_samples_all.append(roc_auc_score(y_test, y_pred2, average='samples'))
                    test_zero_one_loss_all.append(zero_one_loss(y_test, y_pred2, normalize=False))    
                    print(n,d,ss, sl)
                    params.append(f'n_estimators: {n}, max_depth: {d}, min_samples_split: {ss}, min_samples_leaf: {sl}')
    
    # get max score's parameters 
    print('Highest score: ', max(test_accuracy_score_all))
    print(params[test_accuracy_score_all.index(max(test_accuracy_score_all))])
    max_index = test_accuracy_score_all.index(max(test_accuracy_score_all))
    
    train_scores = []
    for i in training[:11]: 
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
    for i in testing[:11]: 
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
    
    score_names =  ['accuracy', 'fbeta_micro', 'fbeta_macro', 'fbeta_weighted', 'fbeta_samples', 'hamming_loss', 'jaccard_micro', 'jaccard_macro', 'jaccard_weighted', 'jaccard_samples', 'log_loss', 'multilabel_confusion_matrix', 'precision_recall_fsupport_micro', 'precision_recall_fsupport_macro', 'precision_recall_fsupport_weighted', 'precision_recall_fsupport_samples', 'roc_auc_micro', 'roc_auc_macro','roc_auc_weighted', 'roc_auc_samples', 'zero_one_loss', 'average_precision_micro', 'average_precision_macro', 'average_precision_weighted', 'average_precision_samples']
    score_names = score_names[:11]
    
    
    
    assert len(train_scores) == len(score_names)
    assert len(test_scores) == len(score_names)
          
    testdot1_5 = pd.DataFrame({'classifier': clf.__class__.__name__
                             , 'training': train_scores
                            , 'test': test_scores
                            , 'score': score_names})
    return testdot1_5


def cross_validation_randomforestclassifier(X, y, cv=5): 
    params = []
   
    # varying parameters
    n_estimators=[100, 300, 500]
    max_depth=[2, 4, 6]
    min_samples_split=[2, 5, 10]
    min_samples_leaf=[1, 2, 5]
    combinations = len(n_estimators) *len(max_depth) *len(min_samples_split) *len(min_samples_leaf)
    print(combinations)
    score_names_all = []
    test_scores_all = []
    train_scores_all = []
    
    for n in n_estimators: 
        for d in max_depth: 
            for ss in min_samples_split: 
                for sl in min_samples_leaf: 
                    clf = RandomForestClassifier(n_estimators=n, max_depth=d, min_samples_split=ss, min_samples_leaf=sl, max_features=None, n_jobs=-1, random_state=24, class_weight='balanced')
                    # train on training set  
                    scoring = ['accuracy', 'average_precision',  'f1_micro', 'f1_macro', 'f1_weighted', 'f1_samples', 'precision_micro', 'precision_macro', 'precision_weighted', 'precision_samples', 'recall_micro', 'recall_macro', 'recall_weighted', 'recall_samples'] 
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
                    print(n, d, ss, sl)
                    params.append(f'n_estimators: {n}, max_depth: {d}, min_samples_split: {ss}, min_samples_leaf: {sl}')
    
    
    # get max score's parameters 
    for el in test_scores_all: 
        if el[score_names.index(scoring[0])] == max([i[score_names.index(scoring[0])] for i in test_scores_all]): 
            windi = test_scores_all.index(el)
    
    print('Highest score: ', max([i[score_names.index(scoring[0])] for i in test_scores_all]))
    print(params[windi])

    train_scores = train_scores_all[windi]
    test_scores = test_scores_all[windi]
    score_names =  score_names_all[windi]
    
    cv5_6 = pd.DataFrame({'classifier': clf.__class__.__name__
                             , 'training': train_scores
                            , 'test': test_scores
                            , 'score': score_names})
        
    return cv5_6



def train_test_split_randomforestclassifier(X_train, y_train): 
    params = []
   
    # varying parameters
    n_estimators=[100, 300, 500]
    max_depth=[2, 4, 6]
    min_samples_split=[2, 5, 10]
    min_samples_leaf=[1, 2, 5]
    combinations = len(n_estimators) *len(max_depth) *len(min_samples_split) *len(min_samples_leaf)
    print(combinations)
    
    # train-test split
    train_accuracy_score_all = []
    train_fbeta_score_micro_all = []
    train_fbeta_score_macro_all = []
    train_fbeta_score_weighted_all = []
    train_fbeta_score_samples_all = []
    train_hamming_loss_all = []
    train_jaccard_score_micro_all = []
    train_jaccard_score_macro_all = []
    train_jaccard_score_weighted_all = []
    train_jaccard_score_samples_all = []
    train_log_loss_all = []
    train_multilabel_confusion_matrix_all = []
    train_precision_recall_fscore_support_micro_all =[]
    train_precision_recall_fscore_support_macro_all = []
    train_precision_recall_fscore_support_weighted_all = []
    train_precision_recall_fscore_support_samples_all = []
    train_roc_auc_score_micro_all =[]
    train_roc_auc_score_macro_all = []
    train_roc_auc_score_weighted_all = []
    train_roc_auc_score_samples_all = []
    train_zero_one_loss_all = []
    train_average_precision_score_micro_all = []
    train_average_precision_score_macro_all = []
    train_average_precision_score_weighted_all = []
    train_average_precision_score_samples_all = []
    
    test_accuracy_score_all = []
    test_fbeta_score_micro_all = []
    test_fbeta_score_macro_all =  [] 
    test_fbeta_score_weighted_all = []
    test_fbeta_score_samples_all = []
    test_hamming_loss_all = []
    test_jaccard_score_micro_all = []
    test_jaccard_score_macro_all = []
    test_jaccard_score_weighted_all = []
    test_jaccard_score_samples_all = []
    test_log_loss_all = []
    test_multilabel_confusion_matrix_all = []
    test_precision_recall_fscore_support_micro_all =[]
    test_precision_recall_fscore_support_macro_all = []
    test_precision_recall_fscore_support_weighted_all = []
    test_precision_recall_fscore_support_samples_all = []
    test_roc_auc_score_micro_all = []
    test_roc_auc_score_macro_all = []
    test_roc_auc_score_weighted_all = []
    test_roc_auc_score_samples_all = []
    test_zero_one_loss_all = []  
    test_average_precision_score_micro_all = []
    test_average_precision_score_macro_all = []
    test_average_precision_score_weighted_all = []
    test_average_precision_score_samples_all = []
        
    training = [train_accuracy_score_all, train_fbeta_score_micro_all, train_fbeta_score_macro_all, train_fbeta_score_weighted_all, train_fbeta_score_samples_all,train_hamming_loss_all,train_jaccard_score_micro_all,train_jaccard_score_macro_all,train_jaccard_score_weighted_all,train_jaccard_score_samples_all,train_log_loss_all,train_multilabel_confusion_matrix_all,train_precision_recall_fscore_support_micro_all,train_precision_recall_fscore_support_macro_all,train_precision_recall_fscore_support_weighted_all,train_precision_recall_fscore_support_samples_all,train_roc_auc_score_micro_all,train_roc_auc_score_macro_all,train_roc_auc_score_weighted_all,train_roc_auc_score_samples_all,train_zero_one_loss_all,train_average_precision_score_micro_all,train_average_precision_score_macro_all,train_average_precision_score_weighted_all,train_average_precision_score_samples_all]
    testing = [test_accuracy_score_all, test_fbeta_score_micro_all, test_fbeta_score_macro_all, test_fbeta_score_weighted_all, test_fbeta_score_samples_all,test_hamming_loss_all,test_jaccard_score_micro_all,test_jaccard_score_macro_all,test_jaccard_score_weighted_all,test_jaccard_score_samples_all,test_log_loss_all,test_multilabel_confusion_matrix_all,test_precision_recall_fscore_support_micro_all,test_precision_recall_fscore_support_macro_all,test_precision_recall_fscore_support_weighted_all,test_precision_recall_fscore_support_samples_all,test_roc_auc_score_micro_all,test_roc_auc_score_macro_all,test_roc_auc_score_weighted_all,test_roc_auc_score_samples_all,test_zero_one_loss_all,test_average_precision_score_micro_all,test_average_precision_score_macro_all,test_average_precision_score_weighted_all,test_average_precision_score_samples_all]
    
    for n in n_estimators: 
        for d in max_depth: 
            for ss in min_samples_split: 
                for sl in min_samples_leaf: 
                    clf = RandomForestClassifier(n_estimators=n, max_depth=d, min_samples_split=ss, min_samples_leaf=sl, max_features=None, n_jobs=-1, random_state=24, class_weight='balanced')
                    # train on training set  
                    clf.fit(X_train, y_train) 
                    y_pred = clf.predict(X_train)    
                    train_accuracy_score_all.append(accuracy_score(y_train, y_pred))
                    train_fbeta_score_micro_all.append(fbeta_score(y_train, y_pred, average='micro', beta=1))
                    train_fbeta_score_macro_all.append(fbeta_score(y_train, y_pred, average='macro', beta=1))
                    train_fbeta_score_weighted_all.append(fbeta_score(y_train, y_pred, average='weighted', beta=1))
                    train_fbeta_score_samples_all.append(fbeta_score(y_train, y_pred, average='samples', beta=1))
                    train_hamming_loss_all.append(hamming_loss(y_train, y_pred))
                    train_jaccard_score_micro_all.append(jaccard_score(y_train, y_pred, average='micro'))
                    train_jaccard_score_macro_all.append(jaccard_score(y_train, y_pred, average='macro'))
                    train_jaccard_score_weighted_all.append(jaccard_score(y_train, y_pred, average='weighted'))
                    train_jaccard_score_samples_all.append(jaccard_score(y_train, y_pred, average='samples'))
                    train_log_loss_all.append(log_loss(y_train, y_pred))
                    train_multilabel_confusion_matrix_all.append(multilabel_confusion_matrix(y_train, y_pred))
                    train_precision_recall_fscore_support_micro_all.append(precision_recall_fscore_support(y_train, y_pred, beta=1, average='micro'))
                    train_precision_recall_fscore_support_macro_all.append(precision_recall_fscore_support(y_train, y_pred, beta=1, average='macro'))
                    train_precision_recall_fscore_support_weighted_all.append(precision_recall_fscore_support(y_train, y_pred, beta=1, average='weighted'))
                    train_precision_recall_fscore_support_samples_all.append(precision_recall_fscore_support(y_train, y_pred, beta=1, average='samples'))
                    train_roc_auc_score_micro_all.append(roc_auc_score(y_train, y_pred, average='micro'))
                    train_roc_auc_score_macro_all.append(roc_auc_score(y_train, y_pred, average='macro'))
                    train_roc_auc_score_weighted_all.append(roc_auc_score(y_train, y_pred, average='weighted'))
                    train_roc_auc_score_samples_all.append(roc_auc_score(y_train, y_pred, average='samples'))
                    train_zero_one_loss_all.append(zero_one_loss(y_train, y_pred, normalize=False))    
     
                    # test set 
                    y_pred2 = clf.predict(X_test)  
                    test_accuracy_score_all.append(accuracy_score(y_test, y_pred2))
                    test_fbeta_score_micro_all.append(fbeta_score(y_test, y_pred2, average='micro', beta=1))
                    test_fbeta_score_macro_all.append(fbeta_score(y_test, y_pred2, average='macro', beta=1))
                    test_fbeta_score_weighted_all.append(fbeta_score(y_test, y_pred2, average='weighted', beta=1))
                    test_fbeta_score_samples_all.append(fbeta_score(y_test, y_pred2, average='samples', beta=1))
                    test_hamming_loss_all.append(hamming_loss(y_test, y_pred2))
                    test_jaccard_score_micro_all.append(jaccard_score(y_test, y_pred2, average='micro'))
                    test_jaccard_score_macro_all.append(jaccard_score(y_test, y_pred2, average='macro'))
                    test_jaccard_score_weighted_all.append(jaccard_score(y_test, y_pred2, average='weighted'))
                    test_jaccard_score_samples_all.append(jaccard_score(y_test, y_pred2, average='samples'))
                    test_log_loss_all.append(log_loss(y_test, y_pred2))
                    test_multilabel_confusion_matrix_all.append(multilabel_confusion_matrix(y_test, y_pred2))
                    test_precision_recall_fscore_support_micro_all.append(precision_recall_fscore_support(y_test, y_pred2, beta=1, average='micro'))
                    test_precision_recall_fscore_support_macro_all.append(precision_recall_fscore_support(y_test, y_pred2, beta=1, average='macro'))
                    test_precision_recall_fscore_support_weighted_all.append(precision_recall_fscore_support(y_test, y_pred2, beta=1, average='weighted'))
                    test_precision_recall_fscore_support_samples_all.append(precision_recall_fscore_support(y_test, y_pred2, beta=1, average='samples'))
                    test_roc_auc_score_micro_all.append(roc_auc_score(y_test, y_pred2, average='micro'))
                    test_roc_auc_score_macro_all.append(roc_auc_score(y_test, y_pred2, average='macro'))
                    test_roc_auc_score_weighted_all.append(roc_auc_score(y_test, y_pred2, average='weighted'))
                    test_roc_auc_score_samples_all.append(roc_auc_score(y_test, y_pred2, average='samples'))
                    test_zero_one_loss_all.append(zero_one_loss(y_test, y_pred2, normalize=False))    
                    print(n,d,ss,sl)
                    params.append(f'n_estimators: {n}, max_depth: {d}, min_samples_split: {ss}, min_samples_leaf: {sl}')
    
    # get max score's parameters 
    print('Highest score: ', max(test_accuracy_score_all))
    print(params[test_accuracy_score_all.index(max(test_accuracy_score_all))])
    print('Number of combinations: ', len(n_neighbors)*len(leaf_sizes))
    
    max_index = test_accuracy_score_all.index(max(test_accuracy_score_all))
    
    train_scores = []
    for i in training[:11]: 
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
    for i in testing[:11]: 
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
    
    score_names =  ['accuracy', 'fbeta_micro', 'fbeta_macro', 'fbeta_weighted', 'fbeta_samples', 'hamming_loss', 'jaccard_micro', 'jaccard_macro', 'jaccard_weighted', 'jaccard_samples', 'log_loss', 'multilabel_confusion_matrix', 'precision_recall_fsupport_micro', 'precision_recall_fsupport_macro', 'precision_recall_fsupport_weighted', 'precision_recall_fsupport_samples', 'roc_auc_micro', 'roc_auc_macro','roc_auc_weighted', 'roc_auc_samples', 'zero_one_loss', 'average_precision_micro', 'average_precision_macro', 'average_precision_weighted', 'average_precision_samples']
    score_names = score_names[:11]
    
    assert len(train_scores) == len(score_names)
    assert len(test_scores) == len(score_names)
          
    testdot1_6 = pd.DataFrame({'classifier': clf.__class__.__name__
                             , 'training': train_scores
                            , 'test': test_scores
                            , 'score': score_names})
    return testdot1_6


  
def best_train_test_split_extratreesclassifier( X_train, y_train, n_estimators, max_depth, min_samples_split, min_samples_leaf, n_jobs, random_state, class_weight):       
    train_accuracy_score_all = []
    train_fbeta_score_micro_all = []
    train_fbeta_score_macro_all = []
    
    test_accuracy_score_all = []
    test_fbeta_score_micro_all = []
    test_fbeta_score_macro_all =  [] 
    
        
    training = [train_accuracy_score_all, train_fbeta_score_micro_all, train_fbeta_score_macro_all]
    testing = [test_accuracy_score_all, test_fbeta_score_micro_all, test_fbeta_score_macro_all]
    
    clf = ExtraTreesClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, n_jobs=n_jobs, random_state=random_state, class_weight=class_weight) 
    clf.fit(X_train, y_train)
    # training set 
    y_pred = clf.predict(X_train)    
    train_accuracy_score_all.append(accuracy_score(y_train, y_pred))
    train_fbeta_score_micro_all.append(fbeta_score(y_train, y_pred, average='micro', beta=1))
    train_fbeta_score_macro_all.append(fbeta_score(y_train, y_pred, average='macro', beta=1))     
    
    # test set 
    y_pred2 = clf.predict(X_test)  
    test_accuracy_score_all.append(accuracy_score(y_test, y_pred2))
    test_fbeta_score_micro_all.append(fbeta_score(y_test, y_pred2, average='micro', beta=1))
    test_fbeta_score_macro_all.append(fbeta_score(y_test, y_pred2, average='macro', beta=1))
    
    max_index = test_accuracy_score_all.index(max(test_accuracy_score_all))
    
    train_scores = []
    for i in training[:11]: 
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
    for i in testing[:11]: 
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
    
    score_names =  ['accuracy', 'fbeta_micro', 'fbeta_macro']
    
    assert len(train_scores) == len(score_names)
    assert len(test_scores) == len(score_names)
    
    best_testdot1_5 = pd.DataFrame({'classifier': clf.__class__.__name__
                             , 'training': train_scores
                            , 'test': test_scores
                            , 'score': score_names})
    return best_testdot1_5, clf



def best_cross_validation_randomforestclassifier(X, y, n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, n_jobs, random_state, class_weight): 
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, max_features=max_features, n_jobs=n_jobs, random_state=random_state, class_weight=class_weight)
    # train on training set  
    scoring = ['accuracy',  'f1_micro', 'f1_macro', 'f1_weighted', 'f1_samples'] 
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
    
    best_cv5_6 = pd.DataFrame({'classifier': clf.__class__.__name__
                             , 'training': train_scores
                            , 'test': test_scores
                            , 'score': score_names})
    return best_cv5_6, clf


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
    # processing classifiers: search best parameter combination   
    cv5_1 = cross_validation_knn(X, y, cv=5)
    testdot1_1 = train_test_split_knn(X_train, y_train)

    testdot1_2 = train_test_split_radiusneighborsclassifier(X_train, y_train)
    
    cv5_3 = cross_validation_decisiontreeclassifier(X, y, cv=5)
    testdot1_3 = train_test_split_decisiontreeclassifier(X_train, y_train)

    cv5_4 = cross_validation_extratreeclassifier(X, y, cv=5)
    testdot1_4 = train_test_split_extratreeclassifier(X_train, y_train)

    cv5_5 = cross_validation_extratreesclassifier(X, y, cv=5)
    testdot1_5 = train_test_split_extratreesclassifier(X_train, y_train)

    cv5_6 = cross_validation_randomforestclassifier(X, y, cv=5)
    testdot1_6 = train_test_split_randomforestclassifier(X_train, y_train)


#%%
    # best models with single parameter combination
    
    # best parameters
    n_estimators=10
    max_depth=25
    min_samples_split=2
    min_samples_leaf=1
    # training
    start = timer()
    best_testdot1_5, clf = best_train_test_split_extratreesclassifier(X_train, y_train,n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, n_jobs=-1, random_state=24, class_weight='balanced')    
    end = timer()
    duration = end - start
    print(clf.__class__.__name__)
    test_score = np.round(best_testdot1_5['test'][best_testdot1_5['score']=='fbeta_macro'].iloc[0],2)
    print('Highest score: ', test_score)
    print('duration: ', duration, 'seconds.')

    # best parameters
    n_estimators=300
    max_depth=6
    min_samples_split=5
    min_samples_leaf=1
    # training
    start = timer()
    best_cv5_6, clf = best_cross_validation_randomforestclassifier(X_train, y_train, n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, max_features=None, n_jobs=-1, random_state=24, class_weight='balanced')    
    end = timer()
    duration = end - start
    print(clf.__class__.__name__)
    test_score = np.round(best_cv5_6['test'][best_cv5_6['score']=='f1_weighted'].iloc[0],2)
    print('Highest score: ', test_score)
    print('duration: ', duration, 'seconds.')


#%%
    # save model   
    os.chdir(SAVE_PATH)
    filename = f'model_{SOURCE}_{SYSTEM}_{METHOD}_{clf.__class__.__name__}_nest{n_estimators}_d{max_depth}_mss{min_samples_split}_msl{min_samples_leaf}_train{len(X)}_cat{df[LABEL].nunique()}_testacc{test_score}.sav'
    pickle.dump(clf, open(filename, 'wb'))


