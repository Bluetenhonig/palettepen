# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 16:25:48 2020

@author: Linda Samsinger

=====================
ML Classification Comparison
=====================


Problem: color classification - color value to basic color 
Score: GridSearchCV
Classifiers: multi-class 


"""

# import modules 
 
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# preprocessing
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
# machine learning: scores 
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
# machine learning: metrics 
from sklearn.metrics import accuracy_score

#%%

# declare variables 
PATH = r'D:\thesis\input_color_name_dictionaries\thesaurus\datasets'
FILE = 'eeffcnd_thesaurus_basicitten_upinterval.xlsx'
SOURCE = 'THESAURUS'
SYSTEM = 'ITTEN'
METHOD = 'INTERVAL' 

FEATURE = 'cielab'
LABEL =  'cat1'


#%%

# functions
 
def analyze_dataset(df, label): 
    """ exploratory data analysis """
    print(f"Basic colors' distribution {SYSTEM} in the color name dictionary {SOURCE}: ")
    valcts = df[label].value_counts()
    print(valcts)
    valctsplt = df[label].value_counts().plot(kind='bar')
    print(valctsplt)


def get_X_y(df, lab_column, label): 
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
    
    # X and y 
    X = np.array(lab2pts)
    y = np.array(lab2dt)
    print(f"You have {len(X)} colors in color name dictionary {SOURCE} for {df[LABEL].nunique()} color categories in LAB.")
    # shuffle 
    X, y = shuffle(X, y, random_state=24)   
    return X, y 


def grid_search_pipeline(X, y): 
    """ making pipeline and gridsearchcv"""
    clf = Pipeline(steps=[  
                        ('scaler', StandardScaler()) 
                        , ('clf', LogisticRegression())
                   ]) 

    param_grid = [
                 { 'clf': [DecisionTreeClassifier(random_state=24, max_leaf_nodes=1000, class_weight='balanced')], 
             'clf__max_depth':  [5, 7, 10, 11, 12, 13, 14,  15, 16, 17]
                    }, 
                 { 'clf': [ExtraTreeClassifier(max_features=None, random_state=24)], 
             'clf__max_depth': [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 30, 50, 100]
                    }, 
                 { 'clf': [ExtraTreesClassifier(max_features=None, n_jobs=-1, random_state=24)], 
             'clf__n_estimators':  [1, 2, 5, 7, 10, 15, 20], 
             'clf__max_depth':  [2, 4, 6, 8, 10, 12, 15, 25, 30]
                    }, 
                 { 'clf': [RandomForestClassifier(max_features=None, n_jobs=-1, random_state=24)], 
             'clf__n_estimators':  [61, 62, 63, 64, 65, 66, 67, 68, 69, 70], 
             'clf__max_depth':  [2, 4, 6, 8, 10, 12, 15, 25, 30]
                    },
              { 'clf': [KNeighborsClassifier(weights='distance', n_jobs = -1)], 
              'clf__n_neighbors':  list(range(1,30)),
               'clf__leaf_size': list(range(1,30))   # estimator__ prefix to get through MOC
                  }, 
              {'clf': [RadiusNeighborsClassifier(weights='distance', n_jobs = -1 )], 
               'clf__radius': [300, 200, 100],
               'clf__leaf_size': [100, 50, 15]
               },
                { 'clf': [RidgeClassifier(max_iter=110, class_weight='balanced', random_state=24)], 
             'clf__alpha': [100000, 50000, 10000, 5000]
                  },
                 { 'clf': [LogisticRegression(penalty='elasticnet', random_state=24, max_iter=1000, n_jobs = -1, l1_ratio=.5)], 
             'clf__solver':   ['saga'],
             'clf__multi_class':   ['auto', 'ovr', 'multinomial']
                   }, 
                 { 'clf': [LogisticRegressionCV(penalty='elasticnet',  n_jobs=-1, random_state=24, l1_ratios=[.5])], 
             'clf__solver':   ['saga'],
             'clf__multi_class':   ['auto', 'ovr', 'multinomial']
                    },  
                 { 'clf': [RidgeClassifierCV(cv = 5, class_weight='balanced')], 
             'clf__class_weight':   [None, 'balanced']
                  }, 
                 { 'clf': [GaussianNB()]
                  }, 
                 { 'clf': [NearestCentroid()], 
             'clf__shrink_threshold':   [6,5,4, 4.5, 3, 3.5, 2,1]
                  }, 
              { 'clf': [LinearSVC(dual = False, random_state=24, max_iter=1000000000)], 
               'clf__C': np.linspace(0,10,200).tolist(), 
               'clf__multi_class': ['ovr', 'crammer_singer']
               },  #based on https://scikit-learn.org/stable/modules/svm.html, param C trades off misclassification of training examples against simplicity of the decision surface. A low C makes the decision surface smooth, while a high C aims at classifying all training examples correctly, exponentially spaced to get good values
                 { 'clf': [LinearDiscriminantAnalysis()], 
             'clf__solver':   ['lsqr', 'eigen'], 
             'clf__shrinkage':   [None, 'auto'] 
                   }, 
                 { 'clf': [QuadraticDiscriminantAnalysis()]
                   },   
                 { 'clf': [LabelPropagation(kernel='knn', gamma=0, n_jobs=-1)], 
             'clf__n_neighbors':   np.arange(1, 21).tolist(), 
                   }, 
                 { 'clf': [LabelSpreading(kernel='knn', gamma=0, n_jobs=-1)], 
             'clf__n_neighbors':   np.arange(1, 21).tolist(), 
             'clf__alpha':  [.2, .4, .6, .8]
                   }, 
                 { 'clf': [NuSVC(max_iter=-1, random_state=24)], 
             'clf__nu':  [1, 10, 20], 
             'clf__kernel': ['linear', 'poly','rbf', 'sigmoid', 'precomputed'], 
               'clf__gamma':  ['scale', 'auto'], 
               'clf__shrinking':  [True,False], 
               'clf__class_weight':  [None, 'balanced'], 
               'clf__decision_function_shape':  ['ovo','ovr'], 
                   }, 
                 { 'clf': [SVC(probability=True, random_state=24)], 
             'clf__C':  [.2, .4, .6, .8], 
             'clf__kernel': ['linear', 'poly','rbf', 'sigmoid', 'precomputed'], 
               'clf__gamma':  ['scale', 'auto'], 
               'clf__shrinking':  [True,False], 
               'clf__class_weight':  [None, 'balanced'], 
               'clf__decision_function_shape':  ['ovo','ovr'], 
                   }, 
            { 'clf': [GaussianProcessClassifier(copy_X_train=False, random_state=24, n_jobs=-1)], 
             'clf__n_restarts_optimizer':  [1, 2]
                 }, 
                 { 'clf': [SGDClassifier(n_jobs=-1, random_state=24, )], 
             'clf__penalty':  ['l2', 'elasticnet'], 
             'clf__learning_rate': ['optimal', 'adaptive'], 
             'clf__eta0': [0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4],
             'clf__class_weight': [None, 'balanced'],
                   }, 
                 { 'clf': [Perceptron(n_jobs=-1, random_state=24, )], 
             'clf__penalty':  ['l2', 'elasticnet'], 
             'clf__class_weight': [None, 'balanced'],
                   }, 
                 { 'clf': [PassiveAggressiveClassifier(n_jobs=-1, random_state=24, average=True)], 
             'clf__C':  [.001, .01, .1, .2, .4, .6, .8, 1, 1.2], 
             'clf__class_weight': [None, 'balanced']
                   }, 
                 { 'clf': [GradientBoostingClassifier(random_state=24)], 
             'clf__learning_rate': [.06, .07, .08,.09], 
             'clf__n_estimators': [180, 190, 200],
             'clf__max_depth': [2, 4, 6, 8],
             'clf__init': [None, 'zero']
                    }, 
            ]
    

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
    
    search = GridSearchCV(clf, param_grid, scoring=scoring, refit= 'balanced_accuracy', n_jobs=-1, cv=5, return_train_score=True)
    search.fit(X, y)
      
    print(f'Training Machine Learning Classifier for {SYSTEM} Color Categories: successful!')
    
    return search 

def best_result_gridsearch(search): 
    multiclass_train = []
    multiclass_test = []
    multiclass_scorers = []
    multiclass_scorenames = []
    for key in search.cv_results_.keys():
        if key.startswith('mean_'):
            multiclass_scorers.append(key)
        if key.startswith('mean_test'): 
            key = key[10:]
            multiclass_scorenames.append(key)
            
    for key in  multiclass_scorers:    
       if key.startswith('mean_train'):
            train = search.cv_results_[key][search.best_index_]
            multiclass_train.append(train)
       if key.startswith('mean_test'):
            test = search.cv_results_[key][search.best_index_]
            multiclass_test.append(test)
    return multiclass_train, multiclass_test, multiclass_scorers, multiclass_scorenames


def best_result_gridsearch_per_model(search): 
    """ determine best result / index for each model class separately """
    clfs = []
    params = []
    metrics_train = {}
    metrics_test = {}
    
    for i, clf in enumerate(range(len(search.cv_results_['params']))):
        cl = search.cv_results_['params'][i]['clf'].__class__.__name__
        clfs.append(cl)
        parama = list(search.cv_results_['params'][i].keys())
        parama.remove('clf')
        pra = []
        for para in parama:
            pr = search.cv_results_['params'][clf][para]
            para = para[5:]
            pra.append(f'{para}: {pr}')
        params.append(pra)
    return clfs, params, metrics_train, metrics_test


def best_result_gridsearch_per_metric(search, clfs, params, metrics_train, metrics_test):
    multiclass_train =[] 
    multiclass_test = []
    keys_train = []
    keys_test = []
    
    search.cv_results_['mean_test_balanced_accuracy']
    for key in  multiclass_scorers:
        if key.startswith('mean_train'):
            train = search.cv_results_[key]
            multiclass_train.append(train)
            keys_train.append(key[5:])
        if key.startswith('mean_test'):
            test = search.cv_results_[key]
            multiclass_test.append(test)
            keys_test.append(key[5:])
    for i, arr in  enumerate(multiclass_train):
        metrics_train[keys_train[i]] = arr
    for i, arr in  enumerate(multiclass_test):
        metrics_test[keys_test[i]] = arr
        
    models = {'clfs': clfs
          , 'params': params}    
    models.update(metrics_train)
    models.update(metrics_test)   
    harvest = pd.DataFrame(models)
    harvest2 = harvest.groupby(['clfs']).agg({'test_balanced_accuracy':'max' })

    return multiclass_train, multiclass_test, keys_train, keys_test, harvest, harvest2



def gridsearch_all_params(harvest, harvest2): 
    idx = []
    for i in range(len(harvest2['test_balanced_accuracy'])): 
        print(harvest[['clfs', 'params']][harvest['test_balanced_accuracy'] == harvest2['test_balanced_accuracy'].iloc[i]])
        indices = harvest[['clfs', 'params']][harvest['test_balanced_accuracy'] == harvest2['test_balanced_accuracy'].iloc[i]].index
        idx.append(list(indices))
    
    idx =  [item[0] if len(item)>=2 else item[0] for item in idx ]
    
    trains = []
    tests = []
    harvest3 = harvest.iloc[idx]
    for el in harvest3.columns: 
        if el.startswith('train'): 
            for i, im in enumerate(idx): 
                print(el)
                tr = harvest3[el].iloc[i]
                trains.append(tr)
    for el in harvest3.columns: 
        if el.startswith('test'): 
            for i, im in enumerate(idx): 
                print(el)
                tr = harvest3[el].iloc[i]
                tests.append(tr)       
    allparams = [[par]*len(multiclass_scorenames) for par in harvest3['params']]
    allparams =  [item for sublist in allparams for item in sublist]
    copypaste = pd.DataFrame({
            'classifiers': sorted(list(set(clfs))*len(multiclass_scorenames))
            ,'train': trains
            ,'test': tests
            ,'params': allparams        
            ,'metrics': multiclass_scorenames*(len(trains)//10)})
    return allparams, copypaste 


def best_result_gridsearch_all_models(search): 
    search.cv_results_['params'][search.best_index_]       
    search.cv_results_.keys()
    
    search.cv_results_['mean_test_balanced_accuracy'][search.best_index_]
    search.cv_results_['mean_train_balanced_accuracy'][search.best_index_]
    search.cv_results_['mean_test_f1_micro'][search.best_index_]
    search.cv_results_['mean_train_f1_micro'][search.best_index_]
    print("Best parameter (CV score=%0.3f) test:" % search.best_score_)
    print("Best parameter :", search.best_estimator_)
    print("Best parameter :", search.best_params_)
    print("Best parameter :", search.best_index_)
    print("Best parameters set:")
    best_parameters = search.best_estimator_.get_params()
    for param_name in sorted(param_grid.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
    
    
    results = pd.DataFrame({
            'train':multiclass_train
            , 'test': multiclass_test
            , 'metric': multiclass_scorenames})
    return results


def plot_best_clf_params():      
    """ plot parameters """
    plt.axvline(search.best_estimator_.named_steps['clf'].alpha,
                linestyle=':', label='n_neighbors chosen')
    plt.legend(prop=dict(size=12))
    plt.show()
    
    
def plot_accuracy_error_parameter(param_varying, search): 
    params = search.cv_results_['params']
    print("All parameters: ", params)
    params_vary = []
    for i in range(len(params)): 
        for key, value in params[i].items(): 
            # to specify 
            if key == param_varying: 
                params_vary.append(value) 
    mean_acc_test_score = search.cv_results_['mean_test_score']
    print("All scores: ", mean_acc_test_score)
    allscores = list(mean_acc_test_score)
    allscores_error = [1-i for i in allscores]
    dica_train = list(zip(params_vary, allscores))
    dica_train_error = list(zip(params_vary, allscores_error))
    # plot accuracy score by parameter (varying)
    plt.plot(params_vary, allscores, '--bo')
    plt.title(f"5-fold Cross-Validation accuracy scores on training set")
    plt.ylabel('Accuracy score')
    plt.xlabel(param_varying)
    plt.show()
    
    # plot error accuracy score by parameter (varying)
    plt.plot(params_vary, allscores_error, '--bo')
    plt.title(f"5-fold Cross-Validation error scores on training set")
    plt.ylabel('Error score')
    plt.xlabel(param_varying)
    plt.show()
    return params_vary, dica_train, dica_train_error
    


#%%    
if __name__ == '__main__':
    
    # declare variables
    PARAM_VARYING = 'knn__leaf_size'
    
    # set directory 
    os.chdir(PATH)
    
    # load data 
    df = pd.read_excel(FILE, sep=" ", index_col=0)
    df.info()
 
    # analyze data   
    analyze_dataset(df, LABEL)   
    # preprocessing
    X, y = get_X_y(df, FEATURE, LABEL)
    # processing gridsearch
    gridsearch = grid_search_pipeline(X, y)
    # best results of gridsearch
    multiclass_train, multiclass_test, multiclass_scorers, multiclass_scorenames = best_result_gridsearch(gridsearch)
    clfs, params, metrics_train, metrics_test = best_result_gridsearch_per_model(gridsearch)
    multiclass_train, multiclass_test, keys_train, keys_test, harvest, harvest2 = best_result_gridsearch_per_metric(gridsearch, clfs, params, metrics_train, metrics_test)
    allparams, copypaste  = gridsearch_all_params(harvest, harvest2)
    results = best_result_gridsearch_all_models(gridsearch)
    
    # plot parameter by metric 
    params_vary, dica_train, dica_train_error = plot_accuracy_error_parameter(PARAM_VARYING, gridsearch)


