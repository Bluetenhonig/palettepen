# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 08:05:47 2020

@author: Anonym
"""
import os
import pandas as pd
import numpy as np
import pickle 
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, LabelBinarizer
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
from sklearn import neighbors, datasets

#%%

# declare variables 
PATH = r'D:\thesis\input_color_name_dictionaries\thesaurus\datasets'
FILE = 'eeffcnd_thesaurus_basicitten_upinterval.xlsx'
SOURCE = 'THESAURUS'
SYSTEM = 'ITTEN'
METHOD = 'INTERVAL' 

FILE2 = 'eeffcnd_thesaurus_basicvian_upinterval.xlsx'
SOURCE2 = 'THESAURUS'
SYSTEM2 = 'VIAN'
METHOD2 = 'INTERVAL' 

FEATURE = 'cielab'
LABEL =  'cat1'

COLOR_CATEGORY = 'amber'


MODEL_NAMES = ["GaussianNB", "QDA" ]
MODELS = [ GaussianNB(), QuadraticDiscriminantAnalysis() ]

MODEL_PATH = r'D:\thesis\machine_learning\models'
MODEL_FILE = f'model_THESAURUS_ITTEN_INTERVAL_KNeighborsClassifier_KNN23_p2_train1035_cat6_testacc0.87.sav'


#SAVE_PATH =

#%%

def load_model(path, file): 
    os.chdir(path)
    clf = pickle.load(open(file, 'rb'))
    return clf


def onevsrest_label(df, label, color_category): 
    """ helper function for preprocessing: copy label column and
    make a binary color category for a given color category """
    df[color_category] = df[label]
    df[color_category][df[color_category] == color_category] 
    df[color_category][df[color_category] != color_category] = f'non-{color_category}' 
    df[color_category] = df[color_category].replace(df[color_category][df[color_category] != color_category].tolist(), f'non-{color_category}' )
    lab2pt = df[color_category].tolist() 
    return lab2pt 


def get_X_y(df, lab_column, label, color_category=None): 
    """normalization / standardization
    get (X, y), original CS: LAB 
    input = X = [[0, 0], [1, 1]]; y = [0, 1]"""
    # X = lab color values 
    lab2pts = df[lab_column] 
    lab2pts = [eval(l) for l in lab2pts]
    # y = label encoding 
    if color_category: 
        lab2pt = onevsrest_label(df, label, color_category)
    else: 
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
    return X, y, le 

def test_train_split(X, y, test_size): 
    """ splits data set into test and training set"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    print(f"You have {len(X_train)} training colors and {len(X_test)} test colors - test_size: {test_size*100}.")
    print('Number of classes in y_test: ', len(set(y_test)))
    return X_train, X_test, y_train, y_test  


def plot_classifiers_boundary(X, y, models, model_names, meshsize=.02): 
    datasets = [(X, y)]
    figure = plt.figure(figsize=(9, 3))
    
    i = 1
    # iterate over datasets
    for ds_cnt, ds in enumerate(datasets):
        X, y = ds
        X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=.1, random_state=42)
    
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, meshsize),
                             np.arange(y_min, y_max, meshsize))
    
        cm = plt.cm.RdBu
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])
        ax = plt.subplot(len(datasets), len(MODELS) + 1, i)
        if ds_cnt == 0:
            ax.set_title("Input data")
        # Plot the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                   edgecolors='k')
        # Plot the testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,
                   edgecolors='k')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        ax.legend()
        i += 1
    
        # iterate over classifiers
        for name, clf in zip(model_names, models):
            ax = plt.subplot(len(datasets), len(models) + 1, i)
            clf.fit(X_train, y_train)
            if hasattr(clf, "decision_function"):
                Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
            else:
                Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    
            Z = Z.reshape(xx.shape)
            ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)
            ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                       edgecolors='k')
            ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                       edgecolors='k', alpha=0.6)
    
            ax.set_xlim(xx.min(), xx.max())
            ax.set_ylim(yy.min(), yy.max())
            ax.set_xticks(())
            ax.set_yticks(())
            if ds_cnt == 0:
                ax.set_title(name)
            ax.legend()
            i += 1
    
    
    plt.tight_layout()
    plt.show()



def plot_best_classifier_boundary(clf, X, y, le, system, meshsize=.02): 
    
    #list(le.classes_) 
    #labels = np.unique(y_train)
    colors = ['xkcd:'+ l for l in list(le.classes_) ]
    color_map1 = ['xkcd:'+ l for l in list(le.classes_) ]
                  
    # Create color maps
    cmap_light = ListedColormap(colors)
    cmap_bold = ListedColormap(color_map1)
    
    clf.fit(X, y)
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, meshsize),
                         np.arange(y_min, y_max, meshsize))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # Areas as color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    # Plot also the training points facecolors (c = labels)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
                edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(f"EEFFCND Thesaurus-{system} {len(set(y))}-class best {clf.__class__.__name__} classification")
    plt.ylabel('t-SNE component 1')
    plt.xlabel('t-SNE component 2')
    plt.legend()
    plt.show()



def plot_best_model_data_confusion_matrix(model_path, best_model, x_test, y_test): 
    os.chdir(model_path)
    clf = pickle.load(open(best_model, 'rb'))
    y_pred = clf.predict(X_test) 
    y_true = y_test
    cm = confusion_matrix(y_true, y_pred) 
    print(cm)
    return cm

def get_multilabel_confusion_matrix(y_true, y_pred): 
    cm = multilabel_confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm[0].flatten()
    return (tn, fp, fn, tp)


def plot_one_model_varparameter_accuracies(clf, param_name, param_vals, acc_scores_train, acc_scores_test): 
    plt.plot(param_vals, acc_scores_train, '--bo', label = "training")
    plt.plot(param_vals, acc_scores_test, '--ro', label = "test")
    plt.title(f"Accuracy scores for {clf.__class__.__name__} on training and test set") 
    plt.ylabel('Accuracy score')
    plt.xlabel(param_name)
    plt.legend()
    plt.show()

def plot_one_model_varparameter_errors(clf, param_name, param_vals, acc_errors_train, acc_errors_test): 
    plt.plot(param_vals, acc_errors_train, '--bo', label = "training")
    plt.plot(param_vals, acc_errors_test, '--ro', label = "test")
    plt.title(f"Error scores for {clf.__class__.__name__} on training and test set")
    plt.ylabel('Error score')
    plt.xlabel(param_name)
    plt.legend()
    plt.show()
    
#%%    
if __name__ == '__main__':
   
    #declare variable
    TEST_SIZE = .1 
    
    
    # load data 
    os.chdir(PATH)
    df = pd.read_excel(FILE, sep=" ", index_col=[0])
    df.info()
    df.head()

    #preprocessing: encode
    X, y = get_X_y(df, FEATURE, LABEL, COLOR_CATEGORY)
    # transform: dimensionality reduction 3-D to 2-D
    X = TSNE(n_components=2).fit_transform(X)
    # plot classifiers decision boundary 
    print(MODEL_NAMES, 'boundaries for', len(set(y)), f'{SYSTEM} colors.' )
    plot_classifiers_boundary(X, y, MODELS, MODEL_NAMES)



    # load data 
    os.chdir(PATH)
    df = pd.read_excel(FILE, sep=" ", index_col=[0])
    df.info()
    df.head()

    #preprocessing: encode
    X, y, le = get_X_y(df, FEATURE, LABEL)
     # scorer
    X_train, X_test, y_train, y_test  = test_train_split(X, y, TEST_SIZE)   
    # plot confusion matrix
    plot_best_model_data_confusion_matrix(MODEL_PATH, MODEL_FILE, X_test, y_test) 
    # transform: dimensionality reduction 3-D to 2-D
    X = TSNE(n_components=2).fit_transform(X)
    # scorer
    X_train, X_test, y_train, y_test  = test_train_split(X, y, TEST_SIZE)
    # load classifier
    clf = load_model(MODEL_PATH, MODEL_FILE) 
    # plot classifiers decision boundary 
    print(clf.__class__.__name__, 'boundaries for', len(set(y)), f'{SYSTEM} colors.' )
    plot_best_classifier_boundary(clf, X, y, le, SYSTEM)
  





