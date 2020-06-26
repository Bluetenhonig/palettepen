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

#####################################
### Load Data 
 
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

#%%

# load EFFCND 

### Color-Thesaurus EPFL ###

# to specify 
# original EFFCND - source 
SOURCE = 'THESAURUS' 
PATH = r'D:\thesis\input_color_name_dictionaries\thesaurus\datasets'
FILE = 'effcnd_thesaurus_itten.xlsx'
OUTPUT_FILE = 'effcnd_thesaurus_itten.xlsx' 
# basic color system 
SYSTEM = 'ITTEN'
#METHOD = 'INTERVAL' 

# to specify EFFCND+
#if SYSTEM == 'VIAN_+': 
#    df_addtstcols = pd.read_excel(f'SRGBLABhsvhslLCHHEX_EngALL_testcolsadded1331_VIANHuesColorThesaurus.xlsx')
#    df = df_addtstcols
      
# set directory 
os.chdir(PATH)

# load data 
df = pd.read_excel(FILE, sep=" ", index_col=[0])
df.info()

#%%
# preprocessing 

# eda
LABEL =  'cat' 
print(f"Basic colors' distribution {SYSTEM} in the color name dictionary {SOURCE}: ")
valcts = df[LABEL].value_counts()
print(valcts)
valctsplt = df[LABEL].value_counts().plot(kind='bar')
print(valctsplt)



#%%
# split data set

# get (X, y)

# original CS: LAB 
# build X - X is the input features by row. X = [[0, 0], [1, 1]]
# array([[ 1.02956195e+00,  1.12384202e+00,  1.28943006e+00], []])
# rescale cielab val range to  
lab2pts = df['cielab'] #lab column 
lab2pts = [eval(l) for l in lab2pts]
#X_scaled = preprocessing.scale(X_train)
## scaled data has mean 0 and unit variance 
#X_scaled.mean(axis=0)
#X_scaled.std(axis=0)

# Y is the class labels for each row of X. y = [0, 1]
# for multi-class classification 
# labelencoding 
lab2pt = df[LABEL].tolist() #list(df.index)
le = preprocessing.LabelEncoder()
le.fit(lab2pt) # fit all cat1 colors 
list(le.classes_) # get set of cat1 colors  
lab2dt = le.transform(lab2pt) # transform all to numeric 
list(le.inverse_transform(lab2dt)) # get back all cat1 as string 

## for multi-label classification
#y = np.array([[1, 0, 0, 1], [0, 0, 1, 1], [0, 0, 0, 0]])
#print(y)
## multioutput classification
#y = np.array([['apple', 'green'], ['orange', 'orange'], ['pear', 'green']])
#print(y)

# X and y 
X = np.array(lab2pts)
y = np.array(lab2dt)

print(f"You have {len(X)} colors in color name dictionary {SOURCE} for {df[LABEL].nunique()} color categories in LAB.")

# get training and test set 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

print(f"You have {len(X_train)} training colors and {len(X_test)} test colors.")

# assert
#X_trainpts, X_testpts, y_traindt, y_testdt = train_test_split(lab2pts, lab2pt, test_size=0.1, random_state=42)
#pd.Series(y_traindt).value_counts()
#len(pd.Series(y_traindt).unique())
#pd.Series(y_testdt).value_counts()
#len(pd.Series(y_testdt).unique())

#%%

#####################################
### Build Model

# Define a pipeline to search for the best combination 
models = [
        "K-Nearest Neighbors"
         , "Linear SVM"]

# to specify
m= 0 # 0 for KNN, 1 for SVC 

# TODO: exhaust all parameters 
# pipeline with baseline models 
pipe = Pipeline(steps=[   # transformer: fit + transform
                    ('scaler', StandardScaler())
                   ,('knn', KNeighborsClassifier()) # estimator: fit + predict
              #  , ('svc', SVC(random_state=0, probability = True))
                ])  # estimator: fit + predict

# model parameters
cs = np.linspace(0.01,1,11)**2
# Parameters of pipelines can be set using ‘__’ separated parameter names:
param_grid = {
    'knn__n_neighbors': list(range(1,200)),
    'knn__p': [2],   #Power parameter for the Minkowski metric: p=2 is Euclidean, p=1 is Manhattan
    #'svc__kernel': ['linear'],
    #'svc__C': cs.tolist() #based on https://scikit-learn.org/stable/modules/svm.html, param C trades off misclassification of training examples against simplicity of the decision surface. A low C makes the decision surface smooth, while a high C aims at classifying all training examples correctly, exponentially spaced to get good values
}

# grid search for hyperparameter tuning 
# GridSearchCV will take the data you give it, split it into Train and CV set and train algorithm searching for the best hyperparameters using the CV set. You can specify different split strategies if you want (for example proportion of split).
# But when you perform hyperparameter tuning information about dataset still 'leaks' into the algorithm.
# 1) Take your original dataset and hold out some data as a test set (say, 10%)
# 2) Use grid search on remaining 90%. Split will be done for you by the algorithm here.
# 3) After you got optimal hyperparameters, test it on the test set from #1 to get final estimate of the performance you can expect on new data.
search = GridSearchCV(pipe, param_grid, scoring='accuracy', n_jobs=-1) # default cv=5
search.fit(X_train, y_train)
  
print(f'Training Machine Learning Classifier {models[m]} for {SYSTEM} Color Categories: successful!')

#%%
# evaluating training set

# best result 
print("Best parameter (CV score=%0.3f):" % search.best_score_)
print("Best parameter :", search.best_params_)

# to specify
param_varying = 'knn__n_neighbors'
# all results
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
plt.title(f"5-fold Cross-Validation accuracy scores for {models[m]} on training set")
plt.ylabel('Accuracy score')
plt.xlabel(param_varying)
plt.show()

# plot error accuracy score by parameter (varying)
plt.plot(params_vary, allscores_error, '--bo')
plt.title(f"5-fold Cross-Validation error scores for {models[m]} on training set")
plt.ylabel('Error score')
plt.xlabel(param_varying)
plt.show()

#%%
# evaluate test set 

allscores_test = []
for i in params_vary: 
    if param_varying == 'knn__n_neighbors':
        KNN = i
        p = 2
        # train best model 
        clf = KNeighborsClassifier(n_neighbors=KNN, p=p)
    elif param_varying == 'svc__C':
        C = i
        kernel = 'linear'
        # train best model 
        clf = SVC(C=C, kernel=kernel)
    # train on training set  
    clf.fit(X_train, y_train)
    # evaluate on test set
    y_pred = clf.predict(X_test)
    
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(y_test, y_pred)
    allscores_test.append(acc)
allscores_test_errors = [1-i for i in allscores_test]

# best parameters and score 
print("Best parameter score: ", round(max(allscores_test),3)) 
# 0.878
#0.876
n_neighbours = []
dica_test = list(zip(params_vary, allscores_test))
dica_test_error = list(zip(params_vary, allscores_test_errors))
for dic in dica_test: 
    if dic[1] == max(allscores_test):
        n_neighbours.append(dic[0])
print("Best parameter: ", n_neighbours[0]) 
# 1  

# plot accuracy score by parameter (varying)
plt.plot(params_vary, allscores_test, '--ro')
plt.title(f"Accuracy scores for {models[m]} on test set")
plt.ylabel('Accuracy score')
plt.xlabel(param_varying)
plt.show()

# plot error accuracy score by parameter (varying)
plt.plot(params_vary, allscores_test_errors, '--ro')
plt.title(f"Error scores for {models[m]} on test set")
plt.ylabel('Error score')
plt.xlabel(param_varying)
plt.show()


#%%
# plot training and test set 
# accuracy score 
plt.plot(params_vary, allscores, '--bo', label = "training")
plt.plot(params_vary, allscores_test, '--ro', label = "test")
plt.title(f"Accuracy scores for {models[m]} on training and test set") #90-10 split
plt.ylabel('Accuracy score')
plt.xlabel(param_varying)
# show a legend on the plot
plt.legend()
plt.show()

# error score 
plt.plot(params_vary, allscores_error, '--bo', label = "training")
plt.plot(params_vary, allscores_test_errors, '--ro', label = "test")
plt.title(f"Error scores for {models[m]} on training and test set") #90-10 split
plt.ylabel('Error score')
plt.xlabel(param_varying)
# show a legend on the plot
plt.legend()
plt.show()

#%%
# best model for KNN 

# where the test error accuracy score is lowest 
min_test_erroracc = min(list(zip(*dica_test_error))[1])
print("Lowest test error (accuracy score): ", round(min_test_erroracc,3))
knn_min_test_erroracc = [i[0] for i in dica_test_error if i[1] == min_test_erroracc][0]
print(f"Lowest test error if {param_varying} = ", knn_min_test_erroracc)
KNN = knn_min_test_erroracc
p = 2
train_score = round([i[1] for i in dica_train if i[0] == KNN][0],3)
test_score = round([i[1] for i in dica_test if i[0] == KNN][0],3)
print(f"For {models[0]} with KNN = {KNN}: training score {train_score} and test score {test_score}")

# train best model 
clf = KNeighborsClassifier(n_neighbors=KNN, p=p)
# train on training set  
clf.fit(X_train, y_train)
#%%
# best model for SVC
# where the test error accuracy score is lowest 
min_test_erroracc = min(list(zip(*dica_test_error))[1])
print("Lowest test error (accuracy score): ", round(min_test_erroracc,3))
C_min_test_erroracc = [i[0] for i in dica_test_error if i[1] == min_test_erroracc][0]
print(f"Lowest test error if {param_varying} = ", C_min_test_erroracc)
C = C_min_test_erroracc
KERNEL = 'linear'
train_score = round([i[1] for i in dica_train if i[0] == C][0],3)
test_score = round([i[1] for i in dica_test if i[0] == C][0],3)
print(f"For {models[1]} with C = {C}: training score {train_score} and test score {test_score}")
# train best model 
clf = SVC(random_state=0, C=C, kernel=kernel, probability=True)
# train on training set  
clf.fit(X_train, y_train)

#%%
# save model
import pickle

# set directory 
MODEL_PATH = r'D:\thesis\machine_learning\models'
os.chdir(MODEL_PATH)

#%%
# KNN
# save the model to disk
filename = f'model_{SOURCE}_{SYSTEM}_{models[0]}_KNN{KNN}_p{p}_train{len(X)}_cat{df[LABEL].nunique()}_testacc{test_score}.sav'
filename = f'model_{SOURCE}_{METHOD}_{models[0]}_KNN{KNN}_p{p}_train{len(X)}_cat{df[LABEL].nunique()}_testacc{test_score}.sav'
# model name: classifier, classifier parameters, training data, label classes 
pickle.dump(clf, open(filename, 'wb'))
#%%
# SVC 
filename = f'model_{SOURCE}_{SYSTEM}_{models[1]}_SVC{KERNEL}_C{C}_train{len(X)}_cat{df[LABEL].nunique()}_testacc{test_score}.sav'
filename = f'model_{SOURCE}_{METHOD}_{models[1]}_SVC{KERNEL}_C{C}_train{len(X)}_cat{df[LABEL].nunique()}_testacc{test_score}.sav'
# model name: classifier, classifier parameters, training data, label classes 
pickle.dump(clf, open(filename, 'wb'))

#%%
#####################################
### Model Analysis 

### SVC ###
## get support vectors
#models[1].support_vectors_
## get indices of support vectors
#models[1].support_
## get number of support vectors for each class
#models[1].n_support_
#
## decision function 
#decision_function = models[1].decision_function(X)
#decision_function.shape
## Distance of the samples X to the separating hyperplane.
## we can also calculate the decision function manually
## decision_function = np.dot(X, clf.coef_[0]) + clf.intercept_[0]
    
#%%

#####################################
### Use Model 

# get test colors for classification

#TODO: test points should be generated in RGB, 
#but the classifier is trained in LAB
# that's why it can only take test colors in LAB, 
#should i generate RGB colors and convert them to LAB 
#only to convert them back to RGB for visualization 
 

# TODO: test colors in RGB only 
TESTPOINTS = False
ONEPOINT = True
STEP = 6



def testpoints(step=STEP):
    x = np.linspace(0, 255, step)
    y = np.linspace(0, 255, step)
    z = np.linspace(0, 255, step)
    xyz = np.vstack(np.meshgrid(x, y, z)).reshape(3,-1).T.tolist()
    return xyz


# lab space is uniform which is why if we make voronoi diagram, the euclidean distance is enough for classification
if TESTPOINTS: 
    # to specify  a list of LAB colors
    test_colors_lab = testpoints()
    testpoints_count = len(test_colors_lab)
    print("You have {} test colors in RGB.".format(testpoints_count))
    

#%%
if ONEPOINT: 
    # to specify colors by picking a color in the colorpicker
    test_colors_rgb = []
    
    #%%
    # color picker: choose rgb color
    # rerun this cell to have a list of rgb colors
    import tkinter as tk
    import tkinter.ttk as ttk
    from tkcolorpicker import askcolor
    
    root = tk.Tk()
    style = ttk.Style(root)
    style.theme_use('clam')
    
    askcolor = askcolor((255, 255, 0), root)
    root.mainloop()
    
    rgb = askcolor[0]
    r, g, b = rgb
    rgb = r/255, g/255, b/255
    test_colors_rgb.append(rgb)
    
    #%%
    test_colors_lab = []
    
    for rgb in test_colors_rgb: 
        lab = convert_color(rgb, conversion=cv2.COLOR_RGB2Lab)
        test_colors_lab.append(lab.tolist())

#%%
        
# load model 
# to specify
ML_MODELS_PATH = r'D:\thesis\machine_learning\models'
ML_MODELS_FILE = f'model_THESAURUS_INTERVAL_K-Nearest Neighbors_KNN1_p2_train4847_cat712_testacc0.878.sav'

# load the model from disk
import os
import pickle
os.chdir(ML_MODELS_PATH)
clf = pickle.load(open(ML_MODELS_FILE, 'rb'))

        
#%%
### Use Model to classify test colors into color categories
        
import os 
     
# to specify 
idx = 0
PATH = r'D:\thesis\machine_learning\predictions'
MODEL = models[idx] # trained clfs saved in models 

# load model 
if idx == 0: 
    FOLDER =  f'{models[idx]}_{KNN}_LAB_{SYSTEM}{len(df)}_test{STEP}_labels{df[LABEL].nunique()}_ColorClassification'
elif idx == 1: 
    FOLDER =  f'{models[idx]}_{KERNEL}{C}_LAB_{SYSTEM}{len(df)}_test{STEP}_labels{df[LABEL].nunique()}_ColorClassification'
   
#%%
    
# test classifier 
def categorize_color(color_lab, clf): 
    label = clf.predict([color_lab]) #lab: why? The CIE L*a*b* color space is used for computation, since it fits human perception
    label = le.inverse_transform(label)
    if len(label) == 1: 
        label = label.tolist()[0] 
    else:   
        label = df[LABEL][df[LABEL] == label].iloc[0]
        
    #print('Label: ', label) 
    return label 

# one color
# get classification for 'sunflower yellow' 
sunflower_yellow= eval(df['cielab'].iloc[654])
# get color category 
pred_label = categorize_color(sunflower_yellow, clf)
# get color categories prob distr 
results = clf.predict_proba([sunflower_yellow])[0]
classes = le.inverse_transform(clf.classes_).tolist()
# gets a dictionary of {'class_name': probability}
prob_per_class_dictionary = dict(zip(classes, results))
# gets a list of ['most_probable_class', 'second_most_probable_class', ..., 'least_class']
results_ordered_by_probability = list(map(lambda x: x, sorted(zip(classes, results), key=lambda x: x[1], reverse=True)))
classes_ordered = [i[0] for i in results_ordered_by_probability]
probs_ordered = [i[1] for i in results_ordered_by_probability]
y_pos = np.arange(len(classes_ordered))

# plot proba distr of sunflower yellow over basic colors 
fig, ax = plt.subplots()
# Example data
ax.barh(y_pos, np.array(probs_ordered), align='center')
ax.set_yticks(y_pos)
ax.set_yticklabels(classes_ordered)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Probability')
ax.set_title('"Sunflower yellow": Probability distribution for VIAN color categories')

plt.show()

#%%
# plot test colors with its predicted label 
plt.suptitle(f'LAB-Colors Categorized into {df[LABEL].nunique()} Colors' )
y_pred = []
truefalse = []
for i, color in enumerate(test_colors_lab): 
    fig = plt.figure(figsize = (10,5))  
    label = categorize_color(color, MODEL)
    y_pred.append(label)
    rgb = convert_color(color, cv2.COLOR_LAB2RGB)
    r, g, b = rgb
    lablabel = eval(df['cielab'][df[LABEL] == label].iloc[0])
    rgblabel = eval(df['srgb'][df[LABEL] == label].iloc[0])

    # convert string to list 
    #rgblabel = ini_list.strip('][').split(', ')
    rgb = round(r*255), round(g*255), round(b*255)
    square1 = np.full((10, 10, 3), rgb, dtype=np.uint8) / 255.0
    square2 = np.full((10, 10, 3), rgblabel, dtype=np.uint8)
    # create plot 
    #plt.subplot(len(test_colors_lab), 1, i+1)
    plt.subplot(1, 2, 1)
    plt.imshow(square1) 
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(square2) 
    plt.axis('off')
    plt.suptitle(f'Left (test color) classified as Right (label): {label}')
    # save plot 
    os.chdir(PATH)
    try: 
        os.mkdir(FOLDER)
        os.chdir(os.path.join(PATH,FOLDER))
    except: 
        os.chdir(os.path.join(PATH,FOLDER))
    fig.savefig(f'{i}patch2predlabel.png', dpi=fig.dpi)
    plt.show()
    print('index:', i)
    print('label: ', label)
    print('test color lab: ', color)
    print('test color rgb: ', rgb)
    print('label lab: ', lablabel)
    print('label rgb: ', rgblabel)

    goodness=input('Is the classification correct (1) or not (0)?')
    truefalse.append(goodness)
    
# accuracy 
accuracy = truefalse.count('1')/len(truefalse)*100
error = truefalse.count('0')/len(truefalse)*100

print("The accuracy is {}%".format(accuracy))
print("The error is {}%".format(error))

# save to file
os.chdir(os.path.join(PATH,FOLDER))

file1 = open("classification_accuracy.txt","w") 
file1.write(f"Accuracy: {accuracy} \n") 
file1.write(f"Error: {error}") 
file1.close()  

#%%


# hand-label test colors (lab) into VIAN categories 

# to specify 
idx = 0
PATH = r'D:\thesis\code\pd28vianhues'
MODEL = models[idx] # trained clfs saved in models 
    
     
# test classifier 
def categorize_color(color_lab, clf): 
    label = clf.predict([color_lab]) #lab: why? The CIE L*a*b* color space is used for computation, since it fits human perception
    label = label.tolist()[0] 
    label = df[LABEL][df[LABEL] == label].iloc[0]
        
    #print('Label: ', label) 
    return label 

#plot 
vian_colors = []
plt.suptitle(f'LAB-Colors Categorized into {df[LABEL].nunique()} Colors' )
for i, color in enumerate(test_colors_lab): 
    fig = plt.figure(figsize = (10,5))  
    label = categorize_color(color, MODEL)
    rgb = convert_color(color, cv2.COLOR_LAB2RGB)
    r, g, b = rgb
    lablabel = eval(df['cielab'][df[LABEL] == label].iloc[0])
    rgblabel = eval(df['srgb'][df[LABEL] == label].iloc[0])

    # convert string to list 
    #rgblabel = ini_list.strip('][').split(', ')
    rgb = round(r*255), round(g*255), round(b*255)
    square1 = np.full((10, 10, 3), rgb, dtype=np.uint8) / 255.0
    square2 = np.full((10, 10, 3), rgblabel, dtype=np.uint8)
    # create plot 
    #plt.subplot(len(test_colors_lab), 1, i+1)
    plt.subplot(1, 2, 1)
    plt.imshow(square1) 
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(square2) 
    plt.axis('off')
    plt.suptitle(f'Left (test color) classified as Right (label): {label}')

    plt.show()
    print('index:', i)
    print('label: ', label)
    print('test color lab: ', color)
    print('test color rgb: ', rgb)
    print('label lab: ', lablabel)
    print('label rgb: ', rgblabel)

    vian_cat=input('Which VIAN color category does this test color have?')
    vian_colors.append(vian_cat)
    
# save to file
os.chdir(PATH)
foo = pd.DataFrame({'cielab':test_colors_lab
                    ,'VIAN_color_category': vian_colors
                    })

foo.to_csv(f"lab_vian_colors_testcolors{testpoints_count}.csv", index=False)


#%%

# add test colors with predicted label to data set

import pandas as pd
testpoints_count = 1331

# load test colors 
os.chdir(PATH)
foo = pd.read_excel(f"lab_vian_colors_testcolors{testpoints_count}.xlsx")
test_colors_lab = foo['cielab'].tolist()
vian_colors = foo['VIAN_color_category'].tolist()

# load start data set  
os.chdir(PATH) 
df = pd.read_excel('SRGBLABhsvhslLCHHEX_EngALL_VIANHuesColorThesaurus.xlsx', index_col=[0])   

df2 = {} 
for key in df.keys():
    if key == "cielab": 
        df2["cielab"] = [l for l in test_colors_lab]
    elif key == "VIAN_color_category": 
        df2[key] = vian_colors
    else: 
        df2[key] = None 
 
df2 = pd.DataFrame(df2)  
df_addtstcols = df.append(df2)

df_addtstcols["cielab"] = [str(l) for l in df_addtstcols["cielab"]]

# save enlarged data set to file
#df_addtstcols.to_excel(f'SRGBLABhsvhslLCHHEX_EngALL_testcolsadded{testpoints_count}_VIANHuesColorThesaurus.xlsx', index = False)



#%%

###################################
### Plot Model as ScatterMatrix ###
###################################

# build model for visualization

# 2D for one classifier only 

from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets

# greenred - blueyellow

# to specify: select 2 features / variable for the 2D plot 
# a, b in L*ab
X = np.array(lab2pts)[:, 1:]  # we only take two features.
x_label = 'a - green/red'
y_label = 'b - blue/yellow'
y = np.array(list(df.index))
LABEL = 'VIAN_color_category'
n = df[LABEL].tolist() #label data points in plot
facecolors = df['HEX'].tolist()
facecolors.append('#8000ff')

print(f"Visualizing decision boundaries for {df[LABEL].nunique()} colors.")
                  
def make_meshgrid(x, y, h=.02): # h = step size in the mesh
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

def make_gridpoints(x, y, h=5):
    x_min, x_max = x.min()+1 , x.max() 
    y_min, y_max = y.min()+1 , y.max() 
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

# to specify 
idx = 1

# define model 
model = svm.SVC(kernel="linear", C=0.025)
clf = model.fit(X,y)

# create figure
fig, ax = plt.subplots(figsize=(10,10))
# title for the plots
title = (f'L*ab-Decision Surface with {names[idx]}')
# Set-up grid for plotting.
X0, X1 = X[:, 0], X[:, 1]  
xx, yy = make_meshgrid(X0, X1)
aa, bb = make_gridpoints(X0, X1)

# regions bound by decision boundaries 
plot_contours(ax, clf, xx, yy, colors=facecolors, alpha=0.8) # = ax.contourf()
# 6 data points
ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
# gridpoints 
#ax.scatter(aa, bb, c=faces, s=20, edgecolors='k')
ax.set_ylabel(f'{y_label}')
ax.set_xlabel(f'{x_label}')
#ax.set_xticks(X[:, 0])
#ax.set_yticks(X[:, 1])
for i, txt in enumerate(n):
    ax.annotate(txt, (X0[i], X1[i]))
ax.set_title(title)
# save figure
os.chdir(os.path.join(PATH,FOLDER))
fig.savefig(f'lab_{SYSTEM}_{names[idx]}_decbound_ab.png')
plt.show()




#%%

### Get test colors 

# save dataframe for gridpoints coloring 'faces' in above scatter plot
ab = np.dstack((aa,bb))

lst = []
for i in ab: 
    for j in i: 
        lst.append(j.tolist()) 
    
df2 = pd.DataFrame(lst, columns = ['A','B']) 
df2['L'] = 60
columns = ['L', 'A', 'B']
df2 = df2[columns]

# save df to get HEX for LAB 
os.chdir(r'D:\thesis\code\pd4lab')
df2.to_csv('LAB_ABgridpoints.csv', index=False)
# load with HEX
df2 = pd.read_csv('LABHEX_ABgridpoints.csv')
faces = df2['HEX'].tolist()


#%%
# Luminance - greenred

# to specify: select 2 features / variable for the 2D plot 
# a, b in L*ab
X = np.array(lab2pts)[:, [1,0]]  # we only take two features.
x_label = 'a - green/red'
y_label = 'l - luminance' 
y = np.array(list(df.index))
n = df['name'].tolist() #label data points in plot
facecolors = df['HEX'].tolist()
facecolors.append('#8000ff')

def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

model = svm.SVC(kernel='linear')
clf = model.fit(X, y)

os.chdir(r'D:\thesis\code\6hues')
fig, ax = plt.subplots(figsize=(10,10))
# title for the plots
title = ('L*ab-Decision Surface with SVC (kernel=linear)')
# Set-up grid for plotting.
X0, X1 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid(X0, X1)

plot_contours(ax, clf, xx, yy, colors=facecolors, alpha=0.8)
ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
ax.set_ylabel(f'{y_label}')
ax.set_xlabel(f'{x_label}')
#ax.set_xticks(X[:, 0])
#ax.set_yticks(X[:, 1])
for i, txt in enumerate(n):
    ax.annotate(txt, (X0[i], X1[i]))
ax.set_title(title)
fig.savefig('lab_6basiccolors_SVC_decbound_la.png')
plt.show()





#%%
# luminance - blueyellow

# to specify: select 2 features / variable for the 2D plot 
# a, b in L*ab
X = np.array(lab2pts)[:, [2,0]]  # we only take two features.
x_label =  'b - blue/yellow'
y_label = 'l - luminance' 
y = np.array(list(df.index))
n = df['name'].tolist() #label data points in plot
facecolors = df['HEX'].tolist()
facecolors.append('#8000ff')

def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

model = svm.SVC(kernel='linear')
clf = model.fit(X, y)

fig, ax = plt.subplots(figsize=(10,10))

# title for the plots
title = ('L*ab-Decision Surface with SVC (kernel=linear)')
# Set-up grid for plotting.
X0, X1 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid(X0, X1)

plot_contours(ax, clf, xx, yy, colors=facecolors, alpha=0.8) #regions 
ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
ax.set_ylabel(f'{y_label}')
ax.set_xlabel(f'{x_label}')
#ax.set_xticks(X[:, 0])
#ax.set_yticks(X[:, 1])
for i, txt in enumerate(n):
    ax.annotate(txt, (X0[i], X1[i]))
ax.set_title(title)
fig.savefig('lab_6basiccolors_SVC_decbound_lb.png')
plt.show()


#%%
# 2D: try all classifiers (long exec time)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
import timeit

start = timeit.timeit()

# to specify: select 2 features / variable for the 2D plot 
# a, b in L*ab
X = np.array(lab2pts)[:, 1:]  # we only take two features.
x_label = 'a - green/red'
y_label = 'b - blue/yellow'
y = np.array(list(df.index))
n = df['name'].tolist() #label data points in plot
facecolors = df['HEX'].tolist()
facecolors.append('#8000ff')
        
def make_meshgrid(x, y, h=.02): # h = step size in the mesh
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

if SIX_COLORS: 
    os.chdir(r'D:\thesis\code\6hues')   
else: 
    os.chdir(r'D:\thesis\code\11hues')  
             
# plot the decision boundaries of each classifier                 
figure = plt.figure(figsize=(50, 5)) # figsize=(x,y)

# get color points
X0, X1 = X[:, 0], X[:, 1] 
# make meshgrid for Z contouring
xx, yy = make_meshgrid(X0, X1)

# plot the dataset 
ax = plt.subplot(1, len(classifiers) + 1, 1)

# plot the training points
ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_xticks(())
ax.set_yticks(())
ax.set_ylabel(f'{y_label}')
ax.set_xlabel(f'{x_label}')
for i, txt in enumerate(n):
    ax.annotate(txt, (X0[i], X1[i]),horizontalalignment='center', verticalalignment='top')
ax.set_title("Input data")


# iterate over classifiers
for i,(name, clf) in enumerate(zip(names, classifiers)):
    print('Name:', name)
    ax = plt.subplot(1, len(classifiers) + 1, i+2)
    clf.fit(X, y)

    # Plot the decision boundary. For that, we will assign a color to each    
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Put the result into a color plot
    ax.contourf(xx, yy, Z, colors=facecolors, alpha=.8)

    # Plot the training points
    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_ylabel(f'{y_label}')
    ax.set_xlabel(f'{x_label}')
    for i, txt in enumerate(n):
        ax.annotate(txt, (X0[i], X1[i]),horizontalalignment='center', verticalalignment='top')
    ax.set_title(name)


plt.tight_layout()
fig.suptitle('Decision boundaries of ML Classifiers for Basic Colors')
fig.savefig('lab_mlclfs_comparison.png')
plt.show()

end = timeit.timeit()
print(end - start)

# %%



# TODO in 3D: unsolvable problem bc of reshaping numpy arrays
# also what would the output look like? transparent regions, no solid decision hyperplanes 

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.svm import SVC


# to specify: select 3 features / variable for the 3D plot 
# L*ab
X = np.array(lab2pts)  
x_label = 'luminance'
y_label = 'a - green/red'
z_label = 'b - blue/yellow'
Y = np.array(list(df.index))



# Fit the data with an svm
svc = SVC(kernel='linear')
svc.fit(X,Y)

model = svm.SVC(kernel='linear')
clf = model.fit(X, Y)

# The equation of the separating plane is given by all x in R^3 such that:
def f(x,y):
    z  = (-svc.intercept_[0]-svc.coef_[0][0]*x-svc.coef_[0][1]*y) / svc.coef_[0][2]
    return z
def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contour3D(xx, yy, Z, **params)
    return out

# make meshgrid to shape in 3D 
SPACE_SAMPLING_POINTS = 100

# Define the size of the space which is interesting for the example
X_MIN = -5
X_MAX = 5
Y_MIN = -5
Y_MAX = 5
Z_MIN = -5
Z_MAX = 5

# Generate a regular grid to sample the 3D space for various operations later
xx, yy, zz = np.meshgrid(np.linspace(X_MIN, X_MAX, SPACE_SAMPLING_POINTS),
                         np.linspace(Y_MIN, Y_MAX, SPACE_SAMPLING_POINTS),
                         np.linspace(Z_MIN, Z_MAX, SPACE_SAMPLING_POINTS))



#X0, X1 = X[:, 0], X[:, 1]
#xx, yy = make_meshgrid(X0, X1)

# Plot stuff.
fig = plt.figure(figsize=(10,10))
ax  = fig.add_subplot(111, projection='3d')
ax.plot_surface(xx, yy, f(xx,yy))


# plot X-dots in 3D 
#ax.plot3D(X[Y==0,0], X[Y==0,1], X[Y==0,2],'o')
#ax.plot3D(X[Y==1,0], X[Y==1,1], X[Y==1,2],'v')
#ax.plot3D(X[Y==2,0], X[Y==2,1], X[Y==2,2],'H')    
#ax.plot3D(X[Y==3,0], X[Y==3,1], X[Y==3,2],'h') 
#ax.plot3D(X[Y==4,0], X[Y==4,1], X[Y==4,2],'D') 
#ax.plot3D(X[Y==5,0], X[Y==5,1], X[Y==5,2],'d')    

#ax.contour3D(X, Y, Z, 150, cmap='binary')
ax.set_xlabel(f'{x_label}')
ax.set_ylabel(f'{y_label}')
ax.set_zlabel(f'{z_label}')


plt.show()

#%%
# 3rd position in linspace equals number of views 
for angle in np.linspace(0, 360, 60).tolist():
    # Plot stuff.
    fig = plt.figure()
    ax  = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z(x,y))
    ax.plot3D(X[Y==0,0], X[Y==0,1], X[Y==0,2],'ob')
    ax.plot3D(X[Y==1,0], X[Y==1,1], X[Y==1,2],'sr')
    ax.view_init(60, angle)
    plt.draw()
    plt.pause(.001)
    plt.show()
    
