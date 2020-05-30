# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 11:08:31 2020

@author: Anonym
"""
#TODO: ties 

### Get answers: specify number of objects

print('-----------------------')
print('Welcome to PRICA - the PRIORITY CALCULATOR. \nThe priority calculator helps in the decision-making process by using pairwise comparisons to rank objects of your choosing. \nFollow these steps to determine your best choice of objects.')
print('-----------------------')

print('Write down all unique objects you want to compare.')
objects_len = int(input("How many objects would you like to compare?"))

print('-----------------------')
print("Write down all objects you want to compare.")


objects = []
for object in range(objects_len): 
    object = input(f"{object+1}. Object: ")
    objects.append(object)


#%%
### Get answers: compare pairwise objects  

import itertools
object_comb = list(itertools.combinations(objects, 2))

object_comb_str = []
for i in object_comb: 
    el1 = i[0]
    el2 = ' or '
    el3 = i[1]
    string = el1 + el2 + el3
    object_comb_str.append(string)

answers = []
for i in object_comb_str: 
    print("")
    print("What is your preferred choice?")
    answer = input(f"{i}: ")
    answers.append(answer)

#%%

### Processing: assign scores to objects 

def score4pair(question, answer):
    el1 = question[0]
    el2 = question[1]
    if not scores: #empty dict
        #print('no vals in scores')
        if answer == el1: 
            scores[el1] = 1
            scores[el2] = 2
        else: 
            scores[el1] = 2
            scores[el2] = 1
    else: # append vals to dict
        #print('scores has values')
        if answer == el1: 
            scores[el1] += 1
            try: 
                scores[el2] += 2
            except: 
                scores[el2] = 2
        else: 
            scores[el2] += 1
            try: 
                scores[el1] += 2
            except: 
                scores[el1] = 2
            
    return scores

scores = dict()      
for i in range(objects_len):
    if i == 0: 
        assert scores == {}
    score4pair(object_comb[i], answers[i])

print(scores)


#%%

### Post results: rank objects 

# get key with minimum score  
winner = min(scores, key=scores.get)

print('-----------------------')
print('Your most preferred choice: ', winner)
print('-----------------------')

rank = {key: rank for rank, key in enumerate(sorted(scores, key=scores.get, reverse=False), 1)}


for x, y in rank.items():
    #print ('Rank', y, ':', x)
    print (y, '-', x)


#%%

### Save results: save objects to dataframe
import os
import pandas as pd 
  
save = input("Do you want to save your results? (yes/no)")  
if save == 'yes' or save == 'y': 
    topic = input("What topic did you choose? ") 
    
    df = pd.DataFrame.from_dict(rank, orient='index',columns=['rank'])
    df[topic] = df.index
    df.reset_index(level=0, inplace=True)
    df = df.drop(columns=['index'])
    print(df)
    path = input('Path to where you want to save it:')
    os.chdir(path)
    df.to_csv(f'prica_TXT_{topic}.csv', index=False) 
    print('-----------------------')
    print('>>> Your results were successfully saved to your computer. ')
    
else: 
    pass 

print('-----------------------')
print('Thank you for using PRICA the priority calculator. \nSee you next time!')
print('-----------------------')