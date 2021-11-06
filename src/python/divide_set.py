# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 14:55:07 2021

@author: 92831
"""

#%%
import pandas as pd
import random

#%%
data = pd.read_csv('iris.csv', index_col = 0)
data.loc[data['Species']=='setosa', 'Species'] = 1
data.loc[data['Species']=='versicolor', 'Species'] = 2
data.loc[data['Species']=='virginica', 'Species'] = 3

def generate_set(data, train_set_name, test_set_name):
    l1 = list(range(1, 51))
    l2 = list(range(51, 101))
    l3 = list(range(101, 151))
    
    random.shuffle(l1)
    random.shuffle(l2)
    random.shuffle(l3)
    
    train_set = data.loc[l1[:35]+l2[:35]+l3[:35],:]
    test_set = data[~data.index.isin(train_set.index)]
    
    train_set.to_csv(train_set_name, sep=' ', index=False, header=None)
    test_set.to_csv(test_set_name, sep=' ', index=False, header=None)
