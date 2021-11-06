# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 16:32:39 2021

@author: 92831
"""

#%%
import pandas as pd

#%%
def cal_accuracy(y_file, y_hat_file):
    y_hat = pd.read_csv(y_hat_file, header=None)[0]
    y = pd.read_csv(y_file, sep=' ', header=None)[4]
    accuracy = sum(y_hat==y)/len(y)
    print(accuracy)
    