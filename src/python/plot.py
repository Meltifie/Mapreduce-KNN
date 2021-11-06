# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 16:54:30 2021

@author: 92831
"""


import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np



#%%
# Visualizing 5-D mix data using bubble charts
# leveraging the concepts of hue, size and depth
data = pd.read_csv('test_set3.txt', sep=' ', header=None)
y = pd.read_csv('result/k=2/predict3.txt', header=None)[0]

fig = plt.figure(figsize=(8, 6))
ax = Axes3D(fig)
t = fig.suptitle('k=2, sample 3', fontsize=14)

data1 = data[data[4]==1]
data2 = data[data[4]==2]
data3 = data[data[4]==3] 



ax.scatter(data1[0], data1[1], data1[2], alpha=0.4, c='red', s=data1[3]*100, label='setosa')
ax.scatter(data2[0], data2[1], data2[2], alpha=0.4, c='green', s=data2[3]*100, label='versicolor')
ax.scatter(data3[0], data3[1], data3[2], alpha=0.4, c='blue', s=data3[3]*100, label='virginica')

ax.legend(frameon=True, title='Name') 
ax.set_xlabel('Sepal.Length')
ax.set_ylabel('Sepal.Width')
ax.set_zlabel('Petal.Length')

