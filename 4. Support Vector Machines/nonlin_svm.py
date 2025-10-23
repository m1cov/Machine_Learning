# -*- coding: utf-8 -*-
"""
This code shows an example of using a Support Vector Machine (SVM) with a 
Gaussian kernel (rbf) to solve a classification problem. 
The SVM is constructed from the scikit learn python module. 
It is encouraged to try several scenarios for different values for the penalty
coefficient C and the kernel parameter gamma. 
"""

import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn import svm

# Import, format, and scatter plot data:
raw_data = loadmat('ex6data2.mat')

data = pd.DataFrame(raw_data['X'], columns=['X1', 'X2'])
data['y'] = raw_data['y']

positive = data[data['y'].isin([1])]
negative = data[data['y'].isin([0])]

fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(positive['X1'], positive['X2'], s=30, marker='x', label='Positive')
ax.scatter(negative['X1'], negative['X2'], s=30, marker='o', label='Negative')
ax.legend()

# Create SVM
svc = svm.SVC(C=100, gamma=10, probability=True)    # try it also with different kernels
# probability=True enables not only classification, but also probability estimation
svc.fit(data[['X1', 'X2']], data['y'])
print (svc.score(data[['X1', 'X2']], data['y']))

# Plot the classification results 
data['Probability'] = svc.predict_proba(data[['X1', 'X2']])[:,0]
fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(data['X1'], data['X2'], s=30, c=data['Probability'], cmap='Reds') 