# -*- coding: utf-8 -*-
"""
This code gives an example of applying Gaussian distribution anomaly detection 
to a set of two dimensional data. Precision/recall and F1-score are used to build
the model with a given training set, and then data from a testing set are used to
test the detection algorithm and to plot the anomalies. 
"""

import numpy as np  
import matplotlib.pyplot as plt  
from scipy.io import loadmat 
from scipy import stats   

def estimate_gaussian(X):       # calculate the mean and variance of a dataset
    mu = X.mean(axis=0)
    sigma = X.var(axis=0)
    return mu, sigma

def select_threshold(pval, yval):  # calculate the threshold and the f1-score 
    # Initialize nominal values:
    best_epsilon = 0
    best_f1 = 0
    f1 = 0
    # Initialize step size for 1000 steps across all possible probability values:
    step = (pval.max() - pval.min()) / 1000
    # Epsilon will take all the steps. For each epsilon:
    for epsilon in np.arange(pval.min(), pval.max(), step):
        # Preds are predictions - 1 if the probability is lower than epsilon (anomaly), 0 otherwise.
        # This is done for all features separately, not by multiplying them like in the lesson.
        preds = pval < epsilon
        #print (preds)
        # True positives (preds says it is an anomaly, and yval confirms):
        tp = np.sum(np.logical_and(preds == 1, yval == 1)).astype(float)
        # False positives (preds says it is an anomaly, but yval actualy isn't):
        fp = np.sum(np.logical_and(preds == 1, yval == 0)).astype(float)
        # False negatives (preds says it isn't an anomaly, but yval actualy is): 
        fn = np.sum(np.logical_and(preds == 0, yval == 1)).astype(float)
        
        precision = tp / (tp + fp)                              # calculate precision
        recall = tp / (tp + fn)                                 # calculate recall 
        f1 = (2 * precision * recall) / (precision + recall)    # calucate f1

        if f1 > best_f1:                # If it is the best score so far:
            best_f1 = f1                # store this score as the best one
            best_epsilon = epsilon      # store this epsilon as the best one

    return best_epsilon, best_f1        # return the best threshold and score
    
# Load and plot testing data:
data = loadmat('ex8data1.mat')              
X = data['X']  
fig, ax = plt.subplots(figsize=(12,8))  
ax.scatter(X[:,0], X[:,1])
# Calculate mean and variance of the features
mu, sigma = estimate_gaussian(X)
# Load training and validation data:
Xval = data['Xval']  
yval = data['yval']

# This function from scipy calculates the probability that the features of
# the data from the testing set belong to the distribution that
# was defined by the mu and sigma parameters calculated earlier. 
# Calculate the probability for all features for all testing samples:
p = np.zeros((X.shape[0], X.shape[1]))  
p[:,0] = stats.norm(mu[0], sigma[0]).pdf(X[:,0])  
p[:,1] = stats.norm(mu[1], sigma[1]).pdf(X[:,1])
# Calculate the probability for all features for all training/validation samples:
pval = np.zeros((Xval.shape[0], Xval.shape[1]))  
pval[:,0] = stats.norm(mu[0], sigma[0]).pdf(Xval[:,0])  
pval[:,1] = stats.norm(mu[1], sigma[1]).pdf(Xval[:,1])  
# Calculate the threshold epsilon and the quality of the algorithm f1
epsilon, f1 = select_threshold(pval, yval)  
print (epsilon, f1)
# Outliers contains the locations of data points wich have a probability lower than epsilon
outliers = np.where(p < epsilon)
print (outliers)
# Plot the testing data again:
fig, ax = plt.subplots(figsize=(12,8))  
ax.scatter(X[:,0], X[:,1])  
# On the same graph, plot  the outliers (the anomalies) with larger red dots:
ax.scatter(X[outliers[0],0], X[outliers[0],1], s=50, color='r', marker='o') 