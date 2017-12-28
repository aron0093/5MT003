# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import h5py
import numpy as np
from os import chdir, getcwd
import time

wd = getcwd()
print(wd)
chdir(wd)

path = 'KI-project/input/deepsea_train/train.mat'
t0 = time.time()

trainmat = h5py.File(path, 'r')


t1 = time.time()
tot = t1 - t0
print ('Dataset loaded, time: %f' % tot)

#%%

print (trainmat['trainxdata'].shape) # (1000, 4, 4400000)

#%%

print (trainmat['traindata'].shape) # (919, 4400000)

#%% 

xsub = (trainmat['trainxdata'][:,:, 0:440])
ysub = (trainmat['traindata'][:, 0:440])

#%%

print (xsub.shape)
print (ysub.shape)

#%%

## trainxdata contain the sequence information
    ## 44,000 sequences, n long, 4 bp
    
    # index 0: length of each sequence
    # index 1: 4 bp
    # index 2: 4,400,000 sequences
    
## traindata contain the labels
    
## transpose the data, .T returns the transpose of the data,
    # specify the order with axes
    
#%%  
    
tx0 = time.time()

print ('Start transpose')

xtrain = np.transpose(np.array(xsub), axes = (2, 0, 1))

tx1 = time.time()
totx = tx1 - tx0

print ('Stop transpose, time: %f' % totx)

#%%
## Save transposed x_train to hdf5 file
## whole dataset took 24 hrs to transpose

with h5py.File('xtrain.h5', 'w') as hf:
    hf.create_dataset("tr_xtrain", data = xtrain)

print ("Saved xtrain dataset.")

#%%
## Read x_train file

with h5py.File('xtrain.h5', 'r') as hf:
    xtrain = hf['tr_xtrain'][:]


#%%

#x_train = np.transpose(np.array(trainmat['trainxdata']),axes=(2,0,1))
ytrain = np.array(ysub).T

ty1 = time.time()
toty = ty1 - tx0
print (toty)


#%%

# dimensionality reduction, fuse all aminoacids in one peptide (100 aa long) 

nsamples, l, dn = xtrain.shape
d2_xtrain = xtrain.reshape((nsamples, l * dn))

#%%

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

clf = RandomForestClassifier(n_estimators = 50) 
# n_estimators: 500-1000 is ok, more better but slower
# explore max_features
# default none, reduction might reduce overfitting 
# min_sample_leaf > 50 # reduce noice in training data

#%%
## 10-Fold Cross validation
scores = cross_val_score(clf, d2_xtrain, ytrain, cv=10)

print(scores)
print (np.mean(scores))

#%%
## train model
#clf.fit(d2_xtrain[2:], ytrain[2:])

#%%
## test on first part
#accuracy_score = clf.score(d2_xtrain[0:1], ytrain[0:1])
#prediction = predict(d2_xtrain[0:1])