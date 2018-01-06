'''
06/12/2017 15:22


DanQ modified by Revant Gupta to use an autoencoder as a sequence information capture step that is subsequently fed to a classifier,
the computational efficiency is greatly enchanced.

This version works with keras 2.1.1. (later versions are not guaranteed to work as keras updates can be backward incompatible)

'''
import numpy as np
import h5py
import scipy.io
np.random.seed(1337) # for reproducibility

from keras.preprocessing import sequence
from keras.optimizers import RMSprop, Nadam
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.constraints import maxnorm
from keras.layers.recurrent import LSTM
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Bidirectional

import sklearn as skl

print('loading data')
#trainmat = h5py.File('data/train.mat')
#validmat = scipy.io.loadmat('data/valid.mat')
#testmat = scipy.io.loadmat('data/test.mat')

#X_train = np.transpose(np.array(trainmat['trainxdata']),axes=(2,0,1))
#y_train = np.array(trainmat['traindata']).T

X_train = np.ones((500,1000,4))
y_train = np.zeros((500,919))

nadam = Nadam(lr=0.015, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)

print('building model')

model = Sequential()
model.add(Conv1D(320, 26, input_shape=(1000,4)))

model.add(Dense(500)) # Encoder

model.add(Dense(4000))

model.add(Reshape((1,1000,4))) # Decoder

print('compiling model')
                       
model.compile(optimizer= nadam, loss='mse', metrics = ['accuracy'])

encoder = Model(inputs=model.input, outputs=(model.layers[1]).output)

earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

model.fit(X_train, X_train, batch_size=100, nb_epoch=10, shuffle=True, verbose=2 #show_accuracy=True,
          #validation_data=(np.transpose(validmat['validxdata'],axes=(0,2,1)), validmat['validdata']),
          #callbacks=[checkpointer,earlystopper]
          )

encodings = model.predict(X_train, batch_size = 100, verbose=2)

classifier = skl.ensemble.RandomForestClassifier()

classifier.fit(encodings, y_train)

#evaluation = classifier.predict(np.transpose(testmat['testxdata'],axes=(0,2,1)))

#acc = skl.metrics.accuracy_score(testmat['testdata'], evaluation)

print(acc)

