# -*- coding: utf-8 -*-
"""
Created on Sun May 14 19:07:42 2017

@author: udaykamal
"""

"""
This code is for traffic sign type classifier training. We use the matrix (stored in matreadpath directory) of ROI from the
no noise and level 1,3 of all the challenge types

"""

weightpath="D:\\weights\\weights_classifier.hdf5" #path for classifier model weight to be saved
matreadpath='D:\\matwritepy\\' #path for reading stored ROI and labels matrix for training

from sklearn.utils import shuffle
from keras.layers import Convolution2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.utils import np_utils
from keras.optimizers import Adam
import numpy as np
from keras.models import Sequential
import matplotlib.pyplot as plt
from keras.layers import Activation
from keras.callbacks import ModelCheckpoint


X_train=np.load(matreadpath+'allbb')
Y_train=np.load(matreadpath+'alllabel')


(X_train, Y_train)=shuffle(X_train, Y_train, random_state=0)

#to reduce the number of samples of 14no sign class to balance the data. we take 3/4 portion of the total 14no class samples
X_train=np.append(X_train[Y_train!=14], X_train[Y_train==14][0:int((3/4)*len(X_train[Y_train==14]))], axis=0)
Y_train=np.append(Y_train[Y_train!=14], Y_train[Y_train==14][0:int((3/4)*len(Y_train[Y_train==14]))], axis=0)

(X_train, Y_train)=shuffle(X_train, Y_train, random_state=0) #re shuffling the data

batch_size = 64 
num_epochs = 10    
kernel_size = 3
pool_size = 2
conv_depth_1 = 32
conv_depth_2 = 64
drop_prob_1 = 0.5 
drop_prob_2 = 0.25 
hidden_size = 1024 
 

num_classes = np.unique(Y_train).shape[0]

X_train = X_train.astype('float32') 

X_train /= np.max(X_train) #normalizing training data
 


Y_train = np_utils.to_categorical(Y_train-1, num_classes) # to tranform the labels from 1-14 to 0-13


model = Sequential();
model.add(Convolution2D(conv_depth_1,kernel_size, kernel_size, activation='relu',border_mode='same',input_shape=X_train.shape[1:]))
model.add(Convolution2D(conv_depth_1,kernel_size, kernel_size,  activation='relu',border_mode='same'))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
model.add(Dropout(drop_prob_2))

model.add(Convolution2D(conv_depth_2, kernel_size, kernel_size, border_mode='same', activation='relu'))
model.add(Convolution2D(conv_depth_2, kernel_size, kernel_size, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
model.add(Dropout(drop_prob_2))


model.add(Flatten())
model.add(Dense(hidden_size))
model.add(Activation('relu'))

model.add(Dense(num_classes));
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=1e-4),metrics=['accuracy']);



checkpoint = ModelCheckpoint(weightpath, monitor='val_acc', verbose=1, save_best_only=True, mode='max') #to save the model only when the validation accuracy increases
callbacks_list = [checkpoint]
history = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=num_epochs,verbose=1, validation_split=0.01,callbacks=callbacks_list) 

# to print the training history
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#model.save('cnn2.h5');















