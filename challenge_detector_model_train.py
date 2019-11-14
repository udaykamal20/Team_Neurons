
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 15:17:55 2017

@author: User
"""
"""
This code is used for challenge detector training. We use all level 3 and for dirty lens level 3 and 5 challenge frames for training.

"""

df_load_path='D:\\dataframe\\challenge_detector_data.pkl'
weightpath='D:\\weights\\challenge_detector_weight.h5'

from keras.layers import Convolution2D, MaxPooling2D, Dense, Flatten
from keras.utils import np_utils
import numpy as np
from keras.models import Sequential
from keras.layers import Activation
import cv2
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from keras import backend as K
import pandas as pd
from keras.callbacks import LearningRateScheduler

num_epochs = 20
pool_size = 4
hidden_size = 100

  

img_rows = 309
img_cols = 407



train=pd.read_pickle(df_load_path)
num_classes = len(train['ch_type'].unique())

def get_image(d,ind,size=(img_cols,img_rows)):
    ### Get image by name

    file_name = d['filepath'][d.index[ind]]
    img = cv2.imread(file_name)
    
    #skip the index if any image fails to load just to make sure the training process doesnt stop midway
    while (np.all(img)==None):
        ind+=1
        file_name = d['filepath'][d.index[ind]]
        img = cv2.imread(file_name)
        
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,size)

    return img


Y_train = np_utils.to_categorical(train['ch_type'], num_classes)


def generate_train_batch(data,batch_size=1): # training data generator

    batch_images = np.zeros((batch_size, img_rows, img_cols, 3))
    batch_labels = np.zeros((batch_size, 12))
    while 1:
        for j in range(1000):
            i_line = np.random.randint(len(data)-500)
            img = get_image(data,i_line,size=(img_cols, img_rows))
            batch_images[0] = img
            batch_labels=Y_train[i_line][np.newaxis]
            yield batch_images, batch_labels

def generate_test_batch(data,batch_size=1): # validation data generator

    batch_images = np.zeros((batch_size, img_rows, img_cols, 3))
    batch_labels = np.zeros((batch_size, 12))
    while 1:
        for j_line in range(len(data)-500,len(data)):
            img = get_image(data,j_line,size=(img_cols, img_rows))
            batch_images[0] = img
            batch_labels=Y_train[j_line][np.newaxis]
            yield batch_images, batch_labels


model = Sequential();
model.add(Convolution2D(5,3,3, activation='relu',border_mode='valid',input_shape=(img_rows, img_cols,3)));
model.add(Convolution2D(5,3,3,  activation='relu',border_mode='valid'));
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)));

model.add(Convolution2D(5,3,3, border_mode='valid', activation='relu'));
model.add(Convolution2D(5,3,3, border_mode='valid', activation='relu'));
model.add(MaxPooling2D(pool_size=(2, 2)));

model.add(Convolution2D(5,3,3, border_mode='valid', activation='relu'));
model.add(Convolution2D(5,3,3, border_mode='valid', activation='relu'));
model.add(MaxPooling2D(pool_size=(2, 2)));

model.add(Convolution2D(5,3,3, border_mode='valid', activation='relu'));
model.add(Convolution2D(5,3,3, border_mode='valid', activation='relu'));
model.add(MaxPooling2D(pool_size=(2,2)));


model.add(Flatten())
model.add(Dense(hidden_size))
model.add(Activation('relu'))

model.add(Dense(num_classes));
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=1e-3),metrics=['accuracy']);
training_gen = generate_train_batch(train,1)
val_gen=generate_test_batch(train,1)

#decaying learning rate to half after every two epoch
def scheduler(epoch):
    
    if epoch%2==0 & epoch!=0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr*.5)
        print("lr changed to {}".format(lr*.5))
        
    return K.get_value(model.optimizer.lr)

lr_decay = LearningRateScheduler(scheduler)
checkpoint = ModelCheckpoint(weightpath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')#to save the model only when the validation accuracy increases

callbacks_list = [lr_decay,checkpoint] 

history = model.fit_generator(training_gen,samples_per_epoch=5000,nb_epoch=num_epochs,validation_data=val_gen,nb_val_samples=500,callbacks=callbacks_list)

#for printing training history
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


