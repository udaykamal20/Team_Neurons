#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 22:05:19 2018

@author: root
"""

weightpath="D:\\weights\\weights_classifier.hdf5" #path for classifier model weight to be saved
matreadpath='D:\\matwritepy\\' #path for reading stored ROI and labels matrix for training

from sklearn.utils import shuffle
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.utils import np_utils
from keras.optimizers import Adam
import numpy as np
from keras.models import Sequential
import matplotlib.pyplot as plt
from keras.layers import Activation
from keras.callbacks import ModelCheckpoint
import cv2
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Reshape, Permute
from keras.layers import GlobalAveragePooling2D, BatchNormalization, UpSampling2D
from keras.layers import ZeroPadding2D
from keras.layers import multiply, add, concatenate, merge
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from spatial_transformer import SpatialTransformer

hidden_size = 512 
num_class = 14
nb_epoch = 20

def get_image(d,ind):
    ### Get image by name

    file_name = d['impath'][d.index[ind]]
    img = cv2.imread(file_name)
    
    #skip the index if any image fails to load just to make sure the training process doesnt stop midway
    while (np.all(img)==None):
        ind+=1
        file_name = d['filepath'][d.index[ind]]
        img = cv2.imread(file_name)
        
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = img/255.

    return img



def train_batch(data,batch_size=1): # training data generator

#    batch_images = np.zeros((batch_size, img_rows, img_cols, 3))
    batch_labels = np.zeros((batch_size, 14))
    while 1:
        for j in range(0,len(data)-4000):
            img = get_image(data,j)
            batch_images = np.expand_dims(img , axis = 0)
            batch_labels = data['imlabel'][data.index[j]]
            yield batch_images, batch_labels



def val_batch(data,batch_size=1): # training data generator

#    batch_images = np.zeros((batch_size, img_rows, img_cols, 3))
    batch_labels = np.zeros((batch_size, 14))
    while 1:
        for j in range(len(data)-4000, len(data)):
            img = get_image(data,j)
            batch_images = np.expand_dims(img)
            batch_labels = data['imlabel'][data.index[j]]
            yield batch_images, batch_labels


def make_conv_block(nb_filters, input_tensor, block):
    def make_stage(input_tensor, stage):
        name = 'conv_{}_{}'.format(block, stage)
        x = Conv2D(nb_filters, (3, 3), border_mode='same', name=name)(input_tensor)
        name = 'batch_norm_{}_{}'.format(block, stage)
        x = BatchNormalization(name=name)(x)
        x = Activation('relu')(x)
        return x

    x = make_stage(input_tensor, 1)
    x = make_stage(x, 2)
    return x

def get_init_weight():
    
    b = np.zeros((2, 3), dtype='float32')
    b[0, 0] = 1
    b[1, 1] = 1
    W = np.zeros((50, 6), dtype='float32')
    weights = [W, b.flatten()]
    return weights
    

#def create_st_net():
#    
#    weights  = get_init_weight()
#    img_input = Input(shape=( None, None, 3))
#
#    conv1 = Conv2D(32, (3, 3))(img_input) 
#    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
#
#    conv2 = Conv2D(32, (3, 3))(pool1)
#    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
#    
#    out1 = GlobalAveragePooling2D(pool2)
#    
#    out1 = Dense(50, activation = 'relu')(out1)
#    out = Dense(6, weights=weights)(out1)
#    model = Model(inputs=img_input, outputs= out)
#    
#    return model

def create_model():
    
    weights  = get_init_weight()
    img_input = Input(shape=( None, None, 3))
    input_shape = (None, None, 3)
    
    conv1 = Conv2D(32, (3, 3))(img_input) 
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(32, (3, 3))(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    out1 = GlobalAveragePooling2D(pool2)
    
    out1 = Dense(50, activation = 'relu')(out1)
    out = Dense(6, weights=weights)(out1)
    locnet = Model(inputs=img_input, outputs= out)
    
    in1 = SpatialTransformer(localization_net=locnet,
                             output_size=(32,32), input_shape=input_shape)

    conv1=make_conv_block(32,in1,1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
#    drop1 = Dropout(0.5)(pool1)
    
    conv2=make_conv_block(64,pool1,2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    flat1 = Flatten()(pool2)
    dense1 = Dense(hidden_size, activation = 'relu')(flat1)
    out = Dense(num_class, activation = 'softmax')(dense1)
    
    model = Model(inputs=img_input, outputs= out)
    return model

model = create_model()
print('model_created')

model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=1e-4),metrics=['accuracy'])

checkpoint = ModelCheckpoint(weightpath, monitor='val_acc', verbose=1, save_best_only=True, mode='max') #to save the model only when the validation accuracy increases

train = df.sample(df, frac = 1, random_state = 200)
training_gen = train_batch(train)
val_gen = val_batch(train)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                      patience=1, min_lr= 1e-6, mode = 'min')

early_stop = EarlyStopping(monitor='val_loss', 
                   min_delta=0.001, 
                   patience=5, 
                   mode='min', 
                   verbose=1)

saveweightpath = 'sign_classinfier_with_st.h5'

checkpoint = ModelCheckpoint(saveweightpath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')




history = model.fit_generator( 
                     generator = training_gen, 
                     steps_per_epoch  = len(train)-4000, 
                     epochs           = nb_epoch, 
                     verbose          = 1,
                     validation_data  = val_gen,
                     validation_steps = 4000,
                     callbacks        = [early_stop, checkpoint, reduce_lr], 
                     workers          = 3,
                     max_queue_size   = 8)



plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()