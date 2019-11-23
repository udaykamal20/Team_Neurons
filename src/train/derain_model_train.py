# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 00:38:13 2017

@author: udaykamal
"""
"""
code for derain model training. we use 5000 level 01 rainy frames and corresponding
no challenge frames for training.

"""

df_rain_load_path='D:\\dataframe\\derain_level_01_data.pkl'
derain_weight_savepath="D:\\weights\\weights_derain.hdf5"

import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Input,merge, Convolution2D, Activation, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K


img_rows = 618
img_cols = 814

train_rain=pd.read_pickle(df_rain_load_path)
train_rain=train_rain.sample(frac=1,random_state=200)

def get_image(d,ind,size=(img_cols,img_rows)):
    ### Get image by name
        
    file_name = d['rain'][d.index[ind]]
    img = cv2.imread(file_name)
    
    while (np.all(img)==None):
        ind+=1
        file_name = d['rain'][d.index[ind]]
        img = cv2.imread(file_name)
    
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,size)
    
    file_name = d['clear'][d.index[ind]]
    no_noise_image= cv2.imread(file_name)
    no_noise_image = cv2.cvtColor(no_noise_image,cv2.COLOR_BGR2RGB)
    no_noise_image = cv2.resize(no_noise_image,size)
    
    return img,no_noise_image

def generate_train_batch(data):
    x_batch = np.zeros((1, img_rows, img_cols, 3))
    y_batch = np.zeros((1, img_rows, img_cols, 3))
    while 1:            
        for i_line in range(len(data)-100):
            img,gt_image = get_image(data,i_line,size=(img_cols, img_rows))
            x_batch[0]=img
            y_batch[0]=gt_image
            yield x_batch,y_batch
            

def generate_test_batch(data):
    x_batch = np.zeros((1, img_rows, img_cols, 3))
    y_batch = np.zeros((1, img_rows, img_cols, 3))
    while 1:
        for i_line in range(len(data)-100,len(data)):
            img,gt_image = get_image(data,i_line,size=(img_cols, img_rows))
            x_batch[0]=img
            y_batch[0]=gt_image
            yield x_batch,y_batch
            
def PSNRLoss(y_true, y_pred):
    
    """
    PSNR is Peek Signal to Noise Ratio, which is similar to mean squared error.
    It can be calculated as
    PSNR = 20 * log10(MAXp) - 10 * log10(MSE)
    When providing an unscaled input, MAXp = 255. Therefore 20 * log10(255)== 48.1308036087.
    """
    return 48.13 + (10.0*K.log(1.0/(K.mean(K.square(y_pred - y_true))))/K.log(10.0))
        
def make_conv_block(nb_filters, input_tensor, block):
    def make_stage(input_tensor, stage):
        name = 'conv_{}_{}'.format(block, stage)
        x = Convolution2D(nb_filters, 3, 3, border_mode='same', name=name)(input_tensor)
        name = 'batch_norm_{}_{}'.format(block, stage)
        x = BatchNormalization(name=name)(x)
        x = Activation('relu')(x)
        return x

    x = make_stage(input_tensor, 1)
    x = make_stage(x, 2)
    return x

def derain():
    
    inputs = Input((img_rows, img_cols,3))
    
    conv1=Convolution2D(16,3,3, border_mode='same')(inputs)
    batch1=BatchNormalization()(conv1)
    relu1=Activation('relu')(batch1)
    
    conv2 = make_conv_block(16,relu1,2)
    m2 = merge([relu1, conv2], mode='sum')
    
    conv3=make_conv_block(16,m2,3)
    m3 = merge([m2, conv3], mode='sum')
    
    conv4=make_conv_block(16,m3,4)
    m4 = merge([m3, conv4], mode='sum')
    
    conv5=make_conv_block(16,m4,5)
    m5 = merge([m4, conv5], mode='sum')
    
    conv6=make_conv_block(16,m5,6)
    m6 = merge([m5, conv6], mode='sum')
    
    conv7=make_conv_block(16,m6,7)
    m7 = merge([m6, conv7], mode='sum')
    
    conv8=make_conv_block(16,m7,8)
    m8 = merge([m7, conv8], mode='sum')
    
    conv9=Convolution2D(3,3,3, border_mode='same')(m8)
    batch2=BatchNormalization()(conv9)
    
    autoencoder = Model(inputs, batch2)
    
    return autoencoder

training_gen = generate_train_batch(train_rain)
val_gen=generate_test_batch(train_rain)

model = derain()

model.compile(optimizer=Adam(lr=0.1),loss='mse', metrics=['acc',PSNRLoss])


def scheduler(epoch):
    if epoch!=0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr*.1)
        print("lr changed to {}".format(lr*.1))
    return K.get_value(model.optimizer.lr)

lr_decay = LearningRateScheduler(scheduler)
checkpoint = ModelCheckpoint(derain_weight_savepath, monitor='val_PSNRLoss', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint,lr_decay]
history = model.fit_generator(training_gen,samples_per_epoch=5000,nb_epoch=4,validation_data=val_gen,nb_val_samples=100,callbacks=callbacks_list)


def test_new_img(img):
    
    img = cv2.resize(img,(img_cols, img_rows))
    img = np.reshape(img,(1,img_rows, img_cols,3))
    pred = model.predict(img)
    pred = np.array(pred[0],dtype=np.uint8)
    img= np.array(img[0],dtype=np.uint8)
    return pred,img

#for training history
plt.plot(history.history['PSNRLoss'])
plt.plot(history.history['val_PSNRLoss'])
plt.title('model PSNR')
plt.ylabel('PSNR')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()