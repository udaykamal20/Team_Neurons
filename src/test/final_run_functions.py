a# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 16:33:46 2017

@author: DSP Lab
"""

import cv2
import numpy as np
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input, merge, Flatten, Dense, MaxPooling2D, Dropout,Convolution2D,Cropping2D,ZeroPadding2D, UpSampling2D,Activation, BatchNormalization
from scipy.ndimage.measurements import label
from skimage.util import view_as_blocks


img_rows_noise = 309
img_cols_noise = 407
img_rows_loca = 618
img_cols_loca = 814


##Derain model function

def make_conv_block2(nb_filters, input_tensor, block):
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

    inputs = Input((img_rows_loca, img_cols_loca,3))
    
    conv1=Convolution2D(16,3,3, border_mode='same')(inputs)
    batch1=BatchNormalization()(conv1)
    relu1=Activation('relu')(batch1)
    
    conv2 = make_conv_block2(16,relu1,2)
    m2 = merge([relu1, conv2], mode='sum')
    
    conv3=make_conv_block2(16,m2,3)
    m3 = merge([m2, conv3], mode='sum')
    
    conv4=make_conv_block2(16,m3,4)
    m4 = merge([m3, conv4], mode='sum')
    
    conv5=make_conv_block2(16,m4,5)
    m5 = merge([m4, conv5], mode='sum')
    
    conv6=make_conv_block2(16,m5,6)
    m6 = merge([m5, conv6], mode='sum')
    
    conv7=make_conv_block2(16,m6,7)
    m7 = merge([m6, conv7], mode='sum')
    
    conv8=make_conv_block2(16,m7,8)
    m8 = merge([m7, conv8], mode='sum')
    
    conv9=Convolution2D(3,3,3, border_mode='same')(m8)
    batch2=BatchNormalization()(conv9)
    
    autoencoder = Model(inputs, batch2)
    
    return autoencoder



##Localizer model functions

def make_conv_block(nb_filters, input_tensor, block):
    
    def make_stage(input_tensor, stage):
        
        name = 'conv_{}_{}'.format(block, stage)
        x = Convolution2D(nb_filters, 3, 3, border_mode='same',activation='relu', name=name)(input_tensor)
        name = 'batch_norm_{}_{}'.format(block, stage)
        x = BatchNormalization(name=name)(x)
        x = Activation('relu')(x)
        return x

    x = make_stage(input_tensor, 1)
    x = make_stage(x, 2)
    return x


def get_crop_shape(target, refer):
    # width, the 3rd dimension
    cw = (target.get_shape()[2] - refer.get_shape()[2]).value
    assert (cw >= 0)
    if cw % 2 != 0:
        cw1, cw2 = int(cw/2), int(cw/2) + 1
    else:
        cw1, cw2 = int(cw/2), int(cw/2)
    # height, the 2nd dimension
    ch = (target.get_shape()[1] - refer.get_shape()[1]).value
    assert (ch >= 0)
    if ch % 2 != 0:
        ch1, ch2 = int(ch/2), int(ch/2) + 1
    else:
        ch1, ch2 = int(ch/2), int(ch/2)

    return (ch1, ch2), (cw1, cw2)

def localizer():
    
    inputs = Input((img_rows_loca, img_cols_loca,3))
    concat_axis = 3
    
    conv1=make_conv_block(16,inputs,1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2=make_conv_block(32,pool1,2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3=make_conv_block(64,pool2,3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4=make_conv_block(128,pool3,4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5=make_conv_block(256,pool4,5)

    up_conv5 = UpSampling2D(size=(2, 2))(conv5)
    
    ch, cw = get_crop_shape(conv4, up_conv5)
    crop_conv4 = Cropping2D(cropping=(ch,cw))(conv4)
    
    up6 = merge([up_conv5, crop_conv4], mode='concat', concat_axis=concat_axis)
    
    conv6=make_conv_block(128,up6,6)

    up_conv6 = UpSampling2D(size=(2, 2))(conv6)
    ch, cw = get_crop_shape(conv3, up_conv6)
    
    crop_conv3 = Cropping2D(cropping=(ch,cw))(conv3)
    up7 = merge([up_conv6, crop_conv3], mode='concat', concat_axis=concat_axis)
    
    conv7=make_conv_block(64,up7,7)

    up_conv7 = UpSampling2D(size=(2, 2))(conv7)
    ch, cw = get_crop_shape(conv2, up_conv7)
    
    crop_conv2 = Cropping2D(cropping=(ch,cw))(conv2)
    up8 = merge([up_conv7, crop_conv2], mode='concat', concat_axis=concat_axis)
    
    conv8=make_conv_block(32,up8,8)
    
    up_conv8 = UpSampling2D(size=(2, 2))(conv8)
    ch, cw = get_crop_shape(conv1, up_conv8)
    crop_conv1 = Cropping2D(cropping=(ch,cw))(conv1)

    up9 = merge([up_conv8, crop_conv1], mode='concat', concat_axis=concat_axis)
    
    conv9=make_conv_block(16,up9,9)

    ch, cw = get_crop_shape(inputs, conv9)
    conv9 = ZeroPadding2D(padding=(ch[0], ch[1], cw[0], cw[1]))(conv9)
    
    conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)

    return model

##Sign type classifier model

def classifier():
    
    num_classes=14
    kernel_size = 3
    pool_size = 2
    conv_depth_1 = 32
    conv_depth_2 = 64
    drop_prob_2 = 0.25 
    hidden_size = 1024 
    
    model = Sequential();
    model.add(Convolution2D(conv_depth_1,kernel_size, kernel_size, activation='relu',border_mode='same',input_shape=(32,32,3)));
    model.add(Convolution2D(conv_depth_1,kernel_size, kernel_size,  activation='relu',border_mode='same'));
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)));
    model.add(Dropout(drop_prob_2));
    
    model.add(Convolution2D(conv_depth_2, kernel_size, kernel_size, border_mode='same', activation='relu'));
    model.add(Convolution2D(conv_depth_2, kernel_size, kernel_size, border_mode='same', activation='relu'));
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)));
    model.add(Dropout(drop_prob_2));
    
    
    model.add(Flatten())
    model.add(Dense(hidden_size));
    model.add(Activation('relu'))
    model.add(Dense(num_classes));
    model.add(Activation('softmax'))
    
    return model

#Challenge classifier model

def challenge_detector_model():
    
    pool_size = 4
    hidden_size = 100
    num_classes=12
    
    model = Sequential();
    
    model.add(Convolution2D(5, 3, 3, activation='relu',border_mode='valid',input_shape=(img_rows_noise, img_cols_noise,3)));
    model.add(Convolution2D(5, 3, 3,  activation='relu',border_mode='valid'));
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)));

    
    model.add(Convolution2D(5,3,3, border_mode='valid', activation='relu'));
    model.add(Convolution2D(5,3,3, border_mode='valid', activation='relu'));
    model.add(MaxPooling2D(pool_size=(2, 2)));

    model.add(Convolution2D(5,3,3, border_mode='valid', activation='relu'));
    model.add(Convolution2D(5, 3,3, border_mode='valid', activation='relu'));
    model.add(MaxPooling2D(pool_size=(2, 2)));
    
    model.add(Convolution2D(5,3,3, border_mode='valid', activation='relu'));
    model.add(Convolution2D(5, 3,3, border_mode='valid', activation='relu'));
    model.add(MaxPooling2D(pool_size=(2,2)));
    
    model.add(Flatten())
    model.add(Dense(hidden_size));
    model.add(Activation('relu'))

    model.add(Dense(num_classes));
    model.add(Activation('softmax'))
    
    return model

##Frames preprocessing functions 

#decolor
def CLAHE_decolor(img):
    hsv=cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv[:,:,1]=cv2.createCLAHE(clipLimit=10, tileGridSize=(2,2)).apply(hsv[:,:,1])
    hsv[:,:,1]=cv2.createCLAHE(clipLimit=10, tileGridSize=(2,2)).apply(hsv[:,:,1])
    hsv[:,:,2]=cv2.createCLAHE(clipLimit=10, tileGridSize=(2,2)).apply(hsv[:,:,2])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

#dark
def CLAHE_dark(img):
    hsv=cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv[:,:,1]=cv2.createCLAHE(clipLimit=10, tileGridSize=(2,2)).apply(hsv[:,:,1])
    hsv[:,:,2]=cv2.createCLAHE(clipLimit=50, tileGridSize=(2,2)).apply(hsv[:,:,2])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

#exposure
def CLAHE_exposure(img):
    hsv=cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv[:,:,1]=cv2.createCLAHE(clipLimit=5, tileGridSize=(5,5)).apply(hsv[:,:,1])
    hsv[:,:,2]=cv2.createCLAHE(clipLimit=5, tileGridSize=(2,2)).apply(hsv[:,:,2])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

#rain
def rain_out(img,model_rain):
    img = np.reshape(img,(1,img_rows_loca, img_cols_loca,3))
    pred = model_rain.predict(img)
    return np.array(pred[0],dtype=np.uint8)

def re(stacked): 
    reconstructed = np.empty((1236, 1628, 3), dtype = np.uint8)
    for k in range(4):
        i = np.floor(k/2).astype(int)
        j = k%2
 
        x_start = i*618
        x_end = x_start + 618
        y_start = j*814
        y_end = y_start + 814
       
        reconstructed[x_start:x_end, y_start:y_end, :] = stacked[k, :, :, :]
    return reconstructed


def get_patches(img): 
    y = view_as_blocks(img, (618, 814, 3))
    stacked = y.reshape(4, 618, 814, 3)
    return stacked

def rain_preprocess(img,model_rain):
    
    stacked=get_patches(img) #creates 4 patches for deraining
    for i in range(4):
        stacked[i]=rain_out(stacked[i],model_rain)
        
    img=re(stacked) #stacks four derained patches to create original size
    
    hsv=cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv[:,:,1]=cv2.createCLAHE(clipLimit=5, tileGridSize=(2,2)).apply(hsv[:,:,1])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

#shadow
def dark_normal_seg(I):  
    # I.shape = (1236, 1628, 3)
    # returns patches of dark and normal segments
    # each has 12 patches
    # normal patches has width of 75 (except the last normal patch), so normal.shape=(12, 1236, 75, 3)
    # dark patches has width of 67 (except the first dark patch), so shadow.shape(12, 1236, 67, 3)
   
    # Width and Height of the image
    H = I.shape[0]
 
    wDark = 67
    wNormal = 75
   
    normal = np.zeros((12, H, wNormal, 3), dtype='uint8')
    dark = np.zeros((12, H, wDark, 3), dtype='uint8')
 
    # The first dark patch is 30 pixel wide. For the first patch, we concat 37 rows of zeros to the left of the patch to make it's width 67
    firstDark = np.zeros((H, 67, 3), dtype = 'uint8')
    firstDark[:, -30:, :] = I[:, :30, :]
    dark[0, :, :, :] = firstDark
    iCol = 30
   
    for i in range(11):
        normal[i, :, :, :] = I[:, iCol:(iCol+wNormal), :]
        iCol = iCol + wNormal
        dark[i+1, :, :, :] = I[:, iCol:(iCol+wDark), :]
        iCol = iCol + wDark
   
    # The last patch is 37 pixel wide. For the last normal patch, we concat 38 rows of zeros to the right of the patch to make it's width 75
    lastNormal = np.zeros((H, 75, 3), dtype = 'uint8')
    lastNormal[:, 0:36, :] = I[:, iCol:(iCol+36), :]
    normal[11, :, :, :] = lastNormal
   
    return normal, dark
 
def patches_to_image(normal, dark):
    # shape of normal (12, 1236, 75 3)
    # shape of dark (12, 1236, 67, 3)
    # output img, size (1236, 1628, 3)
   
    wNormal = 75
    wDark = 67
 
    W = 1628
    H = 1236
 
 
    img = np.zeros((H, W, 3), dtype = 'uint8')
 
    # first dark patch
    img[:, 0:30, :] = dark[0, :, -30:, :]
    iCol = 30
 
    for i in range(11):
        img[:, iCol:(iCol + wNormal), :] = normal[i, :, :, :]
 
        iCol = iCol + wNormal
 
        img[:, iCol:(iCol + wDark), :] = dark[i+1, :, :, :]
       
        iCol = iCol + wDark
 
    # last normal patch
    img[:, iCol:(iCol + 36), :] = normal[11, :, 0:36, :]
   
    return img

def CLAHE_shadow(img):
    hsv=cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    white,dark=dark_normal_seg(hsv) #creates dark and white patches
    
    for i in range(12):
        
        white[i,:,:,1]=cv2.createCLAHE(clipLimit=5, tileGridSize=(2,2)).apply(white[i,:,:,1])
        white[i,:,:,2]=cv2.createCLAHE(clipLimit=10, tileGridSize=(2,2)).apply(white[i,:,:,2])
        dark[i,:,:,1]=cv2.createCLAHE(clipLimit=5, tileGridSize=(2,2)).apply(dark[i,:,:,1])
        dark[i,:,:,2]=cv2.createCLAHE(clipLimit=10, tileGridSize=(2,2)).apply(dark[i,:,:,2])
        
    out=patches_to_image(white,dark) #concatenates the processed patches
    return cv2.cvtColor(out, cv2.COLOR_HSV2RGB)

#snow
def CLAHE_snow(img):
    hsv=cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv[:,:,1]=cv2.createCLAHE(clipLimit=10, tileGridSize=(5,5)).apply(hsv[:,:,1])
    hsv[:,:,2]=cv2.createCLAHE(clipLimit=5, tileGridSize=(2,2)).apply(hsv[:,:,2])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

#haze
def CLAHE_haze(img):
    hsv=cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv[:,:,1]=cv2.createCLAHE(clipLimit=5, tileGridSize=(2,2)).apply(hsv[:,:,1])
    hsv[:,:,2]=cv2.createCLAHE(clipLimit=5, tileGridSize=(2,2)).apply(hsv[:,:,2])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

#localizer output, returns the bounding box coordinates
def localizer_output(model, img):
    #M contains the bounding boxes coordinates
    
    M=np.empty((1,4))
    img=cv2.resize(img,(814,618))
    img = np.reshape(img, (1, img_rows_loca, img_cols_loca, 3) )
    pred = model.predict(img)
    img_pred = np.array(pred[0],dtype=np.uint8)
    labels = label(img_pred[:,:,0])
    for sign in range(1, labels[1]+1):
        nonzero = (labels[0] == sign).nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        if ((np.max(nonzeroy)-np.min(nonzeroy)>3) & (np.max(nonzerox)-np.min(nonzerox)>3)): #we ignore the too small detected regions to reduce false positives
            p=[np.min(nonzerox), np.min(nonzeroy), np.max(nonzerox), np.max(nonzeroy)]
            p=np.reshape(p,(1,4))
            M=np.append(M,p,axis=0)
    
    if len(M)==1:
        return np.empty((0,4))
    
    return 2*M[1:] #rescale the coordinates


#classifier output, returns the sign type
def classifier_output(model, bb):
    c=cv2.resize(bb,(32,32))
    c = np.reshape(c,(1,32,32,3))
    d= c/255
    y_pred=model.predict(d,batch_size=1);
    p=np.sort(y_pred) #sorts the scores from lowest to highest
    if (p[0,-1]-p[0,-2])<0.1: #to reduce the false positives we ignores the cases where the difference between two max confidence score is less than 0.1 
        return 0
    return y_pred.argmax(axis=1)+1;


#challenge type output, returns the challenge type
#as 08,09,10,11,12 types has class output 07,08,09,10,11 so we add 1 to their output class   
def out_challenge_type(img,model):
    
    img = cv2.resize(img,(img_cols_noise,img_rows_noise))
    img = np.reshape(img,(1,img_rows_noise, img_cols_noise,3)).astype('float32')
    pred=model.predict(img,1)
    
    if np.max(pred)<0.1: 
        return 0
    y=pred.argmax(axis=1);
    return y+(y>6)
