# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 04:10:51 2017

@author: DSP Lab
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 10:59:01 2017

@author: udaykamal
"""

"""
training code for all localizer. first we train no challenge localizer and save the weight.
then using the pretrained model, we further train 3 more loalizer for all level 01 localizer,
noise localizer, and blur localizer.

**Among the follwing variables:
    
    no_challenge,all_level_01_challenge,noise_level_03_challenge,blur_level_03_challenge

only the corresponding variable needs to be set True and all other needs to be set False

"""
#datapaths create
textreadpath='D:\\text\\train_with_sign.txt'  #datapath for no challenge
df_all_01_load_path='D:\\dataframe\\localizer_all_challenge_level_01_data.pkl' #datapath for all level 01 challenge
df_noise_03_load_path='D:\\dataframe\\localizer_noise_level_01_data.pkl' #datapath for noise level 03 challenge
df_blur_03_load_path='D:\\dataframe\\localizer_blur_level_01_data.pkl'#datapath for blur level 03 challenge

#weightpaths
no_challenge_weightpath='D:\\weights\\no_challenge_weight.h5'
all_level_01_challenge_weightpath='D:\\weights\\all_level_01_challenge_weight.h5'
noise_level_03_challenge_weightpath='D:\\weights\\noise_level_03_challenge_weight.h5'
blur_level_03_challenge_weightpath='D:\\weights\\blur_level_03_challenge_weight.h5'

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Input, merge,Convolution2D,Cropping2D,ZeroPadding2D, MaxPooling2D, UpSampling2D,Activation, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from scipy.ndimage.measurements import label

no_challenge=True
all_level_01_challenge=False
noise_level_03_challenge=False
blur_level_03_challenge=False

 
safe=0.1 #a samll value to be used in IOU calculation to prevent the division by 0 case when the frame contains no ROI

img_rows = 618
img_cols = 814

def get_image(d,ind,size=(img_cols,img_rows)):
    ### Get image by name

    file_name = d['filepath'][d.index[ind]]
    img = cv2.imread(file_name)
    
    while (np.all(img)==None):
        ind+=1
        file_name = d['filepath'][d.index[ind]]
        img = cv2.imread(file_name)
        
    img_size = np.shape(img)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,size)
    
    bb_boxes = d[d['filepath'] == file_name].reset_index()
    img_size_post = np.shape(img)
    
#scaling the ROI coordinates
    bb_boxes['xmin'] = np.round(bb_boxes['xmin']/img_size[1]*img_size_post[1])
    bb_boxes['xmax'] = np.round(bb_boxes['xmax']/img_size[1]*img_size_post[1])
    bb_boxes['ymin'] = np.round(bb_boxes['ymin']/img_size[0]*img_size_post[0])
    bb_boxes['ymax'] = np.round(bb_boxes['ymax']/img_size[0]*img_size_post[0])


    return img,bb_boxes

def get_mask_seg(img,bb_boxes_f):
#to create image mask using the frame. only the ROI region is set to 1 and all other is set to 0
    img_mask = np.zeros_like(img[:,:,0])
    for i in range(len(bb_boxes_f)):
        bb_box_i = [bb_boxes_f.iloc[i]['xmin'],bb_boxes_f.iloc[i]['ymin'],
                bb_boxes_f.iloc[i]['xmax'],bb_boxes_f.iloc[i]['ymax']]
        
        img_mask[int(bb_box_i[1]):int(bb_box_i[3]),int(bb_box_i[0]):int(bb_box_i[2])]= 1
        img_mask = np.reshape(img_mask,(np.shape(img_mask)[0],np.shape(img_mask)[1],1))
        
    return img_mask

def plot_im_mask(im,im_mask):
    ### Function to plot image mask

    im = np.array(im,dtype=np.uint8)
    im_mask = np.array(im_mask,dtype=np.uint8)
    plt.subplot(1,3,1)
    plt.imshow(im)
    plt.axis('off')
    plt.subplot(1,3,2)
    plt.imshow(im_mask[:,:,0])
    plt.axis('off')
    plt.subplot(1,3,3)
    plt.imshow(cv2.bitwise_and(im,im,mask=im_mask));
    plt.axis('off')
    plt.show();
#
def plot_bbox(bb_boxes,ind_bb,color='r',linewidth=2):
    ### Plot bounding box

    bb_box_i = [bb_boxes.iloc[ind_bb]['xmin'],
                bb_boxes.iloc[ind_bb]['ymin'],
                bb_boxes.iloc[ind_bb]['xmax'],
                bb_boxes.iloc[ind_bb]['ymax']]
    plt.plot([bb_box_i[0],bb_box_i[2],bb_box_i[2],
                  bb_box_i[0],bb_box_i[0]],
             [bb_box_i[1],bb_box_i[1],bb_box_i[3],
                  bb_box_i[3],bb_box_i[1]],
             color,linewidth=linewidth)
#
def plot_im_bbox(im,bb_boxes):
    ### Plot image and bounding box
    plt.imshow(im)
    for i in range(len(bb_boxes)):
        plot_bbox(bb_boxes,i,'g')

        bb_box_i = [bb_boxes.iloc[i]['xmin'],bb_boxes.iloc[i]['ymin'],
                bb_boxes.iloc[i]['xmax'],bb_boxes.iloc[i]['ymax']]
        plt.plot(bb_box_i[0],bb_box_i[1],'rs')
        plt.plot(bb_box_i[2],bb_box_i[3],'bs')
    plt.axis('off');
    
#training data generator
def generate_train_batch(data):

    batch_images = np.zeros((1, img_rows, img_cols, 3))
    batch_masks = np.zeros((1, img_rows, img_cols, 1))
    while 1:
        for i_line in range(len(data)-500):
            img,bb_boxes = get_image(data,i_line,size=(img_cols, img_rows),
                                                  augmentation=False,
                                                   trans_range=0,
                                                   scale_range=0
                                                  )
            img_mask = get_mask_seg(img,bb_boxes)
            batch_images[0] = img
            batch_masks[0] =img_mask
            yield batch_images, batch_masks


#validation data generator
def generate_test_batch(data):
    batch_images = np.zeros((1, img_rows, img_cols, 3))
    batch_masks = np.zeros((1, img_rows, img_cols, 1))
    while 1:
        for j_line in range(len(data)-500,len(data)):
            img,bb_boxes = get_image(data,j_line,size=(img_cols, img_rows),
                                                  augmentation=False,
                                                   trans_range=0,
                                                   scale_range=0
                                                  )
            img_mask = get_mask_seg(img,bb_boxes)
            batch_images[0] = img
            batch_masks[0] =img_mask
            yield batch_images, batch_masks


#IOU calculation
def IOU_calc(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)

    return 2*(intersection + safe) / (K.sum(y_true_f) + K.sum(y_pred_f) + safe)

#mdel loss function
def IOU_calc_loss(y_true, y_pred):
    return -IOU_calc(y_true, y_pred)


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
#
#
def get_crop_shape(target, refer):
    
    #cropper to make the layer dimensions consistent
    
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

#model architecture
def localizer():
    
    inputs = Input((img_rows, img_cols,3))
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


model = localizer()
epoch_no=4
LR=1e-4

if no_challenge==True:
    train = pd.read_csv(textreadpath, sep=",", names=["filepath", "xmin", "ymin","xmax","ymax"])
    saveweightpath=no_challenge_weightpath
    LR=1e-3

    
elif all_level_01_challenge==True:
    train = pd.read_pickle(df_all_01_load_path)
    model.load_weights(no_challenge_weightpath)
    epoch_no=10
    saveweightpath=all_level_01_challenge_weightpath
    
elif noise_level_03_challenge==True:
    train = pd.read_pickle(df_noise_03_load_path)
    model.load_weights(all_level_01_challenge_weightpath)
    saveweightpath=noise_level_03_challenge_weightpath
    
elif blur_level_03_challenge==True:
    train = pd.read_pickle(df_blur_03_load_path) 
    model.load_weights(all_level_01_challenge_weightpath)
    saveweightpath=blur_level_03_challenge_weightpath

train=train.sample(frac=1,random_state=200)
training_gen = generate_train_batch(train)
val_gen=generate_test_batch(train)

#decaying learning rate to half after every two epoch
def scheduler(epoch):
    
    if epoch%2==0 & epoch!=0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr*.5)
        print("lr changed to {}".format(lr*.5))
        
    return K.get_value(model.optimizer.lr)

lr_decay = LearningRateScheduler(scheduler)

checkpoint = ModelCheckpoint(saveweightpath, monitor='val_IOU_calc', verbose=1, save_best_only=True, mode='max')

callbacks_list = [lr_decay, checkpoint] 


model.compile(optimizer=Adam(lr=LR),loss=IOU_calc_loss, metrics=[IOU_calc])

#model.load_weights(filepath_load)
###
history = model.fit_generator(training_gen,samples_per_epoch=13000,nb_epoch=epoch_no,validation_data=val_gen,nb_val_samples=500,callbacks=callbacks_list)


#summarize history for accuracy


plt.figure(1)
plt.plot(history.history['IOU_calc'])
plt.plot(history.history['val_IOU_calc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()




def draw_labeled_bboxes(img, labels):

    for sign in range(1, labels[1]+1):
        
        nonzero = (labels[0] == sign).nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        if ((np.max(nonzeroy)-np.min(nonzeroy)>0) & (np.max(nonzerox)-np.min(nonzerox)>0)):
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            # Draw the box on the image       
            cv2.rectangle(img, bbox[0], bbox[1], (0,0,255),1)
    # Return the image
    return img

def get_BB_new_img(img):
    # Take in RGB image
    pred,img = test_new_img(img)
    img  = np.array(img,dtype= np.uint8)
    img_pred = np.array(255*pred[0],dtype=np.uint8)
    heatmap = img_pred[:,:,0]
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(img), labels)
    return draw_img
    
def test_new_img(img):
    img = cv2.resize(img,(img_cols, img_rows))
    img = np.reshape(img,(1,img_rows, img_cols,3))
    pred = model.predict(img)
    return pred,img[0]

#for i in range(1):
#    j=np.random.randint(len(train))
#    test_img =train['filepath'][train.index[j]]
#    im = cv2.imread(test_img)
#    im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
#    pred,im = test_new_img(im)
#    im  = np.array(im,dtype= np.uint8)
#    im_pred = np.array(255*pred[0],dtype=np.uint8)
#    rgb_mask_pred = cv2.cvtColor(im_pred,cv2.COLOR_GRAY2RGB)
#    rgb_mask_pred[:,:,1:3] = 0*rgb_mask_pred[:,:,1:2]
#    
#    
#    img_pred = cv2.addWeighted(rgb_mask_pred,0.55,im,1,0)
#    draw_img = get_BB_new_img(im)
#    plt.figure(i+1)
#    plt.figure(figsize=(10,5))
#    plt.subplot(1,3,1)
#    plt.imshow(im)
#    plt.title('Original')
#    plt.axis('off')
#    plt.subplot(1,3,2)
#    plt.imshow(img_pred)
#    p,q=get_image(train,j)
#    plot_im_bbox(p,q)
#    plt.title('actual_bb')
#    plt.axis('off')
#    plt.subplot(1,3,3)
#    plt.imshow(draw_img)
#    plt.title('predicted Bounding Box')
#    plt.axis('off');
#plt.show()