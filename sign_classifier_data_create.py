#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 21:13:34 2018

@author: root
"""

filepath = '/media/uday/Simulation/deep learning competitions/VIP CUP/Data/ROIs/all/'

import os
import glob
import pandas as pd
import numpy as np
from keras.utils import np_utils

impath = []
imlabel = []
folder = os.listdir(filepath)
for folders in folder:
    path = glob.glob(filepath+folders+'/*jpg')
    if len(path)!=0:
        impath += path
        b=pd.read_csv(filepath + folders + '/Ytrain.csv',dtype= int ,skiprows =0,usecols=(1,), names ='b')
        imlabel += (list(b['b']))
    
df = pd.DataFrame(list(zip(impath, imlabel)),
              columns=['impath', 'imlabel'])

imlabel = np.asarray(imlabel)

Y_train = np_utils.to_categorical(imlabel-1, 14)

imlabel = list(Y_train)

import random
combined = list(zip(impath, imlabel))
random.shuffle(combined)

impath[:], imlabel[:] = zip(*combined)

df = pd.DataFrame(list(zip(impath, imlabel)),
              columns=['impath', 'imlabel'])