# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 20:16:39 2017

@author: udaykamal
"""

"""
for creating the dataframe to be used for localizer training
creates a dataframe for all challenge sequence(of any desired level) frames
using the previously created text file

"""
import pandas as pd
import numpy as np


all_level_01_challenge=True
noise_level_03_challenge=False
blur_level_03_challenge=False

imreadpath='D:\\imwritepy\\'
input_train_path='D:\\text\\train_with_sign.txt'
df_all_01_save_path='D:\\dataframe\\localizer_all_challenge_level_01_data.pkl' #datapath for all level 01 challenge
df_noise_03_save_path='D:\\dataframe\\localizer_noise_level_01_data.pkl' #datapath for noise level 03 challenge
df_blur_03_save_path='D:\\dataframe\\localizer_blur_level_01_data.pkl'#datapath for blur level 03 challenge

df = pd.read_csv(input_train_path, sep=",", names=["filepath", "xmin", "ymin","xmax","ymax"])

if all_level_01_challenge==True:
    challenge_type=np.array(['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'])
    seq_type='01'
    chal_level='01'
    df_save_path=df_all_01_save_path
    
elif noise_level_03_challenge==True:
    challenge_type=np.array(['08'])
    seq_type='01'
    chal_level='03'
    df_save_path=df_noise_03_save_path
    
elif blur_level_03_challenge==True:
    challenge_type=np.array(['07'])
    seq_type='01'
    chal_level='03'
    df_save_path=df_blur_03_save_path


dirc=[]
file=df['filepath']

df_save = pd.DataFrame()
for j in range (len(challenge_type)):
    
    chal_type=challenge_type[j]
    for i in range(len(file)):
        q=file[i]
        p=q.split('\\')
        r=p[2].split('_')
        s=imreadpath+r[0]+'_'+r[1]+'_'+seq_type+'_'+chal_type+'_'+chal_level+'_'+r[5]
        dirc.append(s)
    df['filepath']=dirc
    df_save=df_save.append(df,ignore_index=True) 
  
df_save=df_save.sample(frac=1,random_state=200) #shuffling the whole dataframe
df_save.to_pickle(df_save_path) #saving the dataframe

