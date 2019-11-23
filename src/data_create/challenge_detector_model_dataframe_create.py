# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 20:16:39 2017

@author: udaykamal
"""

"""
code for creating dataframe for challenge classifier model training
here we label challenges as below: (left=challenge type, right=class map)

    00-0
    01-1
    02+07-2
    03-3
    04-4
    05-5
    06-6
    08-7
    09-8
    10-9
    11-10
    12-11

we choose random 2000 frames from every challenge type. total=24000 samples of 12 class 
"""
import pandas as pd
import numpy as np
import glob

imreadpath='D:\\imwritepy\\'
df_save_path='D:\\dataframe\\challenge_detector_data.pkl'

challenge_type=np.array(['00','01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'])

df_all_img = pd.DataFrame()

for i in range (len(challenge_type)):
    if i==0:
        all_img=glob.glob(imreadpath+'*_*_00_00_00_*.jpg') #no challenge type
    else:
        all_img=glob.glob(imreadpath+'*_*_01_' + challenge_type[i] + '_03_*.jpg') #all other challenge type of level 3
    
    if i==5:
        all_img=all_img.append(glob.glob(imreadpath+'*_*_01_' + challenge_type[i] + '_05_*.jpg')) #for challenge type 5, we add level 05 samples as they are different than other levels
    
    if i==7: # as we considered 07 and 02 challenge type as one unified class labelled as 2
        lb=2
        
    elif i>7:
        lb=i-1
        
    else:
        lb=i
        
    all_label=np.full(len(all_img),lb).tolist() #creating the corresponding labels
    
    data = pd.DataFrame({'filepath': all_img, 'ch_type': all_label})
    data=data.sample(frac=2000/len(data)).reset_index(drop=True) #selecting random 2000 samples
    df_all_img=df_all_img.append(data,ignore_index=True)

df_all_img=df_all_img.sample(frac=1,random_state=200) #shuffling the whole dataframe
df_all_img.to_pickle(df_save_path) #saving the dataframe
