# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 21:53:53 2017

@author: udaykamal
"""

"""
creating datapath for deraining model training
we use random 5000 level 01 rain frames as X_data
and corresponding clear frames as Y_data

"""

import pandas as pd
import glob

rain_imreadpath='D:\\imwritepy\\*_*_01_09_01_*'
no_chal_imreadpath='D:\\imwritepy\\*_*_00_00_00_*'
df_rain_save_path='D:\\dataframe\\derain_level_01_data.pkl'

x_dir=glob.glob(rain_imreadpath)
y_dir=glob.glob(no_chal_imreadpath)

df_rain=pd.DataFrame({'rain': x_dir,'clear': y_dir})
df_rain=df_rain.sample(frac=1,random_state=200)
df_rain.drop(df_rain.index[5000:],inplace=True)
df_rain.to_pickle(df_rain_save_path)

