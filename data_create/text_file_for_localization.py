# -*- coding: utf-8 -*-
"""
Created on Wed May 31 01:42:55 2017

@author: udaykamal
"""
"""
creates a text file to be used for creating localizer training dataframe
the lines in the text file has the following format:
    "filepath","xmin","ymin","xmax","ymax"
for each frame of no challenge sequences containing a single sign.
    
"""
#import csv
import numpy as np
from os import listdir
from numpy import loadtxt 



#imreadpath='D:\\imwritepy\\'
imreadpath='G:\\05_03\\'
annotationpath='D:\\labels2\\'
textsavepath='D:\\text\\train_with_sign.txt'

add=listdir(annotationpath);


char=np.array(np.zeros(6,dtype=np.int),dtype='<U100')

for x in range(0, len(add)):
    temp=[];
    txt_file=[];
    file_name=[];
    char=[];
    temp2=[];
    txt_file = annotationpath+add[x];
    file_name=add[x][:add[x].rfind(".")]
    temp=loadtxt(txt_file, dtype="int64",delimiter="_", skiprows=1, usecols=(0,2,3,8,9))
    demo=np.char.mod('%d', temp)
    char=np.array(demo,dtype='<U100');
    for y in range(0,temp.shape[0]):
        char[y,0]=imreadpath+file_name+'_00_00_00_'+char[y,0]+'.jpg'
    with open(textsavepath,'ab')as f:
        np.savetxt(f,char,fmt='%s',delimiter=',',newline='\n')

