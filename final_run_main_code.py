# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 20:09:18 2017

@author: DSP Lab
"""

#import final_run_functions
import cv2
import numpy as np
import time
import os


#All weight path
#rain_denoiser_weight="D:\\weights\\weights_derain.hdf5"
#challenge_detector_weights='D:\\weights\\challenge_detector_weight_actual.h5'
#all_level_01_challenge_weightpath='D:\\weights\\all_level_01_challenge_weight.h5'
#noise_level_03_challenge_weightpath='D:\\weights\\noise_level_03_challenge_weight.h5'
#blur_level_03_challenge_weightpath='D:\\weights\\blur_level_03_challenge_weight.h5'
#classifier_weight="D:\\weights\\weights_classifier.hdf5"
#
##all model initialization
#challenge_detector=final_run_functions.challenge_detector_model()
#model_derain=final_run_functions.derain()
#no_noise_localizer=final_run_functions.localizer()
#noise_localizer=final_run_functions.localizer()
#blur_localizer=final_run_functions.localizer()
#classifier=final_run_functions.classifier()
#
##all model weigth load
#challenge_detector.load_weights(challenge_detector_weights)
#
#model_derain.load_weights(rain_denoiser_weight)
#
#no_noise_localizer.load_weights(all_level_01_challenge_weightpath)
#blur_localizer.load_weights(blur_level_03_challenge_weightpath)
#noise_localizer.load_weights(noise_level_03_challenge_weightpath)
#
#classifier.load_weights(classifier_weight)


vidreadpath='D:\\rain_baki\\' ##contains all the text video sequences
txtwritepath='D:\\outputlabel\\'

dest=os.listdir(vidreadpath) #list of all paths of the videos to be processed

##Uncomment the following section for only output text with generating TruePositive, FalsePositive, FalseNegative vallues"""
#txtreadpath='D:\\txtreadpy\\'  #contains all the test sequences' given annotation text files
#pr_text=open('D:\\all_precision_recall.txt', "w+") #creates a text file where all the Truepositive, Falsepositive, Falsenegative values for each sequences will be written
#pr_text.close()

for seq_num in range(len(dest)):

##Uncomment the following section for only output text with generating TruePositive, FalsePositive, FalseNegative vallues"""    
#    TruePositive, FalsePositive, FalseNegative = 0, 0, 0 #initializing values
#    pr_text=open('D:\\all_precision_recall.txt', "a") #text file for printing true +ve, false +ve, false -ve values  
    
    vidname=dest[seq_num][:14]
    cap=cv2.VideoCapture(vidreadpath+vidname+'.mp4') #creating video capture object
    f=open(txtwritepath+vidname+'.txt', "w+") #creating label file for video
    t=time.time()
    f.write('frameNumber_signType_llx_lly_lrx_lry_ulx_uly_urx_ury\n') #write first line

##Uncomment the following section for only output text with generating Tp,Fp,Fn vallues"""     

##opening ground truth label file
#    txt_file=txtreadpath+vidname[:5]+'.txt';
#    temp=loadtxt(txt_file, dtype="int64",delimiter="_", skiprows=1, usecols=(0,1,2,3,8,9))
    
    ret=1
    demo=np.zeros((1236,1628))
    original_mask=np.zeros((1236,1628))
    text_line=0
    noise=np.zeros((1,13))
    
    #for determining the type of challenge in the whole video
    #we check the first 15 frames then use the class with max occurances
    s=0
    flag=0
    
    while(ret):
        ret, frame=cap.read() #get new frame
        current_frame=cap.get(1) #get frame number
        
        if ret:
            
            #convert frame from BGR to RGB
            frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            if flag==0: #flag to indicate if 15 frames have been checked                
                chal=final_run_functions.out_challenge_type(frame,challenge_detector) #challenge detector(frame)
                noise[0,chal]=noise[0,chal]+1 #keeping track of what challenge types are detected
                                                #in the frames in the noise matrix
                s+=1 #frame count
            if s==15: #if frame count reaches 15
                chal=noise.argmax(axis=1); #challenge type of the video is the dominant
                                            #challenge type from the first 15 frames
                flag=1
            #print(chal)
            
            #preprocessing for the challenge types
            if chal==1:
                frame=final_run_functions.CLAHE_decolor(frame)
            elif chal==4:
                frame=final_run_functions.CLAHE_dark(frame)
            elif chal==9:
                frame=final_run_functions.rain_preprocess(frame,model_derain)
            elif chal==10:
                frame=final_run_functions.CLAHE_shadow(frame)
            elif chal==11:
                frame=final_run_functions.CLAHE_snow(frame)
            elif chal==12:
                frame=final_run_functions.CLAHE_haze(frame)
            elif chal==6:
                frame=final_run_functions.CLAHE_exposure(frame)
            
            #Localize
            if chal in [0, 1, 3, 4, 5, 6, 10, 9, 12, 11]:
                M=final_run_functions.localizer_output(no_noise_localizer,frame) #no challenge localizer
            
            elif chal==2:
                M=final_run_functions.localizer_output(blur_localizer,frame) #Blur
            
            elif chal==8:
                M=final_run_functions.localizer_output(noise_localizer,frame) #Noise

            # M contains co-ordinates of the sign-proposals
            M=M.astype('int') 

##output text without generating Tp,Fp,Fn vallues
            for k in range(0, len(M)):
                bb=frame[M[k, 1]: M[k, 3], M[k, 0]:M[k, 2], :]
                sign=final_run_functions.classifier_output(classifier,bb)
                if sign!=0:
                   f.write("%.3d_%.2d_%d_%d_%d_%d_%d_%d_%d_%d\n" %(cap.get(1),sign,M[k,0],M[k,1],M[k,2],M[k,1],M[k,0],M[k,3],M[k,2],M[k,3]))             
    f.close()
            
##Uncomment the following section for only output text with generating Tp,Fp,Fn vallues"""            
#            
#            ##Generating masks for the predicted bounding boxes
#            
#            #Initializing
#            if (len(M)!=0):
#                pred_mask=np.zeros((len(M),1236*1628))
#            else:
#                pred_mask=np.zeros((1,1236*1628))
#            
#            #creating mask for the i-th bounding box in M
#            for i in range(len(M)):
#                demo=demo*00
#                demo[M[i, 1]: M[i, 3], M[i, 0]:M[i, 2]]=1
#                pred_mask[i]=np.ravel(demo)
#
#            pred_mask=pred_mask.transpose()
#            
#            #no. of ground truth boxes in the current frame
#            count=np.count_nonzero(temp[:,0] == current_frame)
#            if count!=0:
#                #creating masks for ground-truth sign regions
#                original_mask=np.zeros((count,1236*1628))
#                for i in range(count):
#                    demo=demo*0
#                    demo[temp[text_line+i,3]:temp[text_line+i,5],temp[text_line+i,2]:temp[text_line+i,4]]=1
#                    original_mask[i]=np.ravel(demo)
#            else:
#                original_mask=np.zeros((1,1236*1628))
#                
#            result=np.matmul(original_mask,pred_mask) #multuplying ground-truth masks and prediction masks to determine sign-associations
#            #determining the area of ground-truth signs
#            area=np.reshape(np.sum(original_mask,axis=1),(len(original_mask),1))
#            
#                
#            g_ind,p_ind=np.nonzero(result)
#            if count!=0:
#                result=result/area #calculating overlap area
#            FalseNegative = FalseNegative + count-len(np.unique(g_ind)) #ground-truth signs that
#                                                                        #were not localized
#
#            for k in range(0, len(M)):
#                bb=frame[M[k, 1]: M[k, 3], M[k, 0]:M[k, 2], :]#extracting bounding box
#                sign=final_run_functions.classifier_output(classifier,bb)#classifying sign
#                
#                if k in p_ind: #checking if the k-th detected sign was one of the ground truth signs 
#                    ind = np.where(p_ind==k)[0][0] #getting the index of the corresponding ground-truth sign
#                    overlap=result[g_ind[ind],p_ind[ind]] #getting overlap area for the k-th detected sign
#                    g_sign=temp[text_line+g_ind[ind],1] #getting the sign type of the corresponding ground-truth sign
#                    
#                    if sign==0:#if actually a sign but not classsified as a sign region
#                        FalseNegative=FalseNegative+1
#
#                    elif g_sign!=sign:#if sign was not classified correctly
#                        FalsePositive=FalsePositive+1
#
#                    elif overlap<0.5:#if overlap area is less than 50%
#                        FalsePositive=FalsePositive+1
#
#                    elif overlap>0.5:#if none of the above is true
#                        TruePositive=TruePositive+1
#
#                else:
#                    if sign!=0: #if non-sign region was classified as a sign region
#                        FalsePositive=FalsePositive+1
#
#                if sign!=0:
#                    f.write("%.3d_%.2d_%d_%d_%d_%d_%d_%d_%d_%d\n" %(cap.get(1),sign,M[k,0],M[k,1],M[k,2],M[k,1],M[k,0],M[k,3],M[k,2],M[k,3]))
#        
#        text_line=text_line+count
#    
#    
#    f.close()
    print(t-time.time()) 
    
##Uncomment the following section for only output text with generating TruePositive, FalsePositive, FalseNegative vallues"""
#    pr_text.write("%s,%d,%d,%d\n" %(vidname, TruePositive, FalsePositive, FalseNegative))
#    pr_text.close()     