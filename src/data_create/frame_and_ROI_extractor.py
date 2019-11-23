# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 16:10:24 2017

@author: User
"""

import glob, cv2, time
import numpy as np
from scipy import misc

"""
This code is used for bounding box and frame extraction from given video sequences.
all the extracted bounding boxes are then first resized into 32x32 size and then
appended in a matrix format along with their corresponding sign type. 
And for the extracted frames(if needed), they are saved in the imwritepath folder. sequences of
specific level and specific challenge type can be used for extraction by setting the variables
'chal_type' and 'level' variable. 
 
Frames are saved in the imwritepath directory with a naming convention:
    'sequenceType_sequenceNumber_challengeSourceType_challengeType_challengeLevel_framenumber.jpg'
    
With this code, we extract all the unique frames(containing the ROI) of the training sequences(both real and synthesized)
for localizer, challenge detector and rain denoiser model training and store the ROI and corresponding labels of only
no challenge, level 1 & 3 of all challenge types for traffic sign classifier model training.

In the vidreadpath, all the TRAINING sequences (both real and synthesized) of
all types of challenges and levels are to be placed.

"""

bbtextpath='D:\\labels\\' ##file path for provided annotation text files
vidreadpath='D:\\vidreadpy\\' ##path for reading video files
imwritepath='D:\\imwritepy\\' ##extracted image write path
matwritepath='D:\\matwritepy\\' ##extracted bounding boxe's matrix write file

dataROI=np.empty((1, 32, 32, 3), dtype='uint8')
dataLABEL=np.zeros((1))

#textfile path
chal_type='*' ## setting this variable to specific number will lead to the extraction of only that type of noise sequence. '*' meanse 
##it will extract frames from all types of challenge sequences

level= np.array([0,1,3]) ## this vairable controls the challenge levels of the frames to be extracted.

"""
for example if chal_type='*' and level='05' then this code will extract level 5 of all challenge types at once.
"""
dertxt=glob.glob(bbtextpath+'*.txt') ##annotation text files for all the frames containing the sign only with their bounding box coordinates

#Dataset Loop
for i in range (len(dertxt)):
    
    start=time.time()
    seqROI=np.empty((1, 32, 32, 3), dtype='uint8') #initializing a matrix for storing all bounding boxes from all the videos of the ith sequences
    seqLABEL=np.empty((1)) # initializing a column vector for storing bounding boxes labels from all the videos of the ith sequences
    
    
    #bounding box co-ordinates
    bb = np.loadtxt(dertxt[i], dtype=int, delimiter='_', skiprows=1)
    
    if len(bb)!=0:
        #video directory
        dervid=glob.glob(vidreadpath+dertxt[i][-9:-4]+'_'+ chal_type+'_*_*.mp4')
        
        #Bounding Box Loop/Sequence Loop
        for j in range (len(dervid)):
            
            #Challenge level checker
            if int(dervid[j][-6:-4]) in level:    # We extract both frames, ROI & corresponding labels if the callenge level is 0 or 1 or 3 
                #access video
                cap = cv2.VideoCapture(dervid[j])
                cap.set(1,299)      ##just to make sure that the video can be read upto the last frame
                ret, frame=cap.read()
                if ret:
                    vidROI=np.empty((len(bb), 32, 32, 3), dtype='uint8') #initiallizing a matrix for boundingboxes of jth video from ith sequence
                    vidLABEL=np.empty((len(bb)))  #initiallizing a matrix for boundingboxes labels of jth video from ith sequence
                    
                    #Video Loop
                    ##Region Extractor
                    #ROI Loop Starter
                    k=0  #for the first frame
                    cap.set(1, bb[k,0]-1)
                    ret, frame = cap.read()
                    cv2.imwrite(imwritepath+dervid[j][-18:-4]+'_'+str(bb[k,0])+'.jpg', frame)
                    vidROI[k, :, :, :]=misc.imresize(frame[bb[k,3]:bb[k,7]+1, bb[k,2]:bb[k,4]+1, [2,1,0]], ## resized ROI extraction
                                                      (32,32))
                    vidLABEL[k]=bb[k,1]
                    
                    for k in range(1, len(bb)):
                        
                        if bb[k,0]!=bb[k-1,0]: #for extracting the unique frames only
                            
                            #accessing the frame with the corresponding bounding box
                            cap.set(1, bb[k,0]-1)
                            ret, frame = cap.read()
                            #Extracting Image
                            cv2.imwrite(imwritepath+dervid[j][-18:-4]+'_'+str(bb[k,0])+'.jpg', frame)                       
                            #Extracting. resozing and Storing ROI
                            vidROI[k, :, :, :]=misc.imresize(frame[bb[k,3]:bb[k,7]+1, bb[k,2]:bb[k,4]+1, [2,1,0]],
                                                    (32,32))
                            #Storing label
                            vidLABEL[k]=bb[k,1]
                            
                        else :
                            #only storing ROI and labels
                            vidROI[k, :, :, :]=misc.imresize(frame[bb[k,3]:bb[k,7]+1, bb[k,2]:bb[k,4]+1, [2,1,0]],
                                                      (32,32))
                            vidLABEL[k]=bb[k,1]
                
                    cap.release()
                    #storing video ROI and labels in a larger matrix for the sequence-type
                    seqROI=np.append(seqROI, vidROI, axis=0)
                    seqLABEL=np.append(seqLABEL, vidLABEL, axis=0)
                
            else: # We extract only the frames for rest of the challenge levels
                cap = cv2.VideoCapture(dervid[j])
                cap.set(1,299)
                ret, frame=cap.read()
                if ret:
                    k=0
                    cap.set(1, bb[k,0]-1)
                    ret, frame = cap.read()
                    cv2.imwrite(imwritepath+dervid[j][-18:-4]+'_'+str(bb[k,0])+'.jpg', frame)
                    
                    for k in range(1, len(bb)):
                        
                        if bb[k,0]!=bb[k-1,0]:
                            
                            cap.set(1, bb[k,0]-1)
                            ret, frame = cap.read()
                            #Extracting Image
                            cv2.imwrite(imwritepath+dervid[j][-18:-4]+'_'+str(bb[k,0])+'.jpg', frame)
                            
                    cap.release() #releasing video
        
        #storing ROIs and labels of all video sequences
        dataROI=np.append(dataROI, seqROI[1:,:,:,:], axis=0) # We reject the first element as it contains garbage value
        dataLABEL=np.append(dataLABEL, seqLABEL[1:], axis=0)
            
        print(time.time()-start)

#writing the all the ROIs in a file
dataROI[1:,:,:,:].dump(matwritepath+'allbb') # We reject the first element as it contains garbage value
dataLABEL[1:].dump(matwritepath+'alllabel')
