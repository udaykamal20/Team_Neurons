# VIP_CUP_2017_Team_Neurons-Traffic_Sign_Detection_Under_Challenging_Condition
This respiratory contains the codes and submission package of the Team Neurons, the champion team of IEEE Video and Image Processing Cup 2017 organized by Olives - Omni Lab for Intelligent Visual Engineering and Science from Georgia Tech. The task was to come up with a traffic sign detection algorithm that is robust even under the challenging weather condition. 

The competition data can be downloaded from the following link: 
https://ieee-dataport.org/open-access/cure-tsd-challenging-unreal-and-real-environment-traffic-sign-detection.

Details of the competition can be found in the following link:
https://ghassanalregibdotcom.wordpress.com/vip-cup/.

The details of our approach and performance can be found the 'Report.pdf' file. 

The following procedures are to be followed to reporduce the submitted results:


# Choose a directory and create the following subdirectories:

	* vidreadpy ""all the training video sequences will be placed here
	* labels ""all the given training annotation text files will be placed here
	* testvid ""all the test video seequences will be placed here
	* imwritepy ""all the extracted frames will be saved here
	* matwritepy ""all the ROI matrix and corresponding label matrix will be saved here
	* weights ""all the trained weights will be saved here
	* dataframe ""different dataframes required for various models' training, will be placed here
	* text ""a required text file for creating the dataframes will be saved here
	* outputlabel ""all the test sequences' output text files will be saved here


# Frame extractions:

1) First all the training video sequences are to be placed in vidreadpy subdirectory

2) All the given training annotation text files are to be placed in labels subdirectory

3) Then "frame_and_ROI_extractor.py" code is to be run. here all the ROI containing unique frames from all challenge type and levels will be extracted and saved to the imwritepy subdirectory. Also the ROI regions and their corresponding labels (from no challenge, level 01, and 03 of all challenge type) will be saved as two different matrixes in matwritepy subdirectory.


# Cahllenge detector model training dataframe create:
To create the necessary dataframes for training the challenge detector model, "challenge_detector_model_dataframe_create.py" code is needed to be run. here we create a dataframe containing the path for no challenge frames and different challenge types level 03 frames (for dirty lens both level 3 and level 5 frames) and their corresponding classes(each class contains 2000 random samples). We considered 02 and 07 no challenge type as a unified challenge namely 'blur'(class 02). The created dataframe will be saved in dataframe subdirectory


# Text file to be used for localizer model training dataframe creation:
Text_file_for_localization.py" code is to be run to create a text file with each line containing the no challenge ROI containing frames' path and the corresponding ROI's xmin,ymin,xmax,ymax coordinates. This will be saved in text subdirectory.


# Localizer model training dataframe creation:
"Data_Frame_Create.py" code is needed to be run to create all the necessary dataframes for localizer training. Total 3types of dataframes will be created. all_level_01_challenge,noise_level_03_challenge and blur_level_03_challenge. In the code, while creating a specific dataframe, only the corresponding boolian variable is needed to be set 'True' and all others to 'False'. For example to create noise_level_03_challenge dataframe, noise_level_03_challenge=True,all_level_01_challenge=False,blur_level_03_challenge=False. Thus this code will be run 3 total 3 times.

# Derain model training dataframe creation:
"derain_model_data_create.py" file is needed to be run to create a dataframe that will be used for derain model training.

# Challenge detector model train:
"challenge_detector_model_train.py" file is needed to be run to train the challenge detector model using the previously saved challenge detector model training dataframe. Frames are resized to 309,407 pixels for training. the model is trained in total 20 epochs. where Among total 24000 samples, 23500 frames are used for training and remaining 500 frames are used for validation. Weights will be saved to the weights subdirectory.

# Derain model Train:
"derain_model_data_create.py" file is to be run to train the derain model using the previously created derain model training dataframe. Weights will be saved to the weights subdirectory.


# Localizer model Train:
Total 4 types of localizer are to be trained. "localizer_train.py" file is needed to be run for this.first a localizer model will be trained with no challenge frames and it's weights will be saved. the variables will be no_challenge=True, all_level_01_challenge=False, noise_level_03_challenge=False, blur_level_03_challenge=False as before. Then using the pretrained weigths, another localizer will be trained using the previously created all_level_01_challenge dataframe. This will be the final model for common localizer. Then in the similar way, using the pretrained weights(common localizer), noise_level_03 and blur_level_03 model will be trained. 

** for each model training, only the corresponding variable(no_challenge, all_level_01_challenge, noise_level_03_challenge, blur_level_03_challenge) is needed to be set True and all other will be set False. **

# Classifier Model Train:
"sign_classifier_train.py" file is needed to be run for sign classifier model training. The training data will be the previously saved ROI and label matrix. Trained weights will be saved into weights subdirectory

# Final output:
"final_run_main_code.py" file is needed to be run for the final test sequences' output text file generation. This code will use the functions from "final_run_functions.py" file. The output text file annotation format is produced as instruced in the problem statement. 

***Please uncomment the mentioned sections in the code to produce the output text file with generating another text file containing all the sequences' corresponding ...Truepositive, Falsepositive, and Falsenegative values(which have been calculated as instructed in the problem statement)***


