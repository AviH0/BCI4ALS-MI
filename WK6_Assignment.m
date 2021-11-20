% Week 6 Assignment

%% Add file paths

addpath(genpath('C:\Toolboxes'))
addpath(genpath('C:\GitHub\BCI4ALS-MI'))
addpath(genpath('C:\Toolboxes\liblsl-Matlab'))
addpath(genpath('C:\Recordings'))

%% Define subject ID
subjectID = "2021-11-14"; % as entered at time of recording

%% read data, output raw and preprocessed files
[EEG_raw, EEG_clean] = open_and_preprocess(subjectID);

%% 2
recording_folder=strcat("C:\Recordings\",subjectID,'\');
MI3_segmentation(recording_folder);
% a
MI4_featureExtraction(recording_folder)

%% b
load('EEG_chans.mat');
[bef_dat,aft_dat]=LaPlacian(EEG_clean.data, ['C04','C03'], ...
    [5:2:11; 4:2:10], EEG_chans);

folder_lap=strcat(recording_folder,'Lap\'); %Folder for data after Laplacian
EEG_data=aft_dat;
save(fullfile(folder_lap, 'cleaned_sub.mat'), 'EEG_data', '-mat');
    %after copying EEG_events, EEG_chans & trainingVec to the lap folder(!)
MI3_segmentation(folder_lap)
MI4_featureExtraction(folder_lap)
