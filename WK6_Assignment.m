% Week 6 Assignment

%% Add file paths

addpath(genpath('C:\Toolboxes'))
addpath(genpath('C:\GitHub\BCI4ALS-MI'))
addpath(genpath('C:\Toolboxes\liblsl-Matlab'))
addpath(genpath('C:\Recordings'))

%% Define subject ID
subjectID = "SubTS2021-11-14"; % as entered at time of recording

%% read data, output raw and preprocessed files
[EEG_raw, EEG_clean] = open_and_preprocess(subjectID);

%% 2
recording_folder = strcat("C:/Recordings/",subjectID);
cleanFile = '/cleaned_sub.mat';
MI3_segmentation(recording_folder, cleanFile);
% a
MI4_featureExtraction(recording_folder)

%% b
load('EEG_chans.mat');
[bef_dat,aft_dat]=LaPlacian(EEG_clean.data, ['C04','C03'], ...
    [5:2:11; 4:2:10], EEG_chans);

EEG_data=aft_dat;
% save laplace file to recording folder
save(fullfile(recording_folder, 'cleaned_sub_lap.mat'), 'EEG_data', '-mat'); 
cleanFile = '/cleaned_sub_lap.mat';

MI3_segmentation(recording_folder, cleanFile)
MI4_featureExtraction(recording_folder)
