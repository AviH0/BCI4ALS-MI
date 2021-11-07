% Week 5 Assignment

%% Add file paths

addpath(genpath('C:\Toolboxes'))
addpath(genpath('C:\GitHub\BCI4ALS-MI'))

%% Collect data
lib = lsl_loadlib(); version = lsl_library_version(lib);
eeglab
MI1_offline_training();

%% define subject folder
fileFolder = 'C:\Recordings\1_20211104'; %change for each subject

%% read file
MI2_preprocess(fileFolder);
%%%%


x=5;
%% plot raw data x voltage

%% PSD
