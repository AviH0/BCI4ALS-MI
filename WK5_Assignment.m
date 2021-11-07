% Week 5 Assignment

%% Add file paths
userName = 'Yarden'; % change to your username

addpath(genpath('C:\Users\userName\Documents\MATLAB\liblsl-Matlab'))
addpath(genpath('C:\Toolboxes'))
addpath(genpath('C:\Users\userName\Documents\MATLAB\BCI4ALS-MI'))

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
