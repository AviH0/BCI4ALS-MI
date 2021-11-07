% Week 5 Assignment

%% Add file paths
addpath(genpath('C:\Users\Galina\Documents\MATLAB\liblsl-Matlab'))
addpath(genpath('C:\Toolboxes'))
addpath(genpath('C:\Users\Galina\Documents\MATLAB\BCI4ALS-MI'))
lib = lsl_loadlib(); version = lsl_library_version(lib);

fileFolder = 'C:\Recordings\1_20211104';
%% read file
MI2_preprocess(fileFolder);

%% plot raw data x voltage

%% PSD
