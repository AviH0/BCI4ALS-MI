% Week 6 Assignment

%% Add file paths

addpath(genpath('C:\Toolboxes'))
addpath(genpath('C:\GitHub\BCI4ALS-MI'))
addpath(genpath('C:\Toolboxes\liblsl-Matlab'))
addpath(genpath('C:\Recordings'))

%% Define subject ID
subjectID = "1_20211104"; % as entered at time of recording

%% read data, output raw and preprocessed files
[EEG_raw, EEG_clean] = open_and_preprocess(subjectID);