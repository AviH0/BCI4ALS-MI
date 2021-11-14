% Week 5 Assignment

%% Add file paths

%addpath(genpath('C:\Toolboxes'))
addpath(genpath('C:\GitHub\BCI4ALS-MI'))

%% Collect data
%lib = lsl_loadlib(); version = lsl_library_version(lib);
%eeglab
%MI1_offline_training();

%% define subject folder
recordingFolder = 'C:\Recordings\2_20211107'; %change for each subject

%% read file

recordingFile = strcat(recordingFolder,'\EEG.XDF');
%EEG = pop_loadxdf(recordingFile, 'streamtype', 'EEG', 'exclude_markerstreams', {});
data = load(strcat(recordingFolder, "\cleaned_sub.mat"));
%%%%


%% plot raw data x voltage
eegplot(data.EEG_data)
%% PSD
pop_spectopo(EEG)