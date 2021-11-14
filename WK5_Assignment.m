% Week 5 Assignment

%% Add file paths

%addpath(genpath('C:\Toolboxes'))
addpath(genpath('C:\GitHub\BCI4ALS-MI'))
addpath(genpath('C:\Toolboxes\liblsl-Matlab'))
addpath(genpath('C:\Recordings'))


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
x=5;


%% LaPlacian - YARDEN
fileFolder = 'C:\Users\Yarden\Google Drive\University\ELSC\Year 2\BCI\recordings\2_20211107'; %change for each subject

% import data_eeg
struct_data=load(strcat(fileFolder,'\cleaned_sub.mat'));
data_eeg = cell2mat(struct2cell(struct_data));
%probelm with importing channels' labels, so manually
% maybe problem in channel 1 (and no channel 4 for sure)
% so no c3 at the moment
c4_num=2;
four_close_c4=[5 7 9 11];
% spatial laplacian- subtracting from channel c4 the mean of the
% closest channels
c4=data_eeg(c4_num,:);
lap_c4=c4 - mean(data_eeg(four_close_c4,:));
plot(0:(length(c4)-1),c4);
hold on;
plot(0:(length(c4)-1),lap_c4);
xlabel('time (ms)'); ylabel('mv')
title('C4 before and after Spatial Laplacian');
legend('c4', 'c4- after laplacian');

