% Week 5 Assignment

%% Add file paths

addpath(genpath('C:\Toolboxes'))
addpath(genpath('C:\GitHub\BCI4ALS-MI'))
addpath(genpath('C:\Toolboxes\liblsl-Matlab'))
addpath(genpath('C:\Recordings'))

%% Collect data
%lib = lsl_loadlib(); version = lsl_library_version(lib);
%eeglab
%MI1_offline_training();

%% define subject folder
recordingFolder = 'C:\Recordings\1_20211104'; %change for each subject

%% read file

recordingFile = strcat(recordingFolder,'\EEG.XDF');

EEG = pop_loadxdf(recordingFile, 'streamtype', 'EEG', 'exclude_markerstreams', {});
EEG_chans = load(strcat(recordingFolder, "\EEG_chans.mat"));
for i = 1: length(EEG_chans.EEG_chans)
    EEG.chanlocs(i).label = EEG_chans.EEG_chans(i);
end

% check if a cleaned file already exists. If not, run MI2 to preprocess
if exist(recordingFile, 'file') == 2
     % File exists. Load it.
     clean_data = load(strcat(recordingFolder, "\cleaned_sub.mat"));
else
     % File does not exist. Run MI2.
     clean_data = MI2_preprocess(recordingFolder);
end

%% plot raw data x voltage
% eegplot(EEG.data, 'srate', EEG.srate, 'eloc_file', strcat(recordingFolder, '\EEG_chans.mat'))   
%% PSD
pop_spectopo(EEG)

%% LaPlacian on clean data file
clean_data_mat = cell2mat(struct2cell(clean_data));
% probelm with importing channel labels; import manually
% maybe problem in channel 1 (and no channel 4 for sure)
% so no c3 at the moment

c4_num = 2; % channel number
four_close_c4 = [5 7 9 11]; % four electrodes closest to electrode C4
% spatial laplacian- subtracting from channel c4 the mean of the
% closest channels
c4 = clean_data_mat(c4_num,:); % data from electrode C4 only
lap_c4 = c4 - mean(clean_data_mat(four_close_c4,:));

% convert sampling to time
num_samples = length(c4); % number of samples
srate = EEG.srate; % sampling rate
total_ms = num_samples / srate * 1000;

% plot laPlacian before and after
plot(0:(length(c4)-1),c4, 'LineWidth', 2);
hold on;
plot(0:(length(c4)-1),lap_c4,'LineWidth', 2);
xlabel('Samples'); ylabel('Amplitude (mV)')
title('C4 before and after Spatial Laplacian');
legend('c4', 'c4 - after laplacian');
xlim([0, 1000]);

%% plot cleaned and uncleaned data

% plot raw data
eegplot(data.EEG_data)
pop_spectopo(EEG)

% plot cleaned data
eegplot(data.clean_data)
pop_spectopo(clean_data)



