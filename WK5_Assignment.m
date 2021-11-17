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

%% Define subject ID
subjectID = "1_20211104"; % as entered at time of recording

%% read data, output raw and preprocessed files
[EEG_raw, EEG_clean] = open_and_preprocess(subjectID);

%% plot raw data x voltage
% eegplot(EEG.data, 'srate', EEG.srate, 'eloc_file', strcat(recordingFolder, '\EEG_chans.mat'))   
%% PSD
pop_spectopo(EEG_raw)

%% LaPlacian on clean data file
clean_data_mat = EEG_clean.data;
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
srate = EEG_clean.srate; % sampling rate
total_ms = num_samples / srate * 1000;
timeVec = 0:8:total_ms-1;

% plot laPlacian before and after
plot(timeVec,c4, 'LineWidth', 2);
hold on;
plot(timeVec,lap_c4,'LineWidth', 2);
xlabel('Time (ms)'); ylabel('Amplitude (mV)')
title('C4 before and after Spatial Laplacian');
legend('c4', 'c4 - after laplacian');
xlim([0, 1500]);

%% plot cleaned and uncleaned data

% plot raw data
eegplot(EEG_raw.data)
pop_spectopo(EEG_raw)

% plot cleaned data
eegplot(EEG_clean.data)
pop_spectopo(EEG_clean)



