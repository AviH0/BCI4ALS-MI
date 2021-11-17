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
[EEG, clean_data] = open_and_preprocess(subjectID);

%% plot raw data x voltage
eegplot(EEG.data)   
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
eegplot(EEG.data)
pop_spectopo(EEG)

% plot cleaned data
eegplot(clean_data.data)
pop_spectopo(clean_data)



