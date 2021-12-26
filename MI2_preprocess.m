function [EEG] = MI2_preprocess(recordingFolder)
%% Offline Preprocessing
% Assumes recorded using Lab Recorder.
% Make sure you have EEGLAB installed with ERPLAB & loadXDF plugins.

% [recordingFolder] - where the EEG (data & meta-data) are stored.

% Preprocessing using EEGLAB function.
% 1. load XDF file (Lab Recorder LSL output)
% 2. look up channel names - YOU NEED TO UPDATE THIS
% 3. filter data above 0.5 & below 40 Hz
% 4. notch filter @ 50 Hz
% 5. advanced artifact removal (ICA/ASR/Cleanline...) - EEGLAB functionality

%% This code is part of the BCI-4-ALS Course written by Asaf Harel
% (harelasa@post.bgu.ac.il) in 2021. You are free to use, change, adapt and
% so on - but please cite properly if published.

%% Some parameters (this needs to change according to your system):
addpath 'C:\Toolboxes\eeglab2021.1'                     % update to your own computer path
eeglab;                                     % open EEGLAB 
highLim = 40;                               % filter data under 40 Hz
lowLim = 0.5;                               % filter data above 0.5 Hz
recordingFile = strcat(recordingFolder,'/EEG.XDF');

% (1) Load subject data (assume XDF)
EEG = pop_loadxdf(recordingFile, 'streamtype', 'EEG', 'exclude_markerstreams', {});
EEG.setname = 'MI_sub';

% (2) Update channel names - each group should update this according to
% their own openBCI setup.
EEG_chans(1,:) = 'C03';
EEG_chans(2,:) = 'C04';
EEG_chans(3,:) = 'Cz0';
EEG_chans(4,:) = 'Fc1';
EEG_chans(5,:) = 'Fc2';
EEG_chans(6,:) = 'Fc5';
EEG_chans(7,:) = 'Fc6';
EEG_chans(8,:) = 'Cp1';
EEG_chans(9,:) = 'Cp2';
EEG_chans(10,:) = 'Cp5';
EEG_chans(11,:) = 'Cp6';
EEG_chans(12,:) = 'O01';
EEG_chans(13,:) = 'O02';
EEG_chans(14,:) = 'P03';
EEG_chans(15,:) = 'P03';
EEG_chans(16,:) = 'P03';

%% (3) Low-pass filter
EEG = pop_eegfiltnew(EEG, 'hicutoff',highLim,'plotfreqz',1);    % remove data above
EEG = eeg_checkset( EEG );
% (3) High-pass filter
EEG = pop_eegfiltnew(EEG, 'locutoff',lowLim,'plotfreqz',1);     % remove data under
EEG = eeg_checkset( EEG );
% (4) Notch filter - this uses the ERPLAB filter
EEG  = pop_basicfilter( EEG,  1:15 , 'Boundary', 'boundary', 'Cutoff',  50, 'Design', 'notch', 'Filter', 'PMnotch', 'Order',  180 );
EEG = eeg_checkset( EEG );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%% (5) Add advanced artifact removal functions %%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
c4_num = 2; % channel number
four_close_c4 = [5 7 9 11]; % four electrodes closest to electrode C4
% spatial laplacian- subtracting from channel c4 the mean of the
% closest channels
c4 = EEG.data(c4_num,:); % data from electrode C4 only
lap_c4 = c4 - mean(EEG.data(four_close_c4,:));

c3_num = 1; % channel number
four_close_c3 = [4 6 8 10]; % four electrodes closest to electrode C4
% spatial laplacian- subtracting from channel c4 the mean of the
% closest channels
c3 = EEG.data(c3_num,:); % data from electrode C4 only
lap_c3 = c3 - mean(EEG.data(four_close_c3,:));

EEG.data(c3_num,:) = lap_c3;
EEG.data(c4_num,:) = lap_c4;

% Save the data into .mat variables on the computer
EEG_data = EEG.data;            % Pre-processed EEG data
EEG_event = EEG.event;          % Saved markers for sorting the data
save(strcat(recordingFolder,'/','cleaned_sub.mat'),'EEG_data');
save(strcat(recordingFolder,'/','EEG_events.mat'),'EEG_event');
save(strcat(recordingFolder,'/','EEG_chans.mat'),'EEG_chans');

end
