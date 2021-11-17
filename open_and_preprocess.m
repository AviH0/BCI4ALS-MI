function [EEG, clean_data] = open_and_preprocess(subjectNumber)

% THIS FUNCTION READS RAW SUBJECT DATA, CHECKS IF THERE IS A CLEAN FILE AND
% RUNS THE MI2 FUNCTION IG THE CLEAN FILE DOES NOT EXIST
% INPUT:
% subjectNumber = number of subject as written at time of recording
% OUTPUT:
% EEG = raw data
% clean_data = preprocessed data (using MI2_preprocessing)

mainFolder = 'C:\Recordings\'; %change for each subject

if isstring(subjectNumber) == 0 % check if subject number is string
    subjectNumber = string(subjectNumber);
else 
    
end

recordingFolder = strcat(mainFolder,subjectNumber); % subject's recording folder
recordingFile = strcat(mainFolder,subjectNumber,'\EEG.XDF'); %raw subject data
% read raw XDF file from folder
EEG = pop_loadxdf(recordingFile, 'streamtype', 'EEG', 'exclude_markerstreams', {});

fullCleanFile = strcat(mainFolder,subjectNumber, "\cleaned_sub.mat");

% check if a cleaned file already exists. If not, run MI2 to preprocess
if exist(fullCleanFile, 'file') == 2
     % File exists. Load it.
     clean_data = load(strcat(recordingFolder, "\cleaned_sub.mat"));
else
     % File does not exist. Run MI2.
     clean_data = MI2_preprocess(recordingFolder);
end


end
