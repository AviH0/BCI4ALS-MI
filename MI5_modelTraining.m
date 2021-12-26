function [test_results, fig] = MI5_modelTraining(recordingFolder)
% MI5_LearnModel_Scaffolding outputs a weight vector for all the features
% using a simple multi-class linear approach.
% Add your own classifier (SVM, CSP, DL, CONV, Riemann...), and make sure
% to add an accuracy test.

%% This code is part of the BCI-4-ALS Course written by Asaf Harel
% (harelasa@post.bgu.ac.il) in 2021. You are free to use, change, adapt and
% so on - but please cite properly if published.

%% Read the features & labels 

FeaturesTrain = cell2mat(struct2cell(load(strcat(recordingFolder,'\FeaturesTrainSelected.mat'))));   % features for train set
LabelTrain = cell2mat(struct2cell(load(strcat(recordingFolder,'\LabelTrain'))));                % label vector for train set

validIdx = randperm(length(FeaturesTrain), floor(0.2*length(FeaturesTrain))); % permutes indices of all training features, takes 20% of samples

% divide into training and validation(test) 
FeaturesValidation = FeaturesTrain(validIdx,:); 
LabelValidation = LabelTrain(validIdx);

FeaturesTrain(validIdx, :) = [];
LabelTrain(validIdx) = [];

% label vector
LabelTest = cell2mat(struct2cell(load(strcat(recordingFolder,'\LabelTest'))));              % label vector for test set
FeaturesTest = cell2mat(struct2cell(load(strcat(recordingFolder,'\FeaturesTest.mat'))));    % features for test set

%% test data - LDA

% DAclass = classify(FeaturesValidation,FeaturesTrain,LabelTrain);                    % classify the test set using a linear classification object (built-in Matlab functionality)

W = LDA(FeaturesTrain,LabelTrain);                                                  % train a linear discriminant analysis weight vector (first column is the constants)

%% test data - SVM

% train using SVM - Linear kernel
t = templateSVM('KernelFunction','linear');
Mdl = fitcecoc(FeaturesTrain,LabelTrain,'Learners',t);
isLoss = loss(Mdl,FeaturesTest,LabelTest);
CVMdl = crossval(Mdl);
genError = kfoldLoss(CVMdl);

% predict based on the SVM model
[labelSVM,scoreSVM] = predict(Mdl,FeaturesTest);
percentAccSVM_l = mean(labelSVM == LabelTest')*100;

% train using SVM - RBF kernel
t_r = templateSVM('KernelFunction','rbf');
Mdl_r = fitcecoc(FeaturesTrain,LabelTrain,'Learners',t_r);
isLoss_r = loss(Mdl_r,FeaturesTest,LabelTest);
CVMdl_r = crossval(Mdl_r);
genError_r = kfoldLoss(CVMdl_r);

% predict based on the SVM model
[labelSVM_r,scoreSVM_r] = predict(Mdl_r,FeaturesTest);
percentAccSVM_r = mean(labelSVM_r == LabelTest')*100;

% train using SVM - RBF kernel
t_q = templateSVM('KernelFunction','polynomial', 'PolynomialOrder',2);
Mdl_q = fitcecoc(FeaturesTrain,LabelTrain,'Learners',t_q);
isLoss_q = loss(Mdl_q,FeaturesTest,LabelTest);

% predict based on the SVM model
[labelSVM_q,scoreSVM_q] = predict(Mdl_q,FeaturesTest);
percentAccSVM_q = mean(labelSVM_q == LabelTest')*100;

accVec = [percentAccSVM_l, percentAccSVM_r, percentAccSVM_q]

fig = figure();
scatter(1:length(accVec),accVec, 'filled')
hold on;
x = [1:3];
y = [50,50,50]; %if three classes- 33 33 33
plot(x,y,'LineWidth',1.5)
names = {'linear';'RBF';'poly'};
set(gca,'xtick',[1:3],'xticklabel',names)
ylim([0,100])
title ('SVM prediction accuracy with different kernel types')
xlabel('kernel type')
ylabel('acurracy (%)')
legend('SVM accuracy', 'chance level')


%% Test data
% test prediction from linear classifier
test_results = (labelSVM'-LabelTest);                                         % prediction - true labels = accuracy
test_results = (sum(test_results == 0)/length(LabelTest))*100;
disp(['test accuracy - ' num2str(test_results) '%'])

save(strcat(recordingFolder,'\TestResults.mat'),'test_results');                    % save the accuracy results
save(strcat(recordingFolder,'\WeightVector.mat'),'W');                              % save the model (W)

end


