# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import scipy.io
import numpy as np



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    featureTest = scipy.io.loadmat('featuresTest.mat')
    featureTest =np.array(featureTest['FeaturesTest'])
    featursTrain =np.array(scipy.io.loadmat('featursTrain.mat')['FeaturesTrain'])
    LabelTest =np.array(scipy.io.loadmat('LabelTest.mat')['LabelTest'])
    LabelTrain = np.array(scipy.io.loadmat('LabelTrain.mat')['LabelTrain'])

    clf = RandomForestClassifier(max_depth=20, random_state=1, n_estimators=7000)
    clf.fit(featursTrain, LabelTrain.T)
    pred = clf.predict(featureTest)
    print(pred)
    print(clf.score(featureTest,LabelTest.T))
    print(LabelTest)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
