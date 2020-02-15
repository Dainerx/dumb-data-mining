import numpy as np
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def naivebayes(XTrain,YTrain,XPredict):
    model = GaussianNB()
    model.fit(XTrain, YTrain)
    return model.predict(XPredict)

def svm(XTrain,YTrain,XPredict):
    model = SVC()
    model.fit(XTrain, YTrain)
    return model.predict(XPredict)

def decisiontree(XTrain,YTrain,XPredict):
    model = DecisionTreeClassifier()
    model.fit(XTrain, YTrain)
    return model.predict(XPredict)


# split data into training and test 
# 70% training and 30% test.
iris = datasets.load_iris()
X=iris.data[:,:4] #all parameters
y = iris.target
XTrain, XTest, YTrain, YTest = train_test_split(X, y, 
test_size=0.3)

#bayes error
print(mean_squared_error(naivebayes(XTrain, YTrain,XTest),YTest))
#decision tree error
print(mean_squared_error(decisiontree(XTrain, YTrain,XTest),YTest))
#svm error
print(mean_squared_error(svm(XTrain, YTrain,XTest),YTest))

# further analysis:
print(metrics.classification_report(YTest, YTrain))
print(metrics.confusion_matrix(YTest, YTrain))