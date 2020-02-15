import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def knn_predict(XTrain,YTrain, XPredict, K = 1):
    neigh = KNeighborsClassifier(n_neighbors=K)
    neigh.fit(XTrain, YTrain)
    Ypredict = neigh.predict(XPredict)
    return Ypredict

def knn_predict_error(XTrain,YTrain, XPredict, YTruth):
    neigh = KNeighborsClassifier(n_neighbors=1)
    neigh.fit(XTrain, YTrain)
    Ypredict = neigh.predict(XPredict)
    return accuracy_score(y_test, pred)

#XT = [[0], [1], [2], [3]]
#YT = [0, 0, 1, 1]
#XPredict = [[2.4]]
#print(knn_predict(XT,YT,XPredict))


# Test on iris data
iris = datasets.load_iris()
x=iris.data[:,:4] #all parameters
y=iris.target #class labels
testSet = [[1.4, 3.8, 2.4, 1.2]]
testSet1 = [[1.4, 3.6, 3.4, 1.2]]
testSet2 = [[4.4, 1.6, 1.7,2.1]]
print(knn_predict(x,y,testSet,10))
print(knn_predict(x,y,testSet1,10))
print(knn_predict(x,y,testSet2,10))
