#A
from sklearn import *
import matplotlib.pyplot as plt
import numpy as np

#B
def B():    
    #1
    iris = datasets.load_iris()
    
    #2 
    print(iris.data)
    print(iris.feature_names)
    target = iris.target
    target_names = iris.target_names
    
    #3
    tn = target_names[target]
    
    
    #4 
    dataMean = iris.data.mean(0)
    dataStd = iris.data.std(0)
    dataMax = iris.data.max(0)
    dataMin = iris.data.min(0)
    
    #5
    nbrClasses = iris.target_names.size #nbr de classes 
    dataCount = iris.data.shape[0] #nbr de data
    varblesCount = iris.data.shape[1] #nbr de variables

#D
#part I
x,y = make_blobs(n_samples  = 1000,n_features = 2, centers = 4)
plt.title('For matrix x')
plt.axis([-15, 15, -15, 15])
plt.xlabel('x')
plt.ylabel('y')
plt.scatter(x[:,0], x[:,1], c=y) #c is color
plt.show()

#part II
x1,y1 = make_blobs(n_samples  = 100,n_features = 2, centers = 2)
x2,y2 = make_blobs(n_samples  = 500,n_features = 2, centers = 3)

plt.figure(figsize=(8, 8))
plt.title('For matrix x1')
plt.axis([-15, 15, -15, 15])
plt.xlabel('x')
plt.ylabel('y')
plt.scatter(x1[:,0], x1[:,1], c=y1) #c is color
plt.show()

plt.figure(figsize=(8, 8))
plt.title('For matrix x2')
plt.axis([-15, 15, -15, 15])
plt.xlabel('x')
plt.ylabel('y')
plt.scatter(x2[:,0], x2[:,1], c=y2) #c is color
plt.show()

#merging both
x3 = np.vstack((x1,x2))
y3 = np.hstack((y1,y2))
plt.figure(figsize=(8, 8))
plt.title('For matrix x3')
plt.axis([-15, 15, -15, 15])
plt.xlabel('x')
plt.ylabel('y')
plt.scatter(x3[:,0], x3[:,1], c=y3) #c is color
plt.show()

