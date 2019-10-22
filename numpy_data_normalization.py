from sklearn.preprocessing import *
import numpy as np

#A
def A():     
    X = np.matrix('1 -1 2; 2 0 0; 0 1 -1')
    #print(X)
    m = X.mean()
    var = X.var()
    
    Xnormalized = scale(X) 
    #print(Xnormalized)
    #near to 0
    mNormalized = Xnormalized.mean() 
    #equal 1
    varNormalized = Xnormalized.var()

#B
def B():     
    X2 = np.matrix('1 -1 2; 2 0 0; 0 1 -1')
    print(X2)
    m2 = X2.mean(axis=1)
    print(m2)
    X2normalized = minmax_scale(X2, feature_range=(0,1))
    print(X2normalized)
    m2Normalized = X2normalized.mean(axis=1)
    print(m2Normalized)

