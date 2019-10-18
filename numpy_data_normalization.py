from sklearn.preprocessing import *
import numpy as np


X = np.matrix('1 -1 2; 2 0 0; 0 1 -1')
print(X)
m = X.mean()
var = X.var()

Xnormalized = scale(X) 
print(Xnormalized)
#near to 0
mNormalized = Xnormalized.mean() 
#equal 1
varNormalized = Xnormalized.var()

