from sklearn import *
import matplotlib.pyplot as plt
from sklearn.preprocessing import *
import numpy as np
from sklearn.decomposition import PCA
from sklearn.lda import LDA

iris = datasets.load_iris()
X = iris.data
classes = iris.target
colormap = np.array(['#0b559f', '#ffff00','#000000'])
labels = iris.target_names

pca = PCA(n_components=3)
XPCA = pca.fit(X).transform(X)
plt.title('PCA reduction')
plt.xlabel('x')
plt.ylabel('y')
plt.scatter(XPCA[:,0], XPCA[:,1], c=colormap[classes]) #c is color
pop_a = mpatches.Patch(color='#0b559f', label=labels[0])
pop_b = mpatches.Patch(color='#ffff00', label=labels[1])
pop_c = mpatches.Patch(color='#000000', label=labels[2])
plt.legend(handles=[pop_a,pop_b,pop_c])
plt.show()


lda = LDA(n_components=3)
XLDA = lda.fit(X,classes).transform(X)
plt.title('LCA reduction')
plt.xlabel('x')
plt.ylabel('y')
plt.scatter(XLDA[:,0], XLDA[:,1], c=colormap[classes]) #c is color
pop_a = mpatches.Patch(color='#0b559f', label=labels[0])
pop_b = mpatches.Patch(color='#ffff00', label=labels[1])
pop_c = mpatches.Patch(color='#000000', label=labels[2])
plt.legend(handles=[pop_a,pop_b,pop_c])
plt.show()

