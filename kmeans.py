import numpy as np
from sklearn import datasets
from sklearn.metrics import pairwise_distances_argmin

def kmeans_implementation(X, nclusters):
    rand = np.random.RandomState()
    i = rand.permutation(X.shape[0])[:nclusters]
    centers = X[i]
    while True:
        labels = pairwise_distances_argmin(X, centers)
        # trouve des nouveaux centres
        new_centers = np.array([X[labels == i].mean(0)
                                for i in range(nclusters)])
        # Converged?
        if np.all(centers == new_centers):
            break
        centers = new_centers
    return centers, labels

def kmeans_implementation_range_clusters(X,nclusters_range):
    best_centers = []
    best_labels = []
    silhouette_best_score = -2
    for i in range(nclusters_range):
        centers, labels = kmeans_implementation(X,nclusters_range[i])
        if (silhouette_score(X, centers) > silhouette_best_score):
            best_centers = centers 
            best_labels  = labels
    return best_centers,best_labels

iris = datasets.load_iris()
x=iris.data[:,:4] #all parameters
centers, labels = kmeans_implementation(x, 4)

print(centers)



