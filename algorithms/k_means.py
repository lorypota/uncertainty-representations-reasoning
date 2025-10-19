##################################################################################
# K-Means Clustering Algorithm:                                                  #
# measure similarity between points using ordinary straight-line distance. Here, #
# k stands for the number of centroids it creates and every point is assigned to #
# the nearest centroid.                                                          #
##################################################################################

from base import BaseClustering
import numpy as np

class KMeans(BaseClustering):

    def __init__(self,k,max_iter=250,tol=1e-8,n_init=10):
        
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init

    def fit(self,X):
        """Assign clusters to all data points in X and set self.labels_"""
        best_inertia = np.inf
        best_labels = None
        best_centroids = None

        # repeat the full algorithm n_init times to reduce risk of bad initialization
        for _ in range(self.n_init):
            # init centroids
            C = self.init_centroids(X)

            for _ in range(self.max_iter):
                # compute current distances
                distances = np.linalg.norm(X[:,None]-C,axis=2)

                # assign each point to nearest cluster & compute total dist
                labels = np.argmin(distances,axis=1)
                # inertia: measure how tight the distances are
                inertia = np.sum(distances[np.arange(len(X)), labels] ** 2)

                old_C = np.copy(C)

                # update centroids
                C = np.array([
                    np.mean(X[labels==i],axis=0)if len(X[labels==i])>0 else C[i]
                    for i in range(len(C))
                ])

                # break if max shift of a centroid is less than the tolerance
                shift = np.max(np.linalg.norm(C - old_C, axis=1))
                if shift < self.tol:
                    break

            # update if inertia is the smallest distance so far.
            if inertia < best_inertia:
                best_inertia = inertia
                best_labels = labels
                best_centroids = C

        self.labels_ = best_labels
        self.centroids_ = best_centroids
        self.inertia_ = best_inertia
        return self
    
    def predict(self,X):
        """Assign new data points X to the self.k clusters after being fit."""
        if not hasattr(self, "labels_") or not hasattr(self, "centroids_"):
            raise ValueError("Model has not been fit yet. Please call 'fit' before 'predict'.")
        
        distances = np.linalg.norm(X[:,None]-self.centroids_,axis=2)
        return np.argmin(distances,axis=1)

    def fit_predict(self,X):
        """Fit the model to X and return the cluster labels for all data points."""
        self.fit(X)
        return self.labels_
    
    def init_centroids(self, X: np.ndarray) -> np.ndarray:
        """Randomly select k points which will represent the k centroids."""
        indices = np.random.choice(X.shape[0],size=self.k,replace=False)
        return X[indices]
