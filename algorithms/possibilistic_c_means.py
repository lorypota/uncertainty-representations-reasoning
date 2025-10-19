import numpy as np
from base import BaseClustering
from fuzzy_c_means import FCM

class PCM(BaseClustering):

    def __init__(self,c,m=2,distance_measure="euclidean",max_iter=100,epsilon=1e-5):
        
        self.nr_clusters = c
        self.m = m
        self.dist_measure = distance_measure
        self.max_iter = max_iter
        self.epsilon = epsilon

    def fit(self,X):
        """Assign possibilities to all data points in X for each 
        cluster K and set self.U_"""
        
        # init U and V using FCM 
        FCM_clustering = FCM(c=self.nr_clusters,m=self.m)
        U = FCM_clustering.fit_predict(X)
        V = FCM_clustering.V_

        # estimate eta_k's
        self.eta = self.get_eta(U,V,X)

        for _ in range(self.max_iter):
            # update cluster prototypes
            V = self.update_V(U,X)

            # compute new memberships
            U_new = self.update_U(V,X)

            # check convergence
            if np.max(abs(U - U_new)) < self.epsilon:
                print("converged")
                break
            else:
                U = U_new
        
        self.U_ = U_new #! was "... = U" before
        self.V_ = self.update_V(self.U_,X)
        return self

    def predict(self,X):
        """Assign fuzzy membership values from each cluster k to new data points X after being fit."""
        if not hasattr(self, "U_") or not hasattr(self, "V_"):
            raise ValueError("Model has not been fit yet. Please call 'fit' before 'predict'.")
        
        return self.update_U(self.V_,X)

    def fit_predict(self,X):
        """Fit the model to X and return the fuzzy membership values for all data points."""
        self.fit(X)
        return self.U_

    def get_eta(self,U,V,X,R=1):
    
        Um = U ** self.m
        
        D2 = self.a_norm_distances(X,V,distance_measure=self.dist_measure)

        numerator = np.sum(
            np.multiply(Um,D2),
            axis=1)

        denominator = Um.sum(axis=1)
        
        
        return R * numerator / denominator


    def a_norm_distances(self,X,V,A=None,distance_measure="euclidean"):

        if A is None:
            if distance_measure == "euclidean":
                A = np.eye(X.shape[1])
            elif distance_measure == "diagonal":
                A = np.diag(1 / np.var(X, axis=0))
            elif distance_measure == "mahalanobis":
                A = np.linalg.inv(np.cov(X, rowvar=False))
            else:
                raise ValueError(f"Unknown distance measure: '{distance_measure}'")
        
        # N.shape: (N,d) - V.shape: (c,d) -> matrix of differences (c,N,d)
        diff = X[None,:,:] - V[:,None,:] # vector diffs y_k - v_i
        # (c,N,d) @ (d,d) @ (c,N,d) -> (c,N)
        return np.einsum("cnd,df,cnf->cn",diff, A, diff)

    def update_U(self,V,X):

        D2 = self.a_norm_distances(X,V,distance_measure=self.dist_measure)
        denominator = 1 + (D2 / self.eta[:,None]) ** (1/(self.m-1))

        return 1 / denominator


    def update_V(self,U,X):

        Um = U ** self.m
        numerator = Um @ X
        denominator = Um.sum(axis=1,keepdims=True)

        return numerator / denominator



