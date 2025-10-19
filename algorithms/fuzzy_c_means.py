import numpy as np
from base import BaseClustering

class FCM(BaseClustering):

    def __init__(self,c,m=2,distance_measure="euclidean",max_iter=100,epsilon=1e-5):
        
        self.c = c # nr of clusters
        self.m = m # fuzziness parameter
        self.distance_measure = distance_measure
        self.max_iter = max_iter
        self.epsilon = epsilon # tolerance to declare convergence

    def fit(self,X):
        """Assign fuzzy membership values to all data points in X for each 
        cluster K and set self.labels_"""
        
        N = X.shape[0]

        # init U0
        U = self.init_u0(self.c,N)

        # iterate until convergence or max_iter
        for _ in range(self.max_iter):
            # compute cluster centers
            V = self.get_cluster_centers(U,X)

            # update membership matrix
            U_new = self.update_U(X,V)

            # check convergence
            if np.max(abs(U - U_new)) < self.epsilon:
                print("converged")
                break
            else:
                U = U_new

        self.U_ = U
        self.V_ = V
        return self

    def predict(self,X):
        """Assign fuzzy membership values from each cluster k to new data points X after being fit."""
        if not hasattr(self, "U_") or not hasattr(self, "V_"):
            raise ValueError("Model has not been fit yet. Please call 'fit' before 'predict'.")
        
        return self.update_U(X,self.V_)

    def fit_predict(self,X):
        """Fit the model to X and return the fuzzy membership values for all data points."""
        self.fit(X)
        return self.U_

    def init_u0(self,c,N,rndm=True):
        """Initialize the c x N membership matrix where each column k represents a data point and
        each element U_ik is the membership degree of point k to cluster i each column k must 
        sum to 1"""
        if rndm:
            U = np.random.rand(c,N)
            return U / np.sum(U,axis=0,keepdims=True)
        else:
            raise NotImplementedError("confusion in the original paper about a fixed initialization")
        
    def get_cluster_centers(self,U,X):

        Um = U ** self.m
        numerator = Um @ X # (c x N) @ (N x d) => (c x d)
        denominator = Um.sum(axis=1, keepdims=True)
        return numerator / denominator

    def update_U(self,X,V,A=None):

        # get squared distances
        d2 = self.a_norm_distances(X=X,V=V,A=A)
        d2 = np.fmax(d2,1e-10) # avoid division by zero
        
        exponent = 1.0 / (self.m-1) # use 1.0 instead of 2.0 bc distances are squared
        
        # get distance ratios
        ratio = (d2[:,None,:] / d2[None,:,:]) ** exponent
        
        # sum and take inverse to get memberships
        return 1.0 / np.sum(ratio, axis=1)

    def a_norm_distances(self,X,V,A=None):

        if A is None:
            if self.distance_measure == "euclidean":
                A = np.eye(X.shape[1])
            elif self.distance_measure == "diagonal":
                A = np.diag(1 / np.var(X, axis=0))
            elif self.distance_measure == "mahalanobis":
                A = np.linalg.inv(np.cov(X, rowvar=False))
            else:
                raise ValueError(f"Unknown distance measure: '{self.distance_measure}'")
        
        # N.shape: (N,d) - V.shape: (c,d) -> matrix of differences (c,N,d)
        diff = X[None,:,:] - V[:,None,:] # vector diffs y_k - v_i
        # (c,N,d) @ (d,d) @ (c,N,d) -> (c,N)
        return np.einsum("cnd,df,cnf->cn",diff, A, diff)
    


