from abc import ABC, abstractmethod

class BaseClustering(ABC):
    """Abstract base class for clustering algorithms"""

    @abstractmethod 
    def fit(self,X):
        """Assign clusters to all data points in X and set self.labels_"""
        pass

    def predict(self,X):
        """Optional: Assign new data points X to clusters after being fit. only override 
        if clustering algorithm supports predicting new points after fitting. """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support predict()."
        )

    def fit_predict(self,X):
        """Fit the model to X and return the cluster labels."""
        self.fit(X)
        return self.labels_