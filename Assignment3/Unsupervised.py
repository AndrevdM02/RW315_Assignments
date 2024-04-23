import sys
import numpy as np

class Kmeans:
    def __init__(self, n_clusters=2, max_iter=300, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.cluster_means = None
        self.labels = None

    def fit(self, X, means=None):

        if self.random_state != None:
            np.random.seed(self.random_state)

        # Generate a random startpoint, for d cluster center points.
        if (means == None).all:
            start_X_index = np.random.randint(low=0, high=X.shape[0], size=self.n_clusters)
            Mean = X[start_X_index,:]
        else:
            Mean = means
            

        # Class labels
        labels = np.zeros(X.shape[0], dtype=int)

        # An array that will contain all of the distances to d cluster points
        dist = np.zeros([X.shape[0], self.n_clusters], dtype=float)

        # Perform max_iter of K-means
        for i in range(0, self.max_iter):
            # Calculate Distances to d cluster points
            for j in range(0, self.n_clusters):
                sum = (X[:,0] - Mean[j,0])**2
                for k in range(1, X.shape[1]):
                    sum += (X[:,k] - Mean[j,k])**2
                dist[:,j] = np.sqrt(sum)
        
            # Assign a label to each observation depending to which cluster center it is the closest to
            # TODO
            for j in range(0,X.shape[0]):
                label = 0
                for k in range(1, self.n_clusters):
                    if dist[j,k-1]>=dist[j,k]:
                        label = k
                labels[j] = label

            old_mean = np.copy(Mean)

            # Calculate new cluster points; mean of all points beloning to the same cluster
            for j in range(0, self.n_clusters):
                for k in range(0, X.shape[1]):
                    Mean[j,k] = np.mean(X[labels==j, k])

            # Check for convergens
            if (old_mean == Mean).all():
                break

        self.cluster_centers = Mean
        self.labels = labels
        return self
