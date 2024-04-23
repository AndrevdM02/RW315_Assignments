import sys
import numpy as np

class Kmeans:
    def __init__(self, n_clusters=2, max_iter=300, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X):

        if self.random_state != None:
            np.random.seed(self.random_state)

        # Generate a random startpoint, for d cluster center points.
        start_X_index = np.random.randint(low=0, high=X.shape[0], size=self.n_clusters)
        Mean = X[start_X_index,:]

        Mean[0,0] = 0
        Mean[1,0] = 5

        Mean[0,1] = 0
        Mean[1,1] = -4
        Mean = Mean.T

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
            
            # labels[dist[:,current_label]>=dist[:,1]] = 1
            # labels[dist[:,1]>dist[:,0]] = 0
                
