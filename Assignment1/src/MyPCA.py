import numpy as np
import sys

class MyPCA:
    def __init__(self, n_components=None, whitening=False, cov_matrix = False):
        self.n_components = n_components
        self.whitening = whitening
        self.data = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.mean = None
        self.Q = None
        self.N = None
        self.u = None
        self.s = None
        self.vh = None
        self.Lam = None
        self.cov_matrix = cov_matrix
        self.xbar = None
        self.Ub = None

    def fit_eigen(self, X, y=None, dim=False):
        if self.whitening is False and self.cov_matrix is False:

            d, self.N = X.shape

            self.mean = np.mean(X,axis=1)[:,np.newaxis]
            centred_data = X - self.mean

            cov_matrix = (centred_data.dot(centred_data.T))/self.N

            eigen_values, eigen_vectors = np.linalg.eigh(cov_matrix)
            
            idx = np.argsort(eigen_values)[::-1]
            eigen_values = eigen_values[idx]
            eigen_vectors = eigen_vectors[:,idx]

            q = eigen_values[:self.n_components]
            self.explained_variance_ = eigen_values
            self.explained_variance_ratio_ = q/sum(eigen_values)

            sgn = np.sign(np.mean(eigen_vectors,axis=0)[:,np.newaxis])
            for a in range(0, self.n_components):
                eigen_vectors[:,a] = sgn[a]*eigen_vectors[:,a]

            self.Q = eigen_vectors[:,:self.n_components]

            return self

        elif self.cov_matrix is True:
            d, self.N = X.shape
            eigen_values, eigen_vectors = np.linalg.eigh(X)
            
            idx = np.argsort(eigen_values)[::-1]
            eigen_values = eigen_values[idx]
            eigen_vectors = eigen_vectors[:,idx]

            q = eigen_values[:self.n_components]
            self.explained_variance_ = eigen_values
            self.explained_variance_ratio_ = q/sum(eigen_values)

            sgn = np.sign(np.mean(eigen_vectors,axis=0)[:,np.newaxis])
            for a in range(0, self.n_components):
                eigen_vectors[:,a] = sgn[a]*eigen_vectors[:,a]

            self.Q = eigen_vectors[:,:self.n_components]

            return self

        else:
            print("False")

    def transform_eigen(self, X, y=None, dim=False):
            centred_data = X - self.mean
            y = centred_data.T.dot(self.Q)
            # y = centred_data.T.dot(self.Q)

            return y
        

    def inverse_transform(self, X, y=None):
        if self.whitening is False:
            y = self.mean.T + (X.dot(self.Q.T))
            return y.T
        else:
            y = self.Q.dot(np.linalg.inv(self.Lam))
            y = self.mean + (y.dot(X))
            return y

    def fit_svd(self, X, y=None):
        if self.whitening is False and self.cov_matrix is False:
            d, N = X.shape
            self.mean = np.mean(X, axis=1)[:,np.newaxis]
            centred_data = X - self.mean

            
            self.u, self.s, self.vh = np.linalg.svd(centred_data, full_matrices=False)

            sgn = np.sign(np.mean(self.u,axis=0)[:,np.newaxis])
            for a in range(0, self.n_components):
                self.u[:,a] = sgn[a]*self.u[:, a]
                self.vh[a,:] = sgn[a]*self.vh[a,:]

            self.explained_variance_ = (self.s**2) / (N)
            self.explained_variance_ratio_ = self.explained_variance_/sum(self.explained_variance_)

            self.Q = self.u[:, :self.n_components]

            


            return self
        
        elif self.cov_matrix is True:
            d, N = X.shape
            self.u, self.s, self.vh = np.linalg.svd(X, full_matrices=False)

            sgn = np.sign(np.mean(self.u,axis=0)[:,np.newaxis])
            for a in range(0, self.n_components):
                self.u[:,a] = sgn[a]*self.u[:, a]
                self.vh[a,:] = sgn[a]*self.vh[a,:]

            self.explained_variance_ = (self.s**2) / (N)
            self.explained_variance_ratio_ = self.explained_variance_/sum(self.explained_variance_)

            self.Q = self.u[:, :self.n_components]

            return self

        else:
            d, N = X.shape
            self.mean = np.mean(X, axis=1)[:,np.newaxis]
            centred_data = X - self.mean

            self.u, self.s, self.vh = np.linalg.svd(centred_data, full_matrices=False)

            sgn = np.sign(np.mean(self.u,axis=0)[:,np.newaxis])
            for a in range(0, self.n_components):
                self.u[:,a] = sgn[a]*self.u[:, a]
                self.vh[a,:] = sgn[a]*self.vh[a,:]

            self.explained_variance_ = (self.s**2) / (N)
            self.explained_variance_ratio_ = self.explained_variance_/sum(self.explained_variance_)

            self.Q = self.u[:, :self.n_components]

            self.Lam = np.linalg.inv(np.diag(self.explained_variance_))
            self.Lam = np.sqrt(self.Lam)
            self.Lam = self.Lam[:self.n_components,:self.n_components]


            return self

    def transform_svd(self, X):
        if self.whitening is False and self.cov_matrix is False:
            centerd_data = X - self.mean
            y = self.Q.T.dot(centerd_data)
            y = y.T
            return y
        
        elif self.cov_matrix is True:
            
            return None
        else:
            centerd_data = X - self.mean
            Qv = self.Lam.T.dot(self.Q.T)
            y = Qv.dot(centerd_data)
            y = y.T
            return y
    
    def fit_lda(self, X, y=None):
        d, self.N = X.shape

        self.mean = np.mean(X,axis=1)[:,np.newaxis]
        centred_data = X - self.mean

        classes = np.unique(y)
        num_classes = len(classes)

        Sb = np.zeros((d,d))
        Sw = np.zeros((d,d))

        # Total scatter
        St = np.cov(centred_data)

        for a in classes:
            # Data points in class a
            X_a = X[:,y == a]

            # Class mean
            class_mean = np.mean(X_a, axis=1)[:,np.newaxis]

            # Between-class scatter
            Sb += X_a.shape[1] * (class_mean - self.mean) @ (class_mean - self.mean).T
            # Sb += (X_a.shape[1]/self.N * (class_mean - self.mean) @ (class_mean - self.mean).T)

            # Within-class scatter
            # Sw += ((X_a - class_mean) @ (X_a - class_mean).T) / len(X_a)
            Sw += (X_a.shape[1]/self.N) * ((X_a - class_mean) @ (X_a - class_mean).T) /  X_a.shape[1]

        # print(Sw)
        # print(Sb)

        lda2 = MyPCA(n_components=np.linalg.matrix_rank(Sw), cov_matrix=True)
        lda2 = lda2.fit_eigen(Sw)
        lam_inv = np.diag(1.0/np.sqrt(lda2.explained_variance_))
        self.Lam = lam_inv

        whiten = lda2.Q @ lam_inv
        whiten_b = whiten.T @ Sb @ whiten

        lda = MyPCA(n_components=np.linalg.matrix_rank(whiten_b), cov_matrix=True)
        lda = lda.fit_eigen(whiten_b)

        self.Ub = lda.Q
        self.Q = lda2.Q
        

        return self
        
    def transform_lda(self, X, y=None):
        centred_data = X - self.mean
        y = self.Ub.T @ self.Lam @ self.Q.T @ centred_data
        return y


