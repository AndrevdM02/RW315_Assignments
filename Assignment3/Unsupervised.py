import sys
import numpy as np
import matplotlib.pyplot as plt # Used for ploting graphs.
from matplotlib.patches import Ellipse

class Kmeans:
    def __init__(self, n_clusters=2, max_iter=300, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.labels = None
        self.conv = None

    def fit(self, X, means=None):

        if self.random_state != None:
            np.random.seed(self.random_state)

        # Generate a random startpoint, for d cluster center points.
        start_X_index = np.random.randint(low=0, high=X.shape[0], size=self.n_clusters)
        Mean = X[start_X_index,:]
            

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

            if np.isnan(Mean).any():
                start_X_index = np.random.randint(low=0, high=X.shape[0], size=self.n_clusters)
                Mean = X[start_X_index,:]

            # Check for convergens
            if (old_mean == Mean).all():
                self.conv = i+1
                break

        self.cluster_centers = Mean
        self.labels = labels
        return self

    def fit_means(self, X, means):

        if self.random_state != None:
            np.random.seed(self.random_state)

        # Generate a random startpoint, for d cluster center points.
        Mean = np.copy(means)
            
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
                self.conv = i+1
                break

        self.cluster_centers = Mean
        self.labels = labels
        return self
    
    def fit_1D(self, X, means):

        if self.random_state != None:
            np.random.seed(self.random_state)

        # Generate a random startpoint, for d cluster center points.
        Mean = np.copy(means)
            

        # Class labels
        labels = np.zeros(X.shape[0], dtype=int)

        # An array that will contain all of the distances to d cluster points
        dist = np.zeros([X.shape[0], self.n_clusters], dtype=float)

        # Perform max_iter of K-means
        for i in range(0, self.max_iter):
            # Calculate Distances to d cluster points
            for j in range(0, self.n_clusters):
                sum = (X[:] - Mean[j])**2
                dist[:,j] = np.sqrt(sum)
        
            # Assign a label to each observation depending to which cluster center it is the closest to
            for j in range(0,X.shape[0]):
                label = 0
                for k in range(1, self.n_clusters):
                    if dist[j,k-1]>=dist[j,k]:
                        label = k
                labels[j] = label

            old_mean = np.copy(Mean)

            # Calculate new cluster points; mean of all points beloning to the same cluster
            for j in range(0, self.n_clusters):
                Mean[j] = np.mean(X[labels==j])

            # Check for convergens
            if (old_mean == Mean).all():
                # At what iteration convergence was achieved
                self.conv = i+1
                break

        self.cluster_centers = Mean
        self.labels = labels
        return self

    def decision_boundary(self, means=None):
        Mean = np.copy(means)

        m = (Mean[1,1] - Mean[0,1]) / (Mean[1,0] - Mean[0,0])
        m = -1/m
        mid = (Mean[0]+Mean[1])/2
        c = mid[1] - (m*mid[0])

        return m, c
    
class Gaussian_Mixture:
    def __init__(self, *, n_clusters=2, max_iter=100, weights_init=None, means_init=None, cov_matrix = None, random_state=None, Lambda = 1e-6, tolerance=0.001):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.weights_init = weights_init
        self.means_init = means_init
        self.cov_matrix = cov_matrix
        self.random_state = random_state
        self.Lambda = Lambda
        self.tolerance = tolerance
        self.init_params = Kmeans
        self.covariances_ = None
        self.means_ = None
        self.gamma_var_ = None
        self.weights_ = None
        self.converged = False
        self.n_iter = None

    def eval_gauss(self, X,mu,sigma):
        val = (2*np.pi)**(sigma.shape[0]/2)
        determinant = (np.linalg.det(sigma)**0.5)
        val = val*determinant
        val = 1.0/val
        exp = np.exp(-0.5*((X-mu).T @ np.linalg.inv(sigma) @ (X-mu)))
        gauss = val*exp
        return gauss

    def gamma(self, X, pi, mu, sigma):
        gamma_var = np.zeros([X.shape[0], self.n_clusters], dtype=float)
        
        gauss = np.zeros([X.shape[0], self.n_clusters], dtype=float)
        for j in range(0,X.shape[0]):
            total = 0
            for i in range(0, self.n_clusters):
                gauss[j,i] = self.eval_gauss(X[j,:],mu[i,:],sigma[i,:,:])
                total += pi[i]*gauss[j,i]
            
            for k in range(0, self.n_clusters):
                gamma_var[j,k] = (pi[k]*gauss[j,k])/total

        return gamma_var
        
    def means(self, X, gamma_var):
        Nj = np.zeros(self.n_clusters, dtype=float)
        for i in range(0, self.n_clusters):
            Nj[i] = np.sum(gamma_var[:,i])
        
        pi = np.zeros(self.n_clusters, dtype=float)
        for i in range(0, self.n_clusters):
            pi[i] = Nj[i]/X.shape[0]

        mu = np.zeros([self.n_clusters, X.shape[1]], dtype=float)

        for i in range(0, self.n_clusters):
            for j in range(0, X.shape[1]):
                mu[i,j] = (1.0/Nj[i]) * np.sum(gamma_var[:,i]*X[:,j])

        return mu, pi
        
    def Sigma(self, X, gamma_var, mu):
        Nj = np.zeros(self.n_clusters, dtype=float)
        for i in range(0, self.n_clusters):
            Nj[i] = np.sum(gamma_var[:,i])

        sigma = np.zeros([self.n_clusters, X.shape[1], X.shape[1]],dtype=float)

        for i in range(0, self.n_clusters):
            diff = X - mu[i]

            sigma[i] = np.dot((gamma_var[:, i][:, np.newaxis] * diff).T, diff) / Nj[i]
            # Apply regularization to covariance matrix
            # sigma[i] += np.eye(X.shape[1]) * self.Lambda

        return sigma
    
    def calculate_log_likelihood(self, X, pi, mu, sigma):
        gauss = np.zeros([X.shape[0], self.n_clusters], dtype=float)
        total = 0
        for j in range(0,X.shape[0]):
            sum = 0
            for i in range(0, self.n_clusters):
                gauss[j,i] = self.eval_gauss(X[j,:],mu[i,:],sigma[i,:,:])
                sum += pi[i]*gauss[j,i]
            sum = np.log(sum)
            total += sum
        return total / X.shape[0]
    #     log_likelihood = 0
    #     for i in range(0,X.shape[0]):
    #         log_probs = np.zeros([self.n_clusters], dtype=float)
    #         for j in range(0,self.n_clusters):
    #             log_probs[j] = np.log(pi[j]) + self.eval_gauss_log(X[i], mu[j], sigma[j])
                
    #         max_log_prob = np.max(log_probs)
    #         log_likelihood += max_log_prob + np.log(np.sum(np.exp(log_probs - max_log_prob)))

    #     return log_likelihood / X.shape[0]
            
    # def eval_gauss_log(self, X, mu, sigma):
    #     exp = -0.5 * ((X-mu).T @ np.linalg.inv(sigma) @ (X-mu))
    #     log_det = -0.5 * np.log(np.linalg.det(sigma))
    #     log_const = -0.5 * X.shape[0] * np.log(2*np.pi)
    #     val = log_det + exp + log_const
    #     return val

    def fit(self, X):
        # Check if the means are given and if they are not, we use kmeans to estimate the means.
        if X.shape[1] > 1:
            kmeans = Kmeans(n_clusters=self.n_clusters, random_state=self.random_state)
            kmeans.fit(X)            

        # Starting means
        if self.means_init is None:
            mu = kmeans.cluster_centers
        else:
            mu = self.means_init

        # Starting weights
        if self.weights_init is None:
            pi = np.zeros(self.n_clusters, dtype=float)
            for i in range(0, self.n_clusters):
                sum = 0
                sum += np.sum(kmeans.labels == i)
                pi[i] = sum / X.shape[0]
        else:
            pi = self.weights_init
        
        # Starting covariance matrix
        if self.cov_matrix is None:
            sigma = np.zeros([self.n_clusters, X.shape[1], X.shape[1]],dtype=float)
            for i in range(0,self.n_clusters):
                data = X[kmeans.labels == i]
                sigma[i] = np.cov(data, rowvar=False)
            # sigma = np.zeros([self.n_clusters, X.shape[1], X.shape[1]],dtype=float)
            # for i in range(0,self.n_clusters):
            #     D = X - mu[i, :]
            #     sigma[i] = (D.T @ D) / X.shape[0]
        else:
            sigma = self.cov_matrix

        prev = float('-inf')
        prev_means = mu
        prev_sigma = sigma
        prev_pi = pi
        mu_converge = False
        sigma_converge = False
        pi_converge = False
        for i in range(0,self.max_iter):
            # E-step
            gamma_var = self.gamma(X, pi, mu, sigma)

            # M-step
            mu, pi = self.means(X, gamma_var)
            sigma = self.Sigma(X, gamma_var, mu)

            # Check for convergence using log likelihood
            ll = self.calculate_log_likelihood(X, pi, mu, sigma)
            if ll < prev:
                self.converged = True
                self.n_iter = i+1
                break

            prev = ll

            # Check for convergence in the parameters
            if not mu_converge:
                if (np.abs(mu-prev_means)<self.tolerance).all():
                    mu_converge = True
                prev_means = mu

            if not sigma_converge:
                if (np.abs(sigma-prev_sigma)<self.tolerance).all():
                    sigma_converge = True
                prev_sigma = sigma

            if not pi_converge:
                if (np.abs(pi-prev_pi)<self.tolerance).all():
                    pi_converge = True
                prev_pi = pi

            if mu_converge and sigma_converge and pi_converge:
                self.converged = True
                self.n_iter = i+1
                break

        self.means_ = mu
        self.covariances_ = sigma
        self.gamma_var_ = gamma_var
        self.weights_ = pi

    def plot_2_clusters(self, X, cov_matrix, mu, gamma_var):
        ax = plt.gca()
        ax.set_xlim([-6, 6])
        ax.set_ylim([-6, 6])
        ax.patch.set_facecolor('white')
        # Set the color of the borders
        ax.spines['top'].set_color('black')
        ax.spines['bottom'].set_color('black')
        ax.spines['left'].set_color('black')
        ax.spines['right'].set_color('black')

        # Set the width of the borders
        ax.spines['top'].set_linewidth(1)
        ax.spines['bottom'].set_linewidth(1)
        ax.spines['left'].set_linewidth(1)
        ax.spines['right'].set_linewidth(1)
        scale_factor = 1.95

        colors = np.array([[0,1,0], [1,0,0]])

        for i in range(mu.shape[0]):
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix[i])

            order = eigenvalues.argsort()[::-1]

            eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:,order]

            width, height = 2*np.sqrt(eigenvalues)*scale_factor

            angle = np.degrees(np.arctan2(*eigenvectors[:,0][::-1]))

            ellipse = Ellipse((mu[i,0], mu[i,1]), width=width, height=height, angle=angle, edgecolor=colors[i,:],facecolor='white',fill=True,linewidth=3,zorder=1)
            ax.add_artist(ellipse)

        for i in range(X.shape[0]):
            col = [[gamma_var[i,1],gamma_var[i,0],0]]
            ax.scatter(X[i,0],X[i,1],c=col,zorder=3,alpha=0.2)

        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        plt.show()

    def estimate_gamma(p,u,s,x,c):
        import math
        from math import exp
        from math import sqrt
        from math import pi
        
        gamma = []
        for i in range(0, len(x)):
            total = 0
            for j in range(0, len(u)):
                const = sqrt(pi*2*s[j]**2)
                const = 1/const
                expon = -0.5*((x[i]-u[j])**2)/s[j]**2
                expon = exp(expon)
                val = const*expon*p[j]
                total += val
                
            const = sqrt(pi*2*s[c]**2)
            const = 1/const
            expon = -0.5*((x[i]-u[c])**2)/s[c]**2
            expon = exp(expon)
            val = const*expon*p[c]
            val = val/total
            gamma.append(val)
            # gamma.append((round(val,4)))
        return gamma
