import sys
import numpy as np

class NBClassifier:
    def __init__(self, priors = False):
        self.priors = priors
        self.class_prior_ = None
        self.class_count_ = None
        self.classes_ = None
        self.means_ = None
        self.var_ = None

    
    def fit(self, X, y):
        classes = np.unique(y)
        self.classes_ = classes
        num_classes = len(classes)
        N = y.shape[0]
        start = np.unique(y)[0]
        # Number of Observations for each class N_j
        Nj = np.zeros(num_classes)
        for a in range(0, num_classes):
            for i in range(0, N):
                if y[i] == a+start:
                    Nj[a] += 1
        self.class_count_ = Nj

        # Estimating the class probability P(C_j) = N_j/N
        Class_j = np.zeros(num_classes)
        for a in range(0, num_classes):
            Class_j[a] = Nj[a]/N
        self.class_prior_ = Class_j

        # Estimating the means
        n_dim = X.shape[1]
        mu = np.zeros((num_classes, n_dim))
        for a in range(0, num_classes):
            for i in range(0, N):
                if y[i] == a+start:
                    for j in range(0, n_dim):
                        mu[a,j] += X[i, j]

            for j in range(0, n_dim):
                mu[a,j] = mu[a,j]/Nj[a]
        self.means_ = mu

        # Estimating the variance
        sigma = np.zeros((num_classes, n_dim))
        for a in range(0, num_classes):
            for i in range(0, N):
                if y[i] == a+start:
                    for j in range(0, n_dim):
                        sigma[a,j] += (X[i,j] - mu[a,j])**2

            for j in range(0, n_dim):
                sigma[a,j] = sigma[a,j]/Nj[a]
        self.var_ = sigma

    def gauss(self, X, mu, sigma):
        y = 1/(np.sqrt(2*np.pi*sigma))*np.exp(-1*(X-mu)**2/(2*sigma))
        return y


    def proba(self, X):
        n_dim = X.shape[1]
        prob = np.ones((X.shape[0], len(self.classes_)))
        for i in range(0, X.shape[0]):
            for j in range(0, len(self.classes_)):    
                for a in range(0, n_dim):
                    prob[i, j] *= self.gauss(X[i, a], self.means_[j, a], self.var_[j, a])
                prob[i, j] *= self.class_prior_[j]

        class_prob = np.ones((X.shape[0], len(self.classes_)))
        for i in range(0, X.shape[0]):
            poster = 0
            for j in range(0, len(self.classes_)):
                poster += prob[i, j]
            
            for j in range(0, len(self.classes_)):
                class_prob[i, j] = prob[i,j]/poster
        
        return class_prob

    def log_gauss(self, X, mu, sigma):
        y = (-np.log(np.sqrt(2*np.pi)) - np.log(np.sqrt(sigma)) - (((X - mu)**2)/(sigma*2)))
        return y

    def log_proba(self, X):
        n_dim = X.shape[1]
        log_prob = np.zeros((X.shape[0], len(self.classes_)))
        for i in range(0, X.shape[0]):
            for j in range(0, len(self.classes_)):    
                for a in range(0, n_dim):
                    log_prob[i, j] += self.log_gauss(X[i, a], self.means_[j, a], self.var_[j, a])
                log_prob[i,j] += np.log(self.class_prior_[j])

        log_class_prob = np.zeros((X.shape[0], len(self.classes_)))
        for i in range(0, X.shape[0]):
            log_poster = 0
            for j in range(0, len(self.classes_)):
                log_poster += np.exp(log_prob[i, j])
            
            for j in range(0, len(self.classes_)):
                log_class_prob[i, j] = log_prob[i,j] - np.log(log_poster)

        # log_class_prob = self.proba(X)
        # log_class_prob = np.log(log_class_prob)
        
        return log_class_prob
    
    
    def predict(self, X):
        y = self.proba(X)
        cls = self.classes_[np.argmax(y, axis=1)]
        return cls


class LRClassifier:
    def __init__(self, lamda_value = 1.0, Bias = True, tol = 1e-5, max_iter = 100):
        self.classes_ = None
        self.lamda_value = lamda_value
        self.Bias = Bias
        self.weights = None
        self.tol = tol
        self.max_iter = max_iter

    def add_bias(self, X):
        if self.Bias:
            return np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        else:
            return X

    def sigmoid(self, X):
        z = (1/(1+np.exp(-X)))
        return z

    def log_likelihood(self, X, w, y):
        z = X.dot(w)
        z = self.sigmoid(z)
        log_likelihood = np.sum(y * np.log(z))
        return log_likelihood
    
    def log_prior(self, w):
        if self.lamda_value > 0:
            # Gaussian prior with zero mean and covariance matrix lambda*I
            prior = -0.5 * np.dot(w, np.dot(np.eye(len(w)) / self.lamda_value, w))
            return prior
        else:
            return 0
        
    def log_posterior(self, X, w, y):
        ll = self.log_likelihood(X, w, y)
        lp = self.log_prior(w)
        log_post = ll + lp
        return log_post 
    
    def negative_log_likelihood(self, X, w, y):
        z = X.dot(w)
        z = self.sigmoid(z)
        l = -np.sum((y * np.log(z)) + (1 - y) * np.log(1 - z)) + w.T.dot(w)/(2*self.lamda_value)
        return l

    def gradient(self, X, w, y):
        z = X.dot(w)
        new_y = self.sigmoid(z)
        error = y - new_y
        grad = X.T.dot(error)
        if self.lamda_value > 0:
            grad -= w / self.lamda_value
        return -grad
    
    def hessian(self, X, w):
        z = X.dot(w)
        new_y = self.sigmoid(z)
        # Calculating the sigmoids differential
        S = np.diag(new_y * (1 - new_y))
        # Calculating the Hessian
        H = X.T.dot(S).dot(X)
        if self.lamda_value > 0:
            H += np.eye(len(w)) / self.lamda_value
        return H

    def fit(self, X, y):
        classes = np.unique(y)
        self.classes_ = classes
        X = self.add_bias(X)
        num_dim = X.shape[1]
        self.weights = np.zeros(num_dim)
    
        for i in range(0, self.max_iter):
            gradient = self.gradient(X, self.weights, y)
            hessian = self.hessian(X, self.weights)

            #Solves the equation for the hessian inverse * gradient vector
            update = np.linalg.solve(hessian, gradient)

            #Creates the new coef by adding the current one with the update
            self.weights -= update

            if np.linalg.norm(update) < self.tol:
                break
    
    def predict_proba(self, X):
        X = self.add_bias(X)
        z = X.dot(self.weights)
        y = self.sigmoid(z)
        return y
    
    def predict(self, X):
        prob = self.predict_proba(X)
        cls = (prob >= 0.5).astype(int)
        return cls