import numpy as np

class LassoRegression:

    '''
    This implementation defines a LassoRegression class with fit and predict methods.
    The fit method takes in the training data X and y, where X is an ndarray of shape (N, M) representing a dataset consisting of N examples, each of dimension M, and y is an ndarray of shape (N,) representing the target values.
    If fit_intercept is True, the method adds a column of ones to X to account for the intercept term. The method then iteratively updates the regression coefficients self.beta using the coordinate descent algorithm, where each iteration updates the coefficient for one feature while holding the others fixed.
    The soft_threshold method is a helper function that implements the soft thresholding operation, which is used to shrink the regression coefficients towards zero.
    The predict method takes in a new dataset X and returns the predicted target values based on the learned regression coefficients self.beta. If fit_intercept is True, the method again adds a column of ones to X to account for the intercept term.
    '''
    def __init__(self, alpha=1.0, fit_intercept=True, max_iter=1000, tol=1e-4):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.beta = None

    def soft_threshold(self, x, gamma):
        if x > 0 and gamma < abs(x):
            return x - gamma
        elif x < 0 and gamma < abs(x):
            return x + gamma
        else:
            return 0

    def fit(self, X, y):
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]

        n_features = X.shape[1]
        self.beta = np.zeros(n_features)
        for _ in range(self.max_iter):
            beta_old = np.copy(self.beta)
            for j in range(n_features):
                X_j = X[:, j]
                y_pred = X @ self.beta - X_j * self.beta[j]
                rho = X_j.T @ (y - y_pred)
                self.beta[j] = self.soft_threshold(rho, self.alpha)

            if np.linalg.norm(self.beta - beta_old) < self.tol:
                break

    def predict(self, X):
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]
        return X @ self.beta
