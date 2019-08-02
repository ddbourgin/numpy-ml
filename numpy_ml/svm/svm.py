import numpy as np
import random


class SVM(object):
    def __init__(self, C=1.0, kernel='linear', tol=0.0001, max_iter=200):
        """
        Implementation of SVM Computing Based on SMO Algorithms.
        kernels include : linear, quadratic, gaussian

        Reference
        ----------------
        <<Machine Learning in Action>>
        Platt J C . Fast Training of Support Vector Machines Using Sequential 
        Minimal Optimization[M]// Advances in kernel methods. MIT Press, 1999.

        Parameters
        ----------------
        C : float in [0, 1] (default: 1.0)
            penalty parameter C of error term.
        kernel: string in ['linear', 'rbf'] (default: "linear")
            specifies the kernel type to be used in the algorithms.
        tol : float (default: 0.0001)
            tolerance for stopping criterion.
        max_iter : int (default: 200)
            hard limit on iterations within solver.
        """
        self.kernels = {
            'linear': self.__kernel_linear,
            "rbf": self.__kernel_rbf
        }
        err_msg = "The kernel must in ['linear', 'rbf'], but got {0}.".format(kernel)
        assert kernel in self.kernels, err_msg
        self.kernel = kernel
        self.C = C
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, X, y):
        """
        Fit the parameters of the SVM on some training data.

        Parameters
        ----------
        X : training data
        y : training label

        Returns
        -------
        support_vectors : .
        """
        # initial
        n = X.shape[0]
        alpha = np.zeros((n))
        kernel_func = self.kernels[self.kernel]
        iters = 0
        while True:
            iters += 1
            alpha_prev = np.copy(alpha)
            for j in range(n):
                i = self.__get_rand_int(0, n - 1, j)  # get random int i~=j
                X_i, X_j, y_i, y_j = X[i, :], X[j, :], y[i], y[j]
                k_ij = kernel_func(X_i, X_i) + kernel_func(X_j, X_j) - 2 * kernel_func(X_i, X_j)
                if k_ij == 0:
                    continue
                alpha_prime_j, alpha_prime_i = alpha[j], alpha[i]
                L, H = self.__calc_L_H(self.C, alpha_prime_j, alpha_prime_i, y_j, y_i)
                # calculate model parameters
                self.w = self.__calc_w(alpha, y, X)
                self.b = self.__calc_b(X, y, self.w)
                # calculate E_i, E_j
                E_i = self.__calc_loss(X_i, y_i, self.w, self.b)
                E_j = self.__calc_loss(X_j, y_j, self.w, self.b)
                # set new alpha values
                alpha[j] = alpha_prime_j + float(y_j * (E_i - E_j)) / k_ij
                alpha[j] = max(alpha[j], L)
                alpha[j] = min(alpha[j], H)
                alpha[i] = alpha_prime_i + y_i * y_j * (alpha_prime_j - alpha[j])
            if np.linalg.norm(alpha - alpha_prev) < self.tol:  # check convergence
                break
            if iters >= self.max_iter:
                # print("The number of iterations exceeds the maximum number of iterations set.")
                return
        # calculate final model parameters
        self.b = self.__calc_b(X, y, self.w)
        if self.kernel == 'linear':
            self.w = self.__calc_w(alpha, y, X)
        # calculate support vectors
        alpha_idx = np.where(alpha > 0)[0]
        support_vectors = X[alpha_idx, :]
        return support_vectors

    def predict(self, X):
        pred = self.__pred(X, self.w, self.b)
        return np.where(pred == -1, 0, pred)

    def __calc_b(self, X, y, w):
        # calculate model parameter b
        b_tmp = y - np.dot(w.T, X.T)
        return np.mean(b_tmp)

    def __calc_w(self, alpha, y, X):
        # calculate model parameter w
        return np.dot(X.T, np.multiply(alpha, y))

    def __pred(self, X, w, b):
        return np.sign(np.dot(w.T, X.T) + b).astype(int)

    def __calc_loss(self, x_k, y_k, w, b):
        # calculate predict error
        return self.__pred(x_k, w, b) - y_k

    def __calc_L_H(self, C, alpha_prime_j, alpha_prime_i, y_j, y_i):
        if (y_i != y_j):
            return (max(0, alpha_prime_j - alpha_prime_i),
                    min(C, C - alpha_prime_i + alpha_prime_j))
        else:
            return (max(0, alpha_prime_i + alpha_prime_j - C),
                    min(C, alpha_prime_i + alpha_prime_j))

    def __get_rand_int(self, a, b, z):
        # get random int
        i = z
        cnt = 0
        while i == z and cnt < 1000:
            i = random.randint(a, b)
            cnt = cnt + 1
        return i

    # define some kernel func
    def __kernel_linear(self, x1, x2):
        return np.dot(x1, x2.T)

    def __kernel_rbf(self, x, y, sigma=1):
        if np.ndim(x) == 1 and np.ndim(y) == 1:
            res = np.exp(-(np.linalg.norm(x - y, 2)) ** 2 / (2 * sigma ** 2))
        elif (np.ndim(x) > 1 and np.ndim(y) == 1) or (np.ndim(x) == 1 and np.ndim(y) > 1):
            res = np.exp(-(np.linalg.norm(x - y, 2, axis=1) ** 2) / (2 * sigma ** 2))
        elif np.ndim(x) > 1 and np.ndim(y) > 1:
            res = np.exp(-(np.linalg.norm(x[:, np.newaxis] - y[np.newaxis, :], 2, axis=2) ** 2) / (2 * sigma ** 2))
        else:
            print("Error")
        return res
