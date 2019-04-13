import numpy as np
from numpy.testing import assert_allclose


class GMM(object):
    def __init__(self, X, C=3):
        self.X = X
        self.C = C  # number of clusters
        self.N = X.shape[0]  # number of objects
        self.d = X.shape[1]  # dimension of each object

    def _initialize_params(self):
        C = self.C
        d = self.d
        rr = np.random.rand(C)

        # randomly initialize the starting GMM parameters
        self.pi = rr / rr.sum()  # cluster priors
        self.Q = np.zeros((self.N, C))  # variational distribution q(T)
        self.mu = np.random.uniform(-5, 10, C * d).reshape(C, d)  # cluster means
        self.sigma = np.array([np.identity(d) for _ in range(C)])  # cluster covariances

        self.best_pi = None
        self.best_mu = None
        self.best_sigma = None
        self.best_elbo = -np.inf

    def likelihood_lower_bound(self):
        N = self.N
        C = self.C

        eps = np.finfo(float).eps
        expec1, expec2 = 0.0, 0.0
        for i in range(N):
            x_i = self.X[i]

            for c in range(C):
                pi_k = self.pi[c]
                z_nk = self.Q[i, c]
                mu_k = self.mu[c, :]
                sigma_k = self.sigma[c, :, :]

                log_pi_k = np.log(pi_k)
                log_p_x_i = log_gaussian_pdf(x_i, mu_k, sigma_k)
                prob = z_nk * (log_p_x_i + log_pi_k)

                expec1 += prob
                expec2 += z_nk * np.log(z_nk + eps)

        loss = expec1 - expec2
        return loss

    def fit(self, max_iter=75, tol=1e-3, verbose=False):
        self._initialize_params()
        prev_vlb = -np.inf

        for _iter in range(max_iter):
            try:
                self._E_step()
                self._M_step()
                vlb = self.likelihood_lower_bound()

                if verbose:
                    print("{}. Lower bound: {}".format(_iter + 1, vlb))

                if np.isnan(vlb) or np.abs((vlb - prev_vlb) / prev_vlb) <= tol:
                    break

                prev_vlb = vlb

                # retain best parameters across fits
                if vlb > self.best_elbo:
                    self.best_elbo = vlb
                    self.best_mu = self.mu
                    self.best_pi = self.pi
                    self.best_sigma = self.sigma

            except np.linalg.LinAlgError:
                print("Singular matrix: components collapsed")
                return -1
        return 0

    def _E_step(self):
        for i in range(self.N):
            x_i = self.X[i, :]

            denom_vals = []
            for c in range(self.C):
                pi_c = self.pi[c]
                mu_c = self.mu[c, :]
                sigma_c = self.sigma[c, :, :]

                log_pi_c = np.log(pi_c)
                log_p_x_i = log_gaussian_pdf(x_i, mu_c, sigma_c)

                # log N(X_i | mu_c, Sigma_c) + log pi_c
                denom_vals.append(log_p_x_i + log_pi_c)

            # log \sum_c exp{ log N(X_i | mu_c, Sigma_c) + log pi_c } ]
            log_denom = logsumexp(denom_vals)
            q_i = np.exp([num - log_denom for num in denom_vals])
            assert_allclose(np.sum(q_i), 1, err_msg="{}".format(np.sum(q_i)))

            self.Q[i, :] = q_i

    def _M_step(self):
        C, N, X = self.C, self.N, self.X
        denoms = np.sum(self.Q, axis=0)

        # update cluster priors
        self.pi = denoms / N

        # update cluster means
        nums_mu = [np.dot(self.Q[:, c], X) for c in range(C)]
        for ix, (num, den) in enumerate(zip(nums_mu, denoms)):
            self.mu[ix, :] = num / den

        # update cluster covariances
        for c in range(C):
            mu_c = self.mu[c, :]
            n_c = denoms[c]

            outer = np.zeros((2, 2))
            for i in range(N):
                wic = self.Q[i, c]
                xi = self.X[i, :]
                outer += wic * np.outer(xi - mu_c, xi - mu_c)

            outer /= n_c
            self.sigma[c, :, :] = outer

        assert_allclose(np.sum(self.pi), 1, err_msg="{}".format(np.sum(self.pi)))


#######################################################################
#                                Utils                                #
#######################################################################


def log_gaussian_pdf(x_i, mu, sigma):
    """
    Compute log N(x_i | mu, sigma)
    """
    n = len(mu)
    a = n * np.log(2 * np.pi)
    _, b = np.linalg.slogdet(sigma)

    y = np.linalg.solve(sigma, x_i - mu)
    c = np.dot(x_i - mu, y)
    return -0.5 * (a + b + c)


def logsumexp(log_probs, axis=None):
    """
    Redefine scipy.special.logsumexp
    see: http://bayesjumping.net/log-sum-exp-trick/
    """
    _max = np.max(log_probs)
    ds = log_probs - _max
    exp_sum = np.exp(ds).sum(axis=axis)
    return _max + np.log(exp_sum)
