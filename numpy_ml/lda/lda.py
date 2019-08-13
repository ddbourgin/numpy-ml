import numpy as np
from scipy.special import digamma, polygamma, gammaln


class LDA(object):
    def __init__(self, T=10):
        """
        Vanilla (non-smoothed) LDA model trained using variational EM.
        Generates maximum-likelihood estimates for model paramters
        `alpha` and `beta`.

        Parameters
        ----------
        T : int
            Number of topics

        Attributes
        ----------
        D : int
            Number of documents
        N : list of length `D`
            Number of words in each document
        V : int
            Number of unique word tokens across all documents
        phi : :py:class:`ndarray <numpy.ndarray>` of shape `(D, N[d], T)`
            Variational approximation to word-topic distribution
        gamma : :py:class:`ndarray <numpy.ndarray>` of shape `(D, T)`
            Variational approximation to document-topic distribution
        alpha : :py:class:`ndarray <numpy.ndarray>` of shape `(1, T)`
            Parameter for the Dirichlet prior on the document-topic distribution
        beta  : :py:class:`ndarray <numpy.ndarray>` of shape `(V, T)`
            Word-topic distribution
        """
        self.T = T

    def _maximize_phi(self):
        """
        Optimize variational parameter phi
        ϕ_{t, n} ∝ β_{t, w_n}  e^( Ψ(γ_t) )
        """
        D = self.D
        N = self.N
        T = self.T

        phi = self.phi
        beta = self.beta
        gamma = self.gamma
        corpus = self.corpus

        for d in range(D):
            for n in range(N[d]):
                for t in range(T):
                    w_n = int(corpus[d][n])
                    phi[d][n, t] = beta[w_n, t] * np.exp(dg(gamma, d, t))

                # Normalize over topics
                phi[d][n, :] = phi[d][n, :] / np.sum(phi[d][n, :])
        return phi

    def _maximize_gamma(self):
        """
        Optimize variational parameter gamma
        γ_t = α_t + \sum_{n=1}^{N_d} ϕ_{t, n}
        """
        D = self.D
        phi = self.phi
        alpha = self.alpha

        gamma = np.tile(alpha, (D, 1)) + np.array(
            list(map(lambda x: np.sum(x, axis=0), phi))
        )
        return gamma

    def _maximize_beta(self):
        """
        Optimize model parameter beta
        β_{t, n} ∝ \sum_{d=1}^D \sum_{i=1}^{N_d} ϕ_{d, t, n} [ i = n]
        """
        T = self.T
        V = self.V

        phi = self.phi
        beta = self.beta
        corpus = self.corpus

        for n in range(V):
            # Construct binary mask [i == n] to be the same shape as phi
            mask = [np.tile((doc == n), (T, 1)).T for doc in corpus]
            beta[n, :] = np.sum(
                np.array(list(map(lambda x: np.sum(x, axis=0), phi * mask))), axis=0
            )

        # Normalize over words
        for t in range(T):
            beta[:, t] = beta[:, t] / np.sum(beta[:, t])

        return beta

    def _maximize_alpha(self, max_iters=1000, tol=0.1):
        """
        Optimize alpha using Blei's O(n) Newton-Raphson modification
        for a Hessian with special structure
        """
        D = self.D
        T = self.T

        alpha = self.alpha
        gamma = self.gamma

        for _ in range(max_iters):
            alpha_old = alpha

            #  Calculate gradient
            g = D * (digamma(np.sum(alpha)) - digamma(alpha)) + np.sum(
                digamma(gamma) - np.tile(digamma(np.sum(gamma, axis=1)), (T, 1)).T,
                axis=0,
            )

            #  Calculate Hessian diagonal component
            h = -D * polygamma(1, alpha)

            #  Calculate Hessian constant component
            z = D * polygamma(1, np.sum(alpha))

            #  Calculate constant
            c = np.sum(g / h) / (z ** (-1.0) + np.sum(h ** (-1.0)))

            #  Update alpha
            alpha = alpha - (g - c) / h

            #  Check convergence
            if np.sqrt(np.mean(np.square(alpha - alpha_old))) < tol:
                break

        return alpha

    def _E_step(self):
        """
        Maximize the VLB with respect to the variational parameters, γ and ϕ
        """
        self.phi = self._maximize_phi()
        self.gamma = self._maximize_gamma()

    def _M_step(self):
        """
        Maximize the VLB with respect to the model parameters, α and β
        """
        self.beta = self._maximize_beta()
        self.alpha = self._maximize_alpha()

    def VLB(self):
        """
        Return the variational lower bound associated with the current model
        parameters.
        """
        phi = self.phi
        alpha = self.alpha
        beta = self.beta
        gamma = self.gamma
        corpus = self.corpus

        D = self.D
        T = self.T
        N = self.N

        a, b, c, _d = 0, 0, 0, 0
        for d in range(D):
            a += (
                gammaln(np.sum(alpha))
                - np.sum(gammaln(alpha))
                + np.sum([(alpha[t] - 1) * dg(gamma, d, t) for t in range(T)])
            )

            _d += (
                gammaln(np.sum(gamma[d, :]))
                - np.sum(gammaln(gamma[d, :]))
                + np.sum([(gamma[d, t] - 1) * dg(gamma, d, t) for t in range(T)])
            )

            for n in range(N[d]):
                w_n = int(corpus[d][n])

                b += np.sum([phi[d][n, t] * dg(gamma, d, t) for t in range(T)])
                c += np.sum([phi[d][n, t] * np.log(beta[w_n, t]) for t in range(T)])
                _d += np.sum([phi[d][n, t] * np.log(phi[d][n, t]) for t in range(T)])

        return a + b + c - _d

    def initialize_parameters(self):
        """
        Provide reasonable initializations for model and variational parameters.
        """
        T = self.T
        V = self.V
        N = self.N
        D = self.D

        # initialize model parameters
        self.alpha = 100 * np.random.dirichlet(10 * np.ones(T), 1)[0]
        self.beta = np.random.dirichlet(np.ones(V), T).T

        # initialize variational parameters
        self.phi = np.array([1 / T * np.ones([N[d], T]) for d in range(D)])
        self.gamma = np.tile(self.alpha, (D, 1)) + np.tile(N / T, (T, 1)).T

    def train(self, corpus, verbose=False, max_iter=1000, tol=5):
        """
        Train the LDA model on a corpus of documents (bags of words).

        Parameters
        ----------
        corpus : list of length `D`
            A list of lists, with each sublist containing the tokenized text of
            a single document.
        verbose : bool
            Whether to print the VLB at each training iteration. Default is
            True.
        max_iter : int
            The maximum number of training iterations to perform before
            breaking. Default is 1000.
        tol : int
            Break the training loop if the difference betwen the VLB on the
            current iteration and the previous iteration is less than `tol`.
            Default is 5.
        """
        self.D = len(corpus)
        self.V = len(set(np.concatenate(corpus)))
        self.N = np.array([len(d) for d in corpus])
        self.corpus = corpus

        self.initialize_parameters()
        vlb = -np.inf

        for i in range(max_iter):
            old_vlb = vlb

            self._E_step()
            self._M_step()

            vlb = self.VLB()
            delta = vlb - old_vlb

            if verbose:
                print("Iteration {}: {:.3f} (delta: {:.2f})".format(i + 1, vlb, delta))

            if delta < tol:
                break


#######################################################################
#                                Utils                                #
#######################################################################


def dg(gamma, d, t):
    """
    E[log X_t] where X_t ~ Dir
    """
    return digamma(gamma[d, t]) - digamma(np.sum(gamma[d, :]))
