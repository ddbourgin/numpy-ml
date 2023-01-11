"""Hidden Markov model module"""

import numpy as np
from numpy_ml.utils.misc import logsumexp


class MultinomialHMM:
    def __init__(self, A=None, B=None, pi=None, eps=None):
        r"""
        A simple hidden Markov model with multinomial emission distribution.

        Parameters
        ----------
        A : :py:class:`ndarray <numpy.ndarray>` of shape `(N, N)` or None
            The transition matrix between latent states in the HMM. Index `i`,
            `j` gives the probability of transitioning from latent state `i` to
            latent state `j`. Default is None.
        B : :py:class:`ndarray <numpy.ndarray>` of shape `(N, V)` or None
            The emission matrix. Entry `i`, `j` gives the probability of latent
            state i emitting an observation of type `j`. Default is None.
        pi : :py:class:`ndarray <numpy.ndarray>` of shape `(N,)` or None
            The prior probability of each latent state. If None, use a uniform
            prior over states. Default is None.
        eps : float or None
            Epsilon value to avoid :math:`\log(0)` errors. If None, defaults to
            the machine epsilon. Default is None.

        Attributes
        ----------
        A : :py:class:`ndarray <numpy.ndarray>` of shape `(N, N)`
            The transition matrix between latent states in the HMM. Index `i`,
            `j` gives the probability of transitioning from latent state `i` to
            latent state `j`.
        B : :py:class:`ndarray <numpy.ndarray>` of shape `(N, V)`
            The emission matrix. Entry `i`, `j` gives the probability of latent
            state `i` emitting an observation of type `j`.
        N : int
            The number of unique latent states
        V : int
            The number of unique observation types
        O : :py:class:`ndarray <numpy.ndarray>` of shape `(I, T)`
            The collection of observed training sequences.
        I : int
            The number of sequences in `O`.
        T : int
            The number of observations in each sequence in `O`.
        """
        eps = np.finfo(float).eps if eps is None else eps

        # prior probability of each latent state
        if pi is not None:
            pi[pi == 0] = eps

        # number of latent state types
        N = None
        if A is not None:
            N = A.shape[0]
            A[A == 0] = eps

        # number of observation types
        V = None
        if B is not None:
            V = B.shape[1]
            B[B == 0] = eps

        self.parameters = {
            "A": A,  # transition matrix
            "B": B,  # emission matrix
            "pi": pi,  # prior probability of each latent state
        }

        self.hyperparameters = {
            "eps": eps,  # epsilon
        }

        self.derived_variables = {
            "N": N,  # number of latent state types
            "V": V,  # number of observation types
        }

    def generate(self, n_steps, latent_state_types, obs_types):
        """
        Sample a sequence from the HMM.

        Parameters
        ----------
        n_steps : int
            The length of the generated sequence
        latent_state_types : :py:class:`ndarray <numpy.ndarray>` of shape `(N,)`
            A collection of labels for the latent states
        obs_types : :py:class:`ndarray <numpy.ndarray>` of shape `(V,)`
            A collection of labels for the observations

        Returns
        -------
        states : :py:class:`ndarray <numpy.ndarray>` of shape `(n_steps,)`
            The sampled latent states.
        emissions : :py:class:`ndarray <numpy.ndarray>` of shape `(n_steps,)`
            The sampled emissions.
        """
        P = self.parameters
        A, B, pi = P["A"], P["B"], P["pi"]

        # sample the initial latent state
        s = np.random.multinomial(1, pi).argmax()
        states = [latent_state_types[s]]

        # generate an emission given latent state
        v = np.random.multinomial(1, B[s, :]).argmax()
        emissions = [obs_types[v]]

        # sample a latent transition, rinse, and repeat
        for i in range(n_steps - 1):
            s = np.random.multinomial(1, A[s, :]).argmax()
            states.append(latent_state_types[s])

            v = np.random.multinomial(1, B[s, :]).argmax()
            emissions.append(obs_types[v])

        return np.array(states), np.array(emissions)

    def log_likelihood(self, O):
        r"""
        Given the HMM parameterized by :math:`(A`, B, \pi)` and an observation
        sequence `O`, compute the marginal likelihood of `O`,
        :math:`P(O \mid A,B,\pi)`, by marginalizing over latent states.

        Notes
        -----
        The log likelihood is computed efficiently via DP using the forward
        algorithm, which produces a 2D trellis, ``forward`` (sometimes referred
        to as `alpha` in the literature), where entry `i`, `j` represents the
        probability under the HMM of being in latent state `i` after seeing the
        first `j` observations:

        .. math::

            \mathtt{forward[i,j]} = P(o_1, \ldots, o_j, q_j=i \mid A, B, \pi)

        Here :math:`q_j = i` indicates that the hidden state at time `j` is of
        type `i`.

        The DP step is:

        .. math::

            \mathtt{forward[i,j]}
               &= \sum_{s'=1}^N \mathtt{forward[s',j-1]} \cdot
                   \mathtt{A[s',i]} \cdot \mathtt{B[i,o_j]} \\
               &= \sum_{s'=1}^N P(o_1, \ldots, o_{j-1}, q_{j-1}=s' \mid A, B, \pi)
                    P(q_j=i \mid q_{j-1}=s') P(o_j \mid q_j=i)

        In words, ``forward[i,j]`` is the weighted sum of the values computed on
        the previous timestep. The weight on each previous state value is the
        product of the probability of transitioning from that state to state `i`
        and the probability of emitting observation `j` in state `i`.

        Parameters
        ----------
        O : :py:class:`ndarray <numpy.ndarray>` of shape `(1, T)`
            A single set of observations.

        Returns
        -------
        likelihood : float
            The likelihood of the observations `O` under the HMM.
        """
        if O.ndim == 1:
            O = O.reshape(1, -1)  # noqa: E741

        I, T = O.shape  # noqa: E741

        if I != 1:  # noqa: E741
            raise ValueError("Likelihood only accepts a single sequence")

        forward = self._forward(O[0])
        log_likelihood = logsumexp(forward[:, T - 1])
        return log_likelihood

    def decode(self, O):
        r"""
        Given the HMM parameterized by :math:`(A, B, \pi)` and an observation
        sequence :math:`O = o_1, \ldots, o_T`, compute the most probable
        sequence of latent states, :math:`Q = q_1, \ldots, q_T`.

        Notes
        -----
        HMM decoding is done efficiently via DP using the Viterbi algorithm,
        which produces a 2D trellis, ``viterbi``, where entry `i`, `j` represents the
        probability under the HMM of being in state `i` at time `j` after having
        passed through the *most probable* state sequence :math:`q_1,\ldots,q_{j-1}`:

        .. math::

            \mathtt{viterbi[i,j]} =
                \max_{q_1, \ldots, q_{j-1}}
                    P(o_1, \ldots, o_j, q_1, \ldots, q_{j-1}, q_j=i \mid A, B, \pi)

        Here :math:`q_j = i` indicates that the hidden state at time `j` is of
        type `i`, and :math:`\max_{q_1,\ldots,q_{j-1}}` represents the maximum over
        all possible latent state sequences for the first `j-1` observations.

        The DP step is:

        .. math::

            \mathtt{viterbi[i,j]} &=
                \max_{s'=1}^N \mathtt{viterbi[s',j-1]} \cdot
                    \mathtt{A[s',i]} \cdot \mathtt{B[i,o_j]} \\
               &=  \max_{s'=1}^N
                   P(o_1,\ldots, o_j, q_1, \ldots, q_{j-1}, q_j=i \mid A, B, \pi)
                   P(q_j=i \mid q_{j-1}=s') P(o_j \mid q_j=i)

        In words, ``viterbi[i,j]`` is the weighted sum of the values computed
        on the previous timestep. The weight on each value is the product of
        the probability of transitioning from that state to state `i` and the
        probability of emitting observation `j` in state `i`.

        To compute the most probable state sequence we maintain a second
        trellis, ``back_pointer``, whose `i`, `j` entry contains the value of the
        latent state at timestep `j-1` that is most likely to lead to latent
        state `i` at timestep `j`.

        When we have completed the ``viterbi`` and ``back_pointer`` trellises for
        all `T` timseteps/observations, we greedily move backwards through the
        ``back_pointer`` trellis to construct the best path for the full
        sequence of observations.

        Parameters
        ----------
        O : :py:class:`ndarray <numpy.ndarray>` of shape `(T,)`
            An observation sequence of length `T`.

        Returns
        -------
        best_path : list of length `T`
            The most probable sequence of latent states for observations `O`.
        best_path_prob : float
            The probability of the latent state sequence in `best_path` under
            the HMM.
        """
        P = self.parameters
        N = self.derived_variables["N"]
        eps = self.hyperparameters["eps"]
        A, B, pi = P["A"], P["B"], P["pi"]

        if O.ndim == 1:
            O = O.reshape(1, -1)  # noqa: E741

        # number of observations in each sequence
        T = O.shape[1]

        # number of training sequences
        I = O.shape[0]  # noqa: E741
        if I != 1:  # noqa: E741
            raise ValueError("Can only decode a single sequence (O.shape[0] must be 1)")

        # initialize the viterbi and back_pointer matrices
        viterbi = np.zeros((N, T))
        back_pointer = np.zeros((N, T)).astype(int)

        ot = O[0, 0]
        for s in range(N):
            back_pointer[s, 0] = 0
            viterbi[s, 0] = np.log(pi[s] + eps) + np.log(B[s, ot] + eps)

        for t in range(1, T):
            ot = O[0, t]
            for s in range(N):
                seq_probs = [
                    viterbi[s_, t - 1] + np.log(A[s_, s] + eps) + np.log(B[s, ot] + eps)
                    for s_ in range(N)
                ]

                viterbi[s, t] = np.max(seq_probs)
                back_pointer[s, t] = np.argmax(seq_probs)

        best_path_log_prob = viterbi[:, T - 1].max()

        # backtrack through the trellis to get the most likely sequence of
        # latent states
        pointer = viterbi[:, T - 1].argmax()
        best_path = [pointer]
        for t in reversed(range(1, T)):
            pointer = back_pointer[pointer, t]
            best_path.append(pointer)
        best_path = best_path[::-1]

        return best_path, best_path_log_prob

    def _forward(self, Obs):
        r"""
        Computes the forward probability trellis for an HMM parameterized by
        :math:`(A, B, \pi)`.

        Notes
        -----
        The forward trellis (sometimes referred to as `alpha` in the HMM
        literature), is a 2D array where entry `i`, `j` represents the probability
        under the HMM of being in latent state `i` after seeing the first `j`
        observations:

        .. math::

            \mathtt{forward[i,j]} =
                P(o_1, \ldots, o_j, q_j=i \mid A, B, \pi)

        Here :math:`q_j = i` indicates that the hidden state at time `j` is of
        type `i`.

        The DP step is::

        .. math::

            forward[i,j] &=
                \sum_{s'=1}^N forward[s',j-1] \times A[s',i] \times B[i,o_j] \\
                &= \sum_{s'=1}^N P(o_1, \ldots, o_{j-1}, q_{j-1}=s' \mid A, B, \pi)
                    \times P(q_j=i \mid q_{j-1}=s') \times P(o_j \mid q_j=i)

        In words, ``forward[i,j]`` is the weighted sum of the values computed
        on the previous timestep. The weight on each previous state value is
        the product of the probability of transitioning from that state to
        state `i` and the probability of emitting observation `j` in state `i`.

        Parameters
        ----------
        Obs : :py:class:`ndarray <numpy.ndarray>` of shape `(T,)`
            An observation sequence of length `T`.

        Returns
        -------
        forward : :py:class:`ndarray <numpy.ndarray>` of shape `(N, T)`
            The forward trellis.
        """
        P = self.parameters
        N = self.derived_variables["N"]
        eps = self.hyperparameters["eps"]
        A, B, pi = P["A"], P["B"], P["pi"]

        T = Obs.shape[0]

        # initialize the forward probability matrix
        forward = np.zeros((N, T))

        ot = Obs[0]
        for s in range(N):
            forward[s, 0] = np.log(pi[s] + eps) + np.log(B[s, ot] + eps)

        for t in range(1, T):
            ot = Obs[t]
            for s in range(N):
                forward[s, t] = logsumexp(
                    [
                        forward[s_, t - 1]
                        + np.log(A[s_, s] + eps)
                        + np.log(B[s, ot] + eps)
                        for s_ in range(N)
                    ]  # noqa: C812
                )
        return forward

    def _backward(self, Obs):
        r"""
        Compute the backward probability trellis for an HMM parameterized by
        :math:`(A, B, \pi)`.

        Notes
        -----
        The backward trellis (sometimes referred to as `beta` in the HMM
        literature), is a 2D array where entry `i`,`j` represents the probability
        of seeing the observations from time `j+1` onward given that the HMM is
        in state `i` at time `j`

        .. math::

            \mathtt{backward[i,j]} = P(o_{j+1},o_{j+2},\ldots,o_T \mid q_j=i,A,B,\pi)

        Here :math:`q_j = i` indicates that the hidden state at time `j` is of type `i`.

        The DP step is::

            backward[i,j] &=
                \sum_{s'=1}^N backward[s',j+1] \times A[i, s'] \times B[s',o_{j+1}] \\
                &= \sum_{s'=1}^N P(o_{j+1}, o_{j+2}, \ldots, o_T \mid q_j=i, A, B, pi)
                    \times P(q_{j+1}=s' \mid q_{j}=i) \times P(o_{j+1} \mid q_{j+1}=s')

        In words, ``backward[i,j]`` is the weighted sum of the values computed
        on the following timestep. The weight on each state value from the
        `j+1`'th timestep is the product of the probability of transitioning from
        state i to that state and the probability of emitting observation `j+1`
        from that state.

        Parameters
        ----------
        Obs : :py:class:`ndarray <numpy.ndarray>` of shape `(T,)`
            A single observation sequence of length `T`.

        Returns
        -------
        backward : :py:class:`ndarray <numpy.ndarray>` of shape `(N, T)`
            The backward trellis.
        """
        P = self.parameters
        A, B = P["A"], P["B"]
        N = self.derived_variables["N"]
        eps = self.hyperparameters["eps"]

        T = Obs.shape[0]

        # initialize the backward trellis
        backward = np.zeros((N, T))

        for s in range(N):
            backward[s, T - 1] = 0

        for t in reversed(range(T - 1)):
            ot1 = Obs[t + 1]
            for s in range(N):
                backward[s, t] = logsumexp(
                    [
                        np.log(A[s, s_] + eps)
                        + np.log(B[s_, ot1] + eps)
                        + backward[s_, t + 1]
                        for s_ in range(N)
                    ]  # noqa: C812
                )
        return backward

    def _initialize_parameters(self):
        P = self.parameters
        A, B, pi = P["A"], P["B"], P["pi"]
        N, V = self.derived_variables["N"], self.derived_variables["V"]

        # Uniform initialization of prior over latent states
        if pi is None:
            pi = np.ones(N)
            pi = pi / pi.sum()

        # Uniform initialization of A
        if A is None:
            A = np.ones((N, N))
            A = A / A.sum(axis=1)[:, None]

        # Random initialization of B
        if B is None:
            B = np.random.rand(N, V)
            B = B / B.sum(axis=1)[:, None]

        P["A"], P["B"], P["pi"] = A, B, pi

    def fit(
        self,
        O,
        latent_state_types,
        observation_types,
        pi=None,
        tol=1e-5,
        verbose=False,
    ):
        """
        Given an observation sequence `O` and the set of possible latent states,
        learn the MLE HMM parameters `A` and `B`.

        Notes
        -----
        Model fitting is done iterativly using the Baum-Welch/Forward-Backward
        algorithm, a special case of the EM algorithm.

        We begin with an intial estimate for the transition (`A`) and emission
        (`B`) matrices and then use these to derive better and better estimates
        by computing the forward probability for an observation and then
        dividing that probability mass among all the paths that contributed to
        it.

        Parameters
        ----------
        O : :py:class:`ndarray <numpy.ndarray>` of shape `(I, T)`
            The set of `I` training observations, each of length `T`.
        latent_state_types : list of length `N`
            The collection of valid latent states.
        observation_types : list of length `V`
            The collection of valid observation states.
        pi : :py:class:`ndarray <numpy.ndarray>` of shape `(N,)`
            The prior probability of each latent state. If None, assume each
            latent state is equally likely a priori. Default is None.
        tol : float
            The tolerance value. If the difference in log likelihood between
            two epochs is less than this value, terminate training. Default is
            1e-5.
        verbose : bool
            Print training stats after each epoch. Default is True.

        Returns
        -------
        A : :py:class:`ndarray <numpy.ndarray>` of shape `(N, N)`
            The estimated transition matrix.
        B : :py:class:`ndarray <numpy.ndarray>` of shape `(N, V)`
            The estimated emission matrix.
        pi : :py:class:`ndarray <numpy.ndarray>` of shape `(N,)`
            The estimated prior probabilities of each latent state.
        """
        # observations
        if O.ndim == 1:
            O = O.reshape(1, -1)  # noqa: E741

        # number of training examples (I) and their lengths (T)
        I, T = O.shape

        # number of types of observation
        self.derived_variables["V"] = len(observation_types)

        # number of latent state types
        self.derived_variables["N"] = len(latent_state_types)

        self._initialize_parameters()

        P = self.parameters

        # iterate E and M steps until convergence criteria is met
        step, delta = 0, np.inf
        ll_prev = np.sum([self.log_likelihood(o) for o in O])

        while delta > tol:
            gamma, xi, phi = self._E_step(O)
            P["A"], P["B"], P["pi"] = self._M_step(O, gamma, xi, phi)
            ll = np.sum([self.log_likelihood(o) for o in O])
            delta = ll - ll_prev
            ll_prev = ll
            step += 1

            if verbose:
                fstr = "[Epoch {}] LL: {:.3f} Delta: {:.5f}"
                print(fstr.format(step, ll_prev, delta))

        #  return A, B, pi

    def _E_step(self, O):
        r"""
        Run a single E-step update for the Baum-Welch/Forward-Backward
        algorithm. This step estimates ``xi`` and ``gamma``, the excepted
        state-state transition counts and the expected state-occupancy counts,
        respectively.

        ``xi[i,j,k]`` gives the probability of being in state `i` at time `k`
        and state `j` at time `k+1` given the observed sequence `O` and the
        current estimates for transition (`A`) and emission (`B`) matrices::

        .. math::

            xi[i,j,k] &= P(q_k=i,q_{k+1}=j \mid O,A,B,pi) \\
                      &= \frac{
                            P(q_k=i,q_{k+1}=j,O \mid A,B,pi)
                         }{P(O \mid A,B,pi)} \\
                      &= \frac{
                            P(o_1,o_2,\ldots,o_k,q_k=i \mid A,B,pi) \times
                            P(q_{k+1}=j \mid q_k=i) \times
                            P(o_{k+1} \mid q_{k+1}=j) \times
                            P(o_{k+2},o_{k+3},\ldots,o_T \mid q_{k+1}=j,A,B,pi)
                         }{P(O \mid A,B,pi)} \\
                      &= \frac{
                            \mathtt{fwd[j, k] * self.A[j, i] *
                            self.B[i, o_{k+1}] * bwd[i, k + 1]}
                         }{\mathtt{fwd[:, T].sum()}}

        The expected number of transitions from state `i` to state `j` across the
        entire sequence is then the sum over all timesteps: ``xi[i,j,:].sum()``.

        ``gamma[i,j]`` gives the probability of being in state `i` at time `j`

        .. math:: \mathtt{gamma[i,j]} = P(q_j = i \mid O, A, B, \pi)

        Parameters
        ----------
        O : :py:class:`ndarray <numpy.ndarray>` of shape `(I, T)`
            The set of `I` training observations, each of length `T`.

        Returns
        -------
        gamma : :py:class:`ndarray <numpy.ndarray>` of shape `(I, N, T)`
            The estimated state-occupancy count matrix.
        xi : :py:class:`ndarray <numpy.ndarray>` of shape `(I, N, N, T)`
            The estimated state-state transition count matrix.
        phi : :py:class:`ndarray <numpy.ndarray>` of shape `(I, N)`
            The estimated prior counts for each latent state.
        """
        I, T = O.shape
        P = self.parameters
        A, B = P["A"], P["B"]
        N = self.derived_variables["N"]
        eps = self.hyperparameters["eps"]

        phi = np.zeros((I, N))
        gamma = np.zeros((I, N, T))
        xi = np.zeros((I, N, N, T))

        for i in range(I):
            Obs = O[i, :]
            fwd = self._forward(Obs)
            bwd = self._backward(Obs)
            log_likelihood = logsumexp(fwd[:, T - 1])

            t = T - 1
            for si in range(N):
                gamma[i, si, t] = fwd[si, t] + bwd[si, t] - log_likelihood
                phi[i, si] = fwd[si, 0] + bwd[si, 0] - log_likelihood

            for t in range(T - 1):
                ot1 = Obs[t + 1]
                for si in range(N):
                    gamma[i, si, t] = fwd[si, t] + bwd[si, t] - log_likelihood
                    for sj in range(N):
                        xi[i, si, sj, t] = (
                            fwd[si, t]
                            + np.log(A[si, sj] + eps)
                            + np.log(B[sj, ot1] + eps)
                            + bwd[sj, t + 1]
                            - log_likelihood
                        )

        return gamma, xi, phi

    def _M_step(self, O, gamma, xi, phi):
        """
        Run a single M-step update for the Baum-Welch/Forward-Backward
        algorithm.

        Parameters
        ----------
        O : :py:class:`ndarray <numpy.ndarray>` of shape `(I, T)`
            The set of `I` training observations, each of length `T`.
        gamma : :py:class:`ndarray <numpy.ndarray>` of shape `(I, N, T)`
            The estimated state-occupancy count matrix.
        xi : :py:class:`ndarray <numpy.ndarray>` of shape `(I, N, N, T)`
            The estimated state-state transition count matrix.
        phi : :py:class:`ndarray <numpy.ndarray>` of shape `(I, N)`
            The estimated starting count matrix for each latent state.

        Returns
        -------
        A : :py:class:`ndarray <numpy.ndarray>` of shape `(N, N)`
            The estimated transition matrix.
        B : :py:class:`ndarray <numpy.ndarray>` of shape `(N, V)`
            The estimated emission matrix.
        pi : :py:class:`ndarray <numpy.ndarray>` of shape `(N,)`
            The estimated prior probabilities for each latent state.
        """
        I, T = O.shape
        P = self.parameters
        DV = self.derived_variables
        eps = self.hyperparameters["eps"]

        N, V = DV["N"], DV["V"]
        A, B, pi = P["A"], P["B"], P["pi"]

        # initialize the estimated transition (A) and emission (B) matrices
        A = np.zeros((N, N))
        B = np.zeros((N, V))
        pi = np.zeros(N)

        count_gamma = np.zeros((I, N, V))
        count_xi = np.zeros((I, N, N))

        for i in range(I):
            Obs = O[i, :]
            for si in range(N):
                for vk in range(V):
                    if not (Obs == vk).any():
                        count_gamma[i, si, vk] = np.log(eps)
                    else:
                        count_gamma[i, si, vk] = logsumexp(gamma[i, si, Obs == vk])

                for sj in range(N):
                    count_xi[i, si, sj] = logsumexp(xi[i, si, sj, :])

        pi = logsumexp(phi, axis=0) - np.log(I + eps)
        np.testing.assert_almost_equal(np.exp(pi).sum(), 1)

        for si in range(N):
            for vk in range(V):
                B[si, vk] = logsumexp(count_gamma[:, si, vk]) - logsumexp(
                    count_gamma[:, si, :]  # noqa: C812
                )

            for sj in range(N):
                A[si, sj] = logsumexp(count_xi[:, si, sj]) - logsumexp(
                    count_xi[:, si, :]  # noqa: C812
                )

            np.testing.assert_almost_equal(np.exp(A[si, :]).sum(), 1)
            np.testing.assert_almost_equal(np.exp(B[si, :]).sum(), 1)
        return np.exp(A), np.exp(B), np.exp(pi)
