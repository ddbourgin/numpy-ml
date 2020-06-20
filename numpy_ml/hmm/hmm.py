"""Hidden Markov model module"""

import numpy as np


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
        self.eps = np.finfo(float).eps if eps is None else eps

        # transition matrix
        self.A = A

        # emission matrix
        self.B = B

        # prior probability of each latent state
        self.pi = pi
        if self.pi is not None:
            self.pi[self.pi == 0] = self.eps

        # number of latent state types
        self.N = None
        if self.A is not None:
            self.N = self.A.shape[0]
            self.A[self.A == 0] = self.eps

        # number of observation types
        self.V = None
        if self.B is not None:
            self.V = self.B.shape[1]
            self.B[self.B == 0] = self.eps

        # set of training sequences
        self.O = None  # noqa: E741

        # number of sequences in O
        self.I = None  # noqa: E741

        # number of observations in each sequence
        self.T = None

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
        # sample the initial latent state
        s = np.random.multinomial(1, self.pi).argmax()
        states = [latent_state_types[s]]

        # generate an emission given latent state
        v = np.random.multinomial(1, self.B[s, :]).argmax()
        emissions = [obs_types[v]]

        # sample a latent transition, rinse, and repeat
        for i in range(n_steps - 1):
            s = np.random.multinomial(1, self.A[s, :]).argmax()
            states.append(latent_state_types[s])

            v = np.random.multinomial(1, self.B[s, :]).argmax()
            emissions.append(obs_types[v])

        return np.array(states), np.array(emissions)

    def log_likelihood(self, O):
        r"""
        Given the HMM parameterized by :math:`(A`, B, \pi)` and an observation
        sequence `O`, compute the marginal likelihood of the observations:
        :math:`P(O \mid A,B,\pi)`, summing over latent states.

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
        eps = self.eps

        if O.ndim == 1:
            O = O.reshape(1, -1)  # noqa: E741

        # number of observations in each sequence
        T = O.shape[1]

        # number of training sequences
        I = O.shape[0]  # noqa: E741
        if I != 1:  # noqa: E741
            raise ValueError("Can only decode a single sequence (O.shape[0] must be 1)")

        # initialize the viterbi and back_pointer matrices
        viterbi = np.zeros((self.N, T))
        back_pointer = np.zeros((self.N, T)).astype(int)

        ot = O[0, 0]
        for s in range(self.N):
            back_pointer[s, 0] = 0
            viterbi[s, 0] = np.log(self.pi[s] + eps) + np.log(self.B[s, ot] + eps)

        for t in range(1, T):
            ot = O[0, t]
            for s in range(self.N):
                seq_probs = [
                    viterbi[s_, t - 1]
                    + np.log(self.A[s_, s] + eps)
                    + np.log(self.B[s, ot] + eps)
                    for s_ in range(self.N)
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
        eps = self.eps
        T = Obs.shape[0]

        # initialize the forward probability matrix
        forward = np.zeros((self.N, T))

        ot = Obs[0]
        for s in range(self.N):
            forward[s, 0] = np.log(self.pi[s] + eps) + np.log(self.B[s, ot] + eps)

        for t in range(1, T):
            ot = Obs[t]
            for s in range(self.N):
                forward[s, t] = logsumexp(
                    [
                        forward[s_, t - 1]
                        + np.log(self.A[s_, s] + eps)
                        + np.log(self.B[s, ot] + eps)
                        for s_ in range(self.N)
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
        eps = self.eps
        T = Obs.shape[0]

        # initialize the backward trellis
        backward = np.zeros((self.N, T))

        for s in range(self.N):
            backward[s, T - 1] = 0

        for t in reversed(range(T - 1)):
            ot1 = Obs[t + 1]
            for s in range(self.N):
                backward[s, t] = logsumexp(
                    [
                        np.log(self.A[s, s_] + eps)
                        + np.log(self.B[s_, ot1] + eps)
                        + backward[s_, t + 1]
                        for s_ in range(self.N)
                    ]  # noqa: C812
                )
        return backward

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
        if O.ndim == 1:
            O = O.reshape(1, -1)  # noqa: E741

        # observations
        self.O = O  # noqa: E741

        # number of training examples (I) and their lengths (T)
        self.I, self.T = self.O.shape

        # number of types of observation
        self.V = len(observation_types)

        # number of latent state types
        self.N = len(latent_state_types)

        # Uniform initialization of prior over latent states
        self.pi = pi
        if self.pi is None:
            self.pi = np.ones(self.N)
            self.pi = self.pi / self.pi.sum()

        # Uniform initialization of A
        self.A = np.ones((self.N, self.N))
        self.A = self.A / self.A.sum(axis=1)[:, None]

        # Random initialization of B
        self.B = np.random.rand(self.N, self.V)
        self.B = self.B / self.B.sum(axis=1)[:, None]

        # iterate E and M steps until convergence criteria is met
        step, delta = 0, np.inf
        ll_prev = np.sum([self.log_likelihood(o) for o in self.O])
        while delta > tol:
            gamma, xi, phi = self._Estep()
            self.A, self.B, self.pi = self._Mstep(gamma, xi, phi)
            ll = np.sum([self.log_likelihood(o) for o in self.O])
            delta = ll - ll_prev
            ll_prev = ll
            step += 1

            if verbose:
                fstr = "[Epoch {}] LL: {:.3f} Delta: {:.5f}"
                print(fstr.format(step, ll_prev, delta))

        return self.A, self.B, self.pi

    def _Estep(self):
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

        Returns
        -------
        gamma : :py:class:`ndarray <numpy.ndarray>` of shape `(I, N, T)`
            The estimated state-occupancy count matrix.
        xi : :py:class:`ndarray <numpy.ndarray>` of shape `(I, N, N, T)`
            The estimated state-state transition count matrix.
        phi : :py:class:`ndarray <numpy.ndarray>` of shape `(I, N)`
            The estimated prior counts for each latent state.
        """
        eps = self.eps

        gamma = np.zeros((self.I, self.N, self.T))
        xi = np.zeros((self.I, self.N, self.N, self.T))
        phi = np.zeros((self.I, self.N))

        for i in range(self.I):
            Obs = self.O[i, :]
            fwd = self._forward(Obs)
            bwd = self._backward(Obs)
            log_likelihood = logsumexp(fwd[:, self.T - 1])

            t = self.T - 1
            for si in range(self.N):
                gamma[i, si, t] = fwd[si, t] + bwd[si, t] - log_likelihood
                phi[i, si] = fwd[si, 0] + bwd[si, 0] - log_likelihood

            for t in range(self.T - 1):
                ot1 = Obs[t + 1]
                for si in range(self.N):
                    gamma[i, si, t] = fwd[si, t] + bwd[si, t] - log_likelihood
                    for sj in range(self.N):
                        xi[i, si, sj, t] = (
                            fwd[si, t]
                            + np.log(self.A[si, sj] + eps)
                            + np.log(self.B[sj, ot1] + eps)
                            + bwd[sj, t + 1]
                            - log_likelihood
                        )

        return gamma, xi, phi

    def _Mstep(self, gamma, xi, phi):
        """
        Run a single M-step update for the Baum-Welch/Forward-Backward
        algorithm.

        Parameters
        ----------
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
        eps = self.eps

        # initialize the estimated transition (A) and emission (B) matrices
        A = np.zeros((self.N, self.N))
        B = np.zeros((self.N, self.V))
        pi = np.zeros(self.N)

        count_gamma = np.zeros((self.I, self.N, self.V))
        count_xi = np.zeros((self.I, self.N, self.N))

        for i in range(self.I):
            Obs = self.O[i, :]
            for si in range(self.N):
                for vk in range(self.V):
                    if not (Obs == vk).any():
                        #  count_gamma[i, si, vk] = -np.inf
                        count_gamma[i, si, vk] = np.log(eps)
                    else:
                        count_gamma[i, si, vk] = logsumexp(gamma[i, si, Obs == vk])

                for sj in range(self.N):
                    count_xi[i, si, sj] = logsumexp(xi[i, si, sj, :])

        pi = logsumexp(phi, axis=0) - np.log(self.I + eps)
        np.testing.assert_almost_equal(np.exp(pi).sum(), 1)

        for si in range(self.N):
            for vk in range(self.V):
                B[si, vk] = logsumexp(count_gamma[:, si, vk]) - logsumexp(
                    count_gamma[:, si, :]  # noqa: C812
                )

            for sj in range(self.N):
                A[si, sj] = logsumexp(count_xi[:, si, sj]) - logsumexp(
                    count_xi[:, si, :]  # noqa: C812
                )

            np.testing.assert_almost_equal(np.exp(A[si, :]).sum(), 1)
            np.testing.assert_almost_equal(np.exp(B[si, :]).sum(), 1)
        return np.exp(A), np.exp(B), np.exp(pi)


#######################################################################
#                                Utils                                #
#######################################################################


def logsumexp(log_probs, axis=None):
    """
    Redefine scipy.special.logsumexp
    see: http://bayesjumping.net/log-sum-exp-trick/
    """
    _max = np.max(log_probs)
    ds = log_probs - _max
    exp_sum = np.exp(ds).sum(axis=axis)
    return _max + np.log(exp_sum)
