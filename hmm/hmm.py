import numpy as np


class HMM:
    def __init__(self, A=None, B=None, pi=None):
        """
        A simple HMM model for discrete observation spaces.

        Parameters
        ----------
        A : numpy array of shape (N, N) (default: None)
            The transition matrix between latent states in the HMM. Index i,j
            gives the probability of transitioning from latent state i to
            latent state j.
        B : numpy array of shape (N, V) (default: None)
            The emission matrix. Entry i,j gives the probability of latent
            state i emitting an observation of type j.
        pi : numpy array of shape (N,) (default: None)
            The prior probability of each latent state.
        """
        eps = np.finfo(float).eps

        # transition matrix
        self.A = A
        self.A[self.A == 0] = eps

        # emission matrix
        self.B = B
        self.B[self.B == 0] = eps

        # prior probability of each latent state
        self.pi = pi
        self.pi[self.pi == 0] = eps

        # number of latent state types
        self.N = None
        if self.A is not None:
            self.N = self.A.shape[0]

        # number of observation types
        self.V = None
        if self.B is not None:
            self.V = self.B.shape[1]

        # set of training sequences
        self.O = None

        # number of sequences in O
        self.I = None

        # number of observations in each sequence
        self.T = None

    def generate(self, n_steps, latent_state_types, obs_types):
        """
        Sample sequences from the HMM.
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
        """
        Given the HMM parameterized by (A, B, pi) and an observation sequence
        O, compute the marginal likelihood of the observations: P(O|A,B,pi),
        summing over latent states.

        This is done efficiently via DP using the forward algorithm, which
        produces a 2D trellis, `forward` (sometimes referred to as `alpha` in the
        literature), where entry i,j represents the probability under the HMM
        of being in latent state i after seeing the first j observations:

            forward[i,j] = P(o_1,o_2,...,o_j,q_j=i|A,B,pi)

        Here q_j = i indicates that the hidden state at time j is of type i.

        The DP step is:

            forward[i,j] = sum_{s'=1}^N forward[s',j-1] * A[s',i] * B[i,o_j]
                         = sum_{s'=1}^N P(o_1,o_2,...,o_{j-1},q_{j-1}=s'|A,B,pi) *
                           P(q_j=i|q_{j-1}=s') * P(o_j|q_j=i)

        In words, forward[i,j] is the weighted sum of the values computed on
        the previous timestep. The weight on each previous state value is the
        product of the probability of transitioning from that state to state i
        and the probability of emitting observation j in state i.

        Parameters
        ----------
        O : np.array of shape (1, T)
            A single set of observations.

        Returns
        -------
        likelihood : float
            The likelihood of the observations O under the HMM.
        """
        if O.ndim == 1:
            O = O.reshape(1, -1)

        self.O = O
        self.I, self.T = self.O.shape

        if self.I != 1:
            raise ValueError("Likelihood only accepts a single sequence")

        forward = self._forward(self.O[0])
        log_likelihood = logsumexp(forward[:, self.T - 1])
        return log_likelihood

    def decode(self, O):
        """
        Given the HMM parameterized by (A, B, pi) and an observation sequence O
        = o_1, ..., o_T, compute the most probable sequence of latent states, Q
        = q_1, ..., q_T.

        This is done efficiently via DP using the Viterbi algorithm, which
        produces a 2D trellis, `viterbi`, where entry i,j represents the
        probability under the HMM of being in state i at time j after having
        passed through the *most probable* state sequence q_1,...,q_{j-1}:

            viterbi[i,j] = max_{q_1,...,q_{j-1}} P(o_1,...,o_j,q_1,...,q_{j-1},q_j=i|A,B,pi)

        Here q_j = i indicates that the hidden state at time j is of type i,
        and max_{q_1,...,q_{j-1}} represents the maximum over all possible
        latent state sequences for the first j-1 observations.

        The DP step is:

            viterbi[i,j] = max_{s'=1}^N viterbi[s',j-1] * A[s',i] * B[i,o_j]
                         = max_{s'=1}^N P(o_1,...,o_j,q_1,...,q_{j-1},q_j=i|A,B,pi) *
                           P(q_j=i|q_{j-1}=s') * P(o_j|q_j=i)

        In words, viterbi[i,j] is the weighted sum of the values computed on
        the previous timestep. The weight on each value is the product of the
        probability of transitioning from that state to state i and the
        probability of emitting observation j in state i.

        To compute the most probable state sequence we maintain a second
        trellis, `back_pointer`, whose i,j entry contains the value of the
        latent state at timestep j-1 that is most likely to lead to latent
        state i at timestep j.

        When we have completed the `viterbi` and `back_pointer` trellises for
        all T timseteps/observations, we greedily move backwards through the
        `back_pointer` trellis to construct the best path for the full sequence
        of observations.

        Parameters
        ----------
        O : np.array of shape (T,)
            An observation sequence of length T

        Returns
        -------
        best_path : list of length T
            The most probable sequence of latent states for observations O
        best_path_prob : float
            The probability of the latent state sequence in `best_path` under
            the HMM
        """
        if O.ndim == 1:
            O = O.reshape(1, -1)

        # observations
        self.O = O

        # number of observations in each sequence
        self.T = self.O.shape[1]

        # number of training sequences
        self.I = self.O.shape[0]
        if self.I != 1:
            raise ValueError("Can only decode a single sequence (O.shape[0] must be 1)")

        # initialize the viterbi and back_pointer matrices
        viterbi = np.zeros((self.N, self.T))
        back_pointer = np.zeros((self.N, self.T)).astype(int)

        ot = self.O[0, 0]
        for s in range(self.N):
            back_pointer[s, 0] = 0
            viterbi[s, 0] = np.log(self.pi[s]) + np.log(self.B[s, ot])

        for t in range(1, self.T):
            ot = self.O[0, t]
            for s in range(self.N):
                seq_probs = [
                    viterbi[s_, t - 1] + np.log(self.A[s_, s]) + np.log(self.B[s, ot])
                    for s_ in range(self.N)
                ]

                viterbi[s, t] = np.max(seq_probs)
                back_pointer[s, t] = np.argmax(seq_probs)

        best_path_log_prob = viterbi[:, self.T - 1].max()

        # backtrack through the trellis to get the most likely sequence of
        # latent states
        pointer = viterbi[:, self.T - 1].argmax()
        best_path = [pointer]
        for t in reversed(range(1, self.T)):
            pointer = back_pointer[pointer, t]
            best_path.append(pointer)
        best_path = best_path[::-1]
        return best_path, best_path_log_prob

    def _forward(self, Obs):
        """
        Computes the forward probability trellis for an HMM parameterized by
        (A, B, pi). `forward` (sometimes referred to as `alpha` in the HMM
        literature), is a 2D trellis where entry i,j represents the probability
        under the HMM of being in latent state i after seeing the first j
        observations:

            forward[i,j] = P(o_1,o_2,...,o_j,q_j=i|A,B,pi)

        Here q_j = i indicates that the hidden state at time j is of type i.

        The DP step is:

            forward[i,j] = sum_{s'=1}^N forward[s',j-1] * A[s',i] * B[i,o_j]
                         = sum_{s'=1}^N P(o_1,o_2,...,o_{j-1},q_{j-1}=s'|A,B,pi) *
                           P(q_j=i|q_{j-1}=s') * P(o_j|q_j=i)

        In words, forward[i,j] is the weighted sum of the values computed on
        the previous timestep. The weight on each previous state value is the
        product of the probability of transitioning from that state to state i
        and the probability of emitting observation j in state i.

        Parameters
        ----------
        Obs : numpy array of shape (T,)
            An observation sequence of length T

        Returns
        -------
        forward : numpy array of shape (N, T)
            The forward trellis
        """
        # initialize the forward probability matrix
        forward = np.zeros((self.N, self.T))

        ot = Obs[0]
        for s in range(self.N):
            forward[s, 0] = np.log(self.pi[s]) + np.log(self.B[s, ot])

        for t in range(1, self.T):
            ot = Obs[t]
            for s in range(self.N):
                forward[s, t] = logsumexp(
                    [
                        forward[s_, t - 1]
                        + np.log(self.A[s_, s])
                        + np.log(self.B[s, ot])
                        for s_ in range(self.N)
                    ]
                )
        return forward

    def _backward(self, Obs):
        """
        Computes the backward probability trellis for an HMM parameterized by
        (A, B, pi). `backward` (sometimes referred to as `beta` in the HMM
        literature), is a 2D trellis where entry i,j represents the probability
        of seeing the observations from time j+1 onward given that the HMM
        is in state i at time j:

            backward[i,j] = P(o_{j+1},o_{j+2},...,o_T|q_j=i,A,B,pi)

        Here q_j = i indicates that the hidden state at time j is of type i.

        The DP step is:

            backward[i,j] = sum_{s'=1}^N backward[s',j+1] * A[i, s'] * B[s',o_{j+1}]
                          = sum_{s'=1}^N P(o_{j+1},o_{j+2},...,o_T|q_j=i,A,B,pi) *
                            P(q_{j+1}=s'|q_{j}=i) * P(o_{j+1}|q_{j+1}=s')

        In words, backward[i,j] is the weighted sum of the values computed on
        the following timestep. The weight on each state value from the j+1'th
        timestep is the product of the probability of transitioning from state
        i to that state and the probability of emitting observation j+1 from
        that state.

        Parameters
        ----------
        Obs : numpy array of shape (T,)
            A single observation sequence of length T

        Returns
        -------
        backward : numpy array of shape (N, T)
            The backward trellis
        """
        # initialize the backward trellis
        backward = np.zeros((self.N, self.T))

        for s in range(self.N):
            backward[s, self.T - 1] = 0

        for t in reversed(range(self.T - 1)):
            ot1 = Obs[t + 1]
            for s in range(self.N):
                backward[s, t] = logsumexp(
                    [
                        np.log(self.A[s, s_])
                        + np.log(self.B[s_, ot1])
                        + backward[s_, t + 1]
                        for s_ in range(self.N)
                    ]
                )
        return backward

    def fit(
        self, O, latent_state_types, observation_types, pi=None, tol=1e-5, verbose=False
    ):
        """
        Given an observation sequence O and the set of possible latent states,
        learn the MLE HMM parameters A and B.

        This is done iterativly using the Baum-Welch/Forward-Backward
        algorithm, a special case of the EM algorithm. We start with an intial
        estimate for the transition (A) and emission (B) matrices and then use
        this to derive better and better estimates by computing the forward
        probability for an observation and then dividing that probability mass
        among all the different paths that contributed to it.

        Parameters
        ----------
        O : np.array of shape (I, T)
            The set of I training observations, each of length T
        latent_state_types : list of length N
            The collection of valid latent states
        observation_types : list of length V
            The collection of valid observation states
        pi : numpy array of shape (N,)
            The prior probability of each latent state. If None, assume each
            latent state is equally likely a priori

        Returns
        -------
        A : numpy array of shape (N, N)
            The estimated transition matrix
        B : numpy array of shape (N, V)
            The estimated emission matrix
        pi : numpy array of shape (N,)
            The estimated prior probabilities of each latent state
        """
        if O.ndim == 1:
            O = O.reshape(1, -1)

        # observations
        self.O = O

        # number of observations
        self.T = self.O.shape[1]

        # number of types of observation
        self.V = len(observation_types)

        # number of latent state types
        self.N = len(latent_state_types)

        # initialize the prior over latent states
        self.pi = pi
        if self.pi is None:
            self.pi = np.random.rand(self.N)
            self.pi = self.pi / self.pi.sum()

        # Randomly intialize A and B matrices, ensuring that the rows sum to 1
        self.A = np.random.rand(self.N, self.N)
        self.A = self.A / self.A.sum(axis=1)[:, None]

        self.B = np.random.rand(self.N, self.V)
        self.B = self.B / self.B.sum(axis=1)[:, None]

        A_ = np.zeros((self.N, self.N))
        B_ = np.zeros((self.N, self.V))
        pi_ = np.zeros(self.N)

        # iterate E and M steps until convergence criteria is met
        step = 0

        def squared_error(x, y):
            return np.sqrt(((x - y) ** 2).mean())

        A_err, B_err = squared_error(A_, self.A), squared_error(B_, self.B)
        pi_err = squared_error(pi_, self.pi)
        while any([A_err > tol, B_err > tol, pi_err > tol]):
            if verbose:
                print(
                    "Training step {}. A err: {:.5f}, B err: {:.5f} pi err: {:.5f}".format(
                        step, A_err, B_err, pi_err
                    )
                )

            # E-step
            gamma, xi, phi = self._Estep()

            # M-step
            A_, B_, pi_ = self.A.copy(), self.B.copy(), self.pi.copy()
            self.A, self.B, self.pi = self._Mstep(gamma, xi, phi)

            # compute error
            A_err, B_err = squared_error(A_, self.A), squared_error(B_, self.B)
            pi_err = squared_error(pi_, self.pi)
            step += 1

        return self.A, self.B, self.pi

    def _Estep(self):
        """
        Run a single E-step update for the Baum-Welch/Forward-Backward
        algorithm. This step estimates xi and gamma, the excepted state-state
        transition counts and the expected state-occupancy counts,
        respectively.

        xi[i,j,k] gives the probability of being in state i at time k and
        state j at time k+1 given the observed sequence O and the current
        estimates for transition (A) and emission (B) matrices:

            xi[i,j,k] = P(q_k=i,q_{k+1}=j|O,A,B,pi)
                      = P(q_k=i,q_{k+1}=j,O|A,B,pi) / P(O|A,B,pi)
                      = [
                            P(o_1,o_2,...,o_k,q_k=i|A,B,pi) *
                            P(q_{k+1}=j|q_k=i) * P(o_{k+1}|q_{k+1}=j) *
                            P(o_{k+2},o_{k+3},...,o_T|q_{k+1}=j,A,B,pi)
                        ] / P(O|A,B,pi)
                      = [
                            fwd[j, k] * self.A[j, i] *
                            self.B[i, o_{k+1}] * bwd[i, k + 1]
                        ] / fwd[:, T].sum()

        The expected number of transitions from state i to state j across the
        entire sequence is then the sum over all timesteps: xi[i,j,:].sum().

        gamma[i,j] gives the probability of being in state i at time j:

            gamma[i,j] = P(q_j=i|O,A,B,pi)

        Returns
        -------
        gamma : numpy array of shape (I, N, T)
            The estimated state-occupancy count matrix
        xi : numpy array of shape (I, N, N, T)
            The estimated state-state transition count matrix
        phi : numpy array of shape (I, N)
            The estimated prior counts for each latent state
        """
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
                            + np.log(self.A[si, sj])
                            + np.log(self.B[sj, ot1])
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
        gamma : numpy array of shape (I, N, T)
            The estimated state-occupancy count matrix
        xi : numpy array of shape (I, N, N, T)
            The estimated state-state transition count matrix
        phi : numpy array of shape (I, N)
            The estimated starting count matrix for each latent state

        Returns
        -------
        A : numpy array of shape (N, N)
            The estimated transition matrix
        B : numpy array of shape (N, V)
            The estimated emission matrix
        pi : numpy array of shape (N,)
            The estimated prior probabilities for each latent state
        """
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
                        count_gamma[i, si, vk] = -np.inf
                    else:
                        count_gamma[i, si, vk] = logsumexp(gamma[i, si, Obs == vk])

                for sj in range(self.N):
                    count_xi[i, si, sj] = logsumexp(xi[i, si, sj, :])

        pi = logsumexp(phi, axis=0) - np.log(self.I)
        np.testing.assert_almost_equal(np.exp(pi).sum(), 1)

        for si in range(self.N):
            for vk in range(self.V):
                B[si, vk] = logsumexp(count_gamma[:, si, vk]) - logsumexp(
                    count_gamma[:, si, :]
                )

            for sj in range(self.N):
                A[si, sj] = logsumexp(count_xi[:, si, sj]) - logsumexp(
                    count_xi[:, si, :]
                )

            np.testing.assert_almost_equal(np.exp(A[si, :]).sum(), 1)
            np.testing.assert_almost_equal(np.exp(B[si, :]).sum(), 1)
        return np.exp(A), np.exp(B), np.exp(pi)


#######################################################################
#                                Utils                                #
#######################################################################


def logsumexp(log_probs):
    """
    Redefine scipy.special.logsumexp
    see: http://bayesjumping.net/log-sum-exp-trick/
    """
    _max = np.max(log_probs)
    ds = log_probs - _max
    exp_sum = np.exp(ds).sum()
    return _max + np.log(exp_sum)
