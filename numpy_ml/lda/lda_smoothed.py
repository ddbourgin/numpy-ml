import numpy as np


class SmoothedLDA(object):
    def __init__(self, T, **kwargs):
        """
        A smoothed LDA model trained using collapsed Gibbs sampling. Generates
        posterior mean estimates for model parameters `phi` and `theta`.

        Parameters
        ----------
        T : int
            Number of topics

        Attributes
        ----------
        D : int
            Number of documents
        N : int
            Total number of words across all documents
        V : int
            Number of unique word tokens across all documents
        phi : :py:class:`ndarray <numpy.ndarray>` of shape `(N[d], T)`
            The word-topic distribution
        theta : :py:class:`ndarray <numpy.ndarray>` of shape `(D, T)`
            The document-topic distribution
        alpha : :py:class:`ndarray <numpy.ndarray>` of shape `(1, T)`
            Parameter for the Dirichlet prior on the document-topic distribution
        beta  : :py:class:`ndarray <numpy.ndarray>` of shape `(V, T)`
            Parameter for the Dirichlet prior on the topic-word distribution
        """
        self.T = T

        self.alpha = (50.0 / self.T) * np.ones(self.T)
        if "alpha" in kwargs.keys():
            self.alpha = (kwargs["alpha"]) * np.ones(self.T)

        self.beta = 0.01
        if "beta" in kwargs.keys():
            self.beta = kwargs["beta"]

    def _init_params(self, texts, tokens):
        self.tokens = tokens
        self.D = len(texts)
        self.V = len(np.unique(self.tokens))
        self.N = np.sum(np.array([len(doc) for doc in texts]))
        self.word_document = np.zeros(self.N)

        # now that we know the number of tokens in our corpus, we can set beta
        self.beta = self.beta * np.ones(self.V)

        count = 0
        for doc_idx, doc in enumerate(texts):
            for word_idx, word in enumerate(doc):
                word_idx = word_idx + count
                self.word_document[word_idx] = doc_idx
            count = count + len(doc)

    def train(self, texts, tokens, n_gibbs=2000):
        """
        Trains a topic model on the documents in texts.

        Parameters
        ----------
        texts : array of length `(D,)`
            The training corpus represented as an array of subarrays, where
            each subarray corresponds to the tokenized words of a single
            document.
        tokens : array of length `(V,)`
            The set of unique tokens in the documents in `texts`.
        n_gibbs : int
            The number of steps to run the collapsed Gibbs sampler during
            training. Default is 2000.

        Returns
        -------
        C_wt : :py:class:`ndarray <numpy.ndarray>` of shape (V, T)
            The word-topic count matrix
        C_dt : :py:class:`ndarray <numpy.ndarray>` of shape (D, T)
            The document-topic count matrix
        assignments : :py:class:`ndarray <numpy.ndarray>` of shape (N, n_gibbs)
            The topic assignments for each word in the corpus on each Gibbs
            step.
        """
        self._init_params(texts, tokens)
        C_wt, C_dt, assignments = self._gibbs_sampler(n_gibbs, texts)
        self.fit_params(C_wt, C_dt)
        return C_wt, C_dt, assignments

    def what_did_you_learn(self, top_n=10):
        """
        Print the `top_n` most probable words under each topic
        """
        for tt in range(self.T):
            top_idx = np.argsort(self.phi[:, tt])[::-1][:top_n]
            top_tokens = self.tokens[top_idx]
            print("\nTop Words for Topic %s:\n" % (str(tt)))
            for token in top_tokens:
                print("\t%s\n" % (str(token)))

    def fit_params(self, C_wt, C_dt):
        """
        Estimate `phi`, the word-topic distribution, and `theta`, the
        topic-document distribution.

        Parameters
        ----------
        C_wt : :py:class:`ndarray <numpy.ndarray>` of shape (V, T)
            The word-topic count matrix
        C_dt : :py:class:`ndarray <numpy.ndarray>` of shape (D, T)
            The document-topic count matrix

        Returns
        -------
        phi : :py:class:`ndarray <numpy.ndarray>` of shape `(V, T)`
            The word-topic distribution
        theta : :py:class:`ndarray <numpy.ndarray>` of shape `(D, T)`
            The document-topic distribution
        """
        self.phi = np.zeros([self.V, self.T])
        self.theta = np.zeros([self.D, self.T])

        b, a = self.beta[0], self.alpha[0]
        for ii in range(self.V):
            for jj in range(self.T):
                self.phi[ii, jj] = (C_wt[ii, jj] + b) / (
                    np.sum(C_wt[:, jj]) + self.V * b
                )

        for dd in range(self.D):
            for jj in range(self.T):
                self.theta[dd, jj] = (C_dt[dd, jj] + a) / (
                    np.sum(C_dt[dd, :]) + self.T * a
                )
        return self.phi, self.theta

    def _estimate_topic_prob(self, ii, d, C_wt, C_dt):
        """
        Compute an approximation of the conditional probability that token ii
        is assigned to topic jj given all previous topic assignments and the
        current document d: p(t_i = j | t_{-i}, w_i, d_i)
        """
        p_vec = np.zeros(self.T)
        b, a = self.beta[0], self.alpha[0]
        for jj in range(self.T):
            # prob of word ii under topic jj
            frac1 = (C_wt[ii, jj] + b) / (np.sum(C_wt[:, jj]) + self.V * b)
            # prob of topic jj under document d
            frac2 = (C_dt[d, jj] + a) / (np.sum(C_dt[d, :]) + self.T * a)
            p_vec[jj] = frac1 * frac2
        return p_vec / np.sum(p_vec)

    def _gibbs_sampler(self, n_gibbs, texts):
        """
        Collapsed Gibbs sampler for estimating the posterior distribution over
        topic assignments.
        """
        # Initialize count matrices
        C_wt = np.zeros([self.V, self.T])
        C_dt = np.zeros([self.D, self.T])
        assignments = np.zeros([self.N, n_gibbs + 1])

        # Randomly initialize topic assignments for words
        for ii in range(self.N):
            token_idx = np.concatenate(texts)[ii]
            assignments[ii, 0] = np.random.randint(0, self.T)

            doc = self.word_document[ii]
            C_dt[doc, assignments[ii, 0]] += 1
            C_wt[token_idx, assignments[ii, 0]] += 1

        # run collapsed Gibbs sampler
        for gg in range(n_gibbs):
            print("Gibbs iteration {} of {}".format(gg + 1, n_gibbs))
            for jj in range(self.N):
                token_idx = np.concatenate(texts)[jj]

                # Decrement count matrices by 1
                doc = self.word_document[jj]
                C_wt[token_idx, assignments[jj, gg]] -= 1
                C_dt[doc, assignments[jj, gg]] -= 1

                # Draw new topic from our approximation of the conditional dist.
                p_topics = self._estimate_topic_prob(token_idx, doc, C_wt, C_dt)
                sampled_topic = np.nonzero(np.random.multinomial(1, p_topics))[0][0]

                # Update count matrices
                C_wt[token_idx, sampled_topic] += 1
                C_dt[doc, sampled_topic] += 1
                assignments[jj, gg + 1] = sampled_topic
        return C_wt, C_dt, assignments
