import heapq
from copy import copy
from collections import Hashable

import numpy as np

from .distance_metrics import euclidean

#######################################################################
#                           Priority Queue                            #
#######################################################################


class PQNode(object):
    def __init__(self, key, val, priority, entry_id, **kwargs):
        """A generic node object for holding entries in :class:`PriorityQueue`"""
        self.key = key
        self.val = val
        self.entry_id = entry_id
        self.priority = priority

    def __repr__(self):
        fstr = "PQNode(key={}, val={}, priority={}, entry_id={})"
        return fstr.format(self.key, self.val, self.priority, self.entry_id)

    def to_dict(self):
        """Return a dictionary representation of the node's contents"""
        d = self.__dict__
        d["id"] = "PQNode"
        return d

    def __gt__(self, other):
        if not isinstance(other, PQNode):
            return -1
        if self.priority == other.priority:
            return self.entry_id > other.entry_id
        return self.priority > other.priority

    def __ge__(self, other):
        if not isinstance(other, PQNode):
            return -1
        return self.priority >= other.priority

    def __lt__(self, other):
        if not isinstance(other, PQNode):
            return -1
        if self.priority == other.priority:
            return self.entry_id < other.entry_id
        return self.priority < other.priority

    def __le__(self, other):
        if not isinstance(other, PQNode):
            return -1
        return self.priority <= other.priority


class PriorityQueue:
    def __init__(self, capacity, heap_order="max"):
        """
        A priority queue implementation using a binary heap.

        Notes
        -----
        A priority queue is a data structure useful for storing the top
        `capacity` largest or smallest elements in a collection of values. As a
        result of using a binary heap, ``PriorityQueue`` offers `O(log N)`
        :meth:`push` and :meth:`pop` operations.

        Parameters
        ----------
        capacity: int
            The maximum number of items that can be held in the queue.
        heap_order: {"max", "min"}
            Whether the priority queue should retain the items with the
            `capacity` smallest (`heap_order` = 'min') or `capacity` largest
            (`heap_order` = 'max') priorities.
        """
        assert heap_order in ["max", "min"], "heap_order must be either 'max' or 'min'"
        self.capacity = capacity
        self.heap_order = heap_order

        self._pq = []
        self._count = 0
        self._entry_counter = 0

    def __repr__(self):
        fstr = "PriorityQueue(capacity={}, heap_order={}) with {} items"
        return fstr.format(self.capacity, self.heap_order, self._count)

    def __len__(self):
        return self._count

    def __iter__(self):
        return iter(self._pq)

    def push(self, key, priority, val=None):
        """
        Add a new (key, value) pair with priority `priority` to the queue.

        Notes
        -----
        If the queue is at capacity and `priority` exceeds the priority of the
        item with the largest/smallest priority currently in the queue, replace
        the current queue item with (`key`, `val`).

        Parameters
        ----------
        key : hashable object
            The key to insert into the queue.
        priority : comparable
            The priority for the `key`, `val` pair.
        val : object
            The value associated with `key`. Default is None.
        """
        if self.heap_order == "max":
            priority = -1 * priority

        item = PQNode(key=key, val=val, priority=priority, entry_id=self._entry_counter)
        heapq.heappush(self._pq, item)

        self._count += 1
        self._entry_counter += 1

        while self._count > self.capacity:
            self.pop()

    def pop(self):
        """
        Remove the item with the largest/smallest (depending on
        ``self.heap_order``) priority from the queue and return it.

        Notes
        -----
        In contrast to :meth:`peek`, this operation is `O(log N)`.

        Returns
        -------
        item : :class:`PQNode` instance or None
            Item with the largest/smallest priority, depending on
            ``self.heap_order``.
        """
        item = heapq.heappop(self._pq).to_dict()
        if self.heap_order == "max":
            item["priority"] = -1 * item["priority"]
        self._count -= 1
        return item

    def peek(self):
        """
        Return the item with the largest/smallest (depending on
        ``self.heap_order``) priority *without* removing it from the queue.

        Notes
        -----
        In contrast to :meth:`pop`, this operation is O(1).

        Returns
        -------
        item : :class:`PQNode` instance or None
            Item with the largest/smallest priority, depending on
            ``self.heap_order``.
        """
        item = None
        if self._count > 0:
            item = copy(self._pq[0].to_dict())
            if self.heap_order == "max":
                item["priority"] = -1 * item["priority"]
        return item


#######################################################################
#                              Ball Tree                              #
#######################################################################


class BallTreeNode:
    def __init__(self, centroid=None, X=None, y=None):
        self.left = None
        self.right = None
        self.radius = None
        self.is_leaf = False

        self.data = X
        self.targets = y
        self.centroid = centroid

    def __repr__(self):
        fstr = "BallTreeNode(centroid={}, is_leaf={})"
        return fstr.format(self.centroid, self.is_leaf)

    def to_dict(self):
        d = self.__dict__
        d["id"] = "BallTreeNode"
        return d


class BallTree:
    def __init__(self, leaf_size=40, metric=None):
        """
        A ball tree data structure.

        Notes
        -----
        A ball tree is a binary tree in which every node defines a
        `D`-dimensional hypersphere ("ball") containing a subset of the points
        to be searched. Each internal node of the tree partitions the data
        points into two disjoint sets which are associated with different
        balls. While the balls themselves may intersect, each point is assigned
        to one or the other ball in the partition according to its distance
        from the ball's center. Each leaf node in the tree defines a ball and
        enumerates all data points inside that ball.

        Parameters
        ----------
        leaf_size : int
            The maximum number of datapoints at each leaf. Default is 40.
        metric : :doc:`Distance metric <numpy_ml.utils.distance_metrics>` or None
            The distance metric to use for computing nearest neighbors. If
            None, use the :func:`~numpy_ml.utils.distance_metrics.euclidean`
            metric. Default is None.

        References
        ----------
        .. [1] Omohundro, S. M. (1989). "Five balltree construction algorithms". *ICSI
           Technical Report TR-89-063*.
        .. [2] Liu, T., Moore, A., & Gray A. (2006). "New algorithms for efficient
           high-dimensional nonparametric classification". *J. Mach. Learn. Res.,
           7*, 1135-1158.
        """
        self.root = None
        self.leaf_size = leaf_size
        self.metric = metric if metric is not None else euclidean

    def fit(self, X, y=None):
        """
        Build a ball tree recursively using the O(M log N) `k`-d construction
        algorithm.

        Notes
        -----
        Recursively divides data into nodes defined by a centroid `C` and radius
        `r` such that each point below the node lies within the hyper-sphere
        defined by `C` and `r`.

        Parameters
        ----------
        X : :py:class:`ndarray <numpy.ndarray>` of shape `(N, M)`
            An array of `N` examples each with `M` features.
        y : :py:class:`ndarray <numpy.ndarray>` of shape `(N, \*)` or None
            An array of target values / labels associated with the entries in
            `X`. Default is None.
        """
        centroid, left_X, left_y, right_X, right_y = self._split(X, y)
        self.root = BallTreeNode(centroid=centroid)
        self.root.radius = np.max([self.metric(centroid, x) for x in X])
        self.root.left = self._build_tree(left_X, left_y)
        self.root.right = self._build_tree(right_X, right_y)

    def _build_tree(self, X, y):
        centroid, left_X, left_y, right_X, right_y = self._split(X, y)

        if X.shape[0] <= self.leaf_size:
            leaf = BallTreeNode(centroid=centroid, X=X, y=y)
            leaf.radius = np.max([self.metric(centroid, x) for x in X])
            leaf.is_leaf = True
            return leaf

        node = BallTreeNode(centroid=centroid)
        node.radius = np.max([self.metric(centroid, x) for x in X])
        node.left = self._build_tree(left_X, left_y)
        node.right = self._build_tree(right_X, right_y)
        return node

    def _split(self, X, y=None):
        # find the dimension with greatest variance
        split_dim = np.argmax(np.var(X, axis=0))

        # sort X and y along split_dim
        sort_ixs = np.argsort(X[:, split_dim])
        X, y = X[sort_ixs], y[sort_ixs] if y is not None else None

        # divide at median value of split_dim
        med_ix = X.shape[0] // 2
        centroid = X[med_ix]  # , split_dim

        # split data into two halves at the centroid (median always appears on
        # the right split)
        left_X, left_y = X[:med_ix], y[:med_ix] if y is not None else None
        right_X, right_y = X[med_ix:], y[med_ix:] if y is not None else None
        return centroid, left_X, left_y, right_X, right_y

    def nearest_neighbors(self, k, x):
        """
        Find the `k` nearest neighbors in the ball tree to a query vector `x`
        using the KNS1 algorithm.

        Parameters
        ----------
        k : int
            The number of closest points in `X` to return
        x : :py:class:`ndarray <numpy.ndarray>` of shape `(1, M)`
            The query vector.

        Returns
        -------
        nearest : list of :class:`PQNode` s of length `k`
            List of the `k` points in `X` to closest to the query vector. The
            ``key`` attribute of each :class:`PQNode` contains the point itself, the
            ``val`` attribute contains its target, and the ``distance``
            attribute contains its distance to the query vector.
        """
        # maintain a max-first priority queue with priority = distance to x
        PQ = PriorityQueue(capacity=k, heap_order="max")
        nearest = self._knn(k, x, PQ, self.root)
        for n in nearest:
            n.distance = self.metric(x, n.key)
        return nearest

    def _knn(self, k, x, PQ, root):
        dist = self.metric
        dist_to_ball = dist(x, root.centroid) - root.radius
        dist_to_farthest_neighbor = dist(x, PQ.peek()["key"]) if len(PQ) > 0 else np.inf

        if dist_to_ball >= dist_to_farthest_neighbor and len(PQ) == k:
            return PQ
        if root.is_leaf:
            targets = [None] * len(root.data) if root.targets is None else root.targets
            for point, target in zip(root.data, targets):
                dist_to_x = dist(x, point)
                if len(PQ) == k and dist_to_x < dist_to_farthest_neighbor:
                    PQ.push(key=point, val=target, priority=dist_to_x)
                else:
                    PQ.push(key=point, val=target, priority=dist_to_x)
        else:
            l_closest = dist(x, root.left.centroid) < dist(x, root.right.centroid)
            PQ = self._knn(k, x, PQ, root.left if l_closest else root.right)
            PQ = self._knn(k, x, PQ, root.right if l_closest else root.left)
        return PQ


#######################################################################
#                         Multinomial Sampler                         #
#######################################################################


class DiscreteSampler:
    def __init__(self, probs, log=False, with_replacement=True):
        """
        Sample from an arbitrary multinomial PMF over the first `N` nonnegative
        integers using Vose's algorithm for the alias method.

        Notes
        -----
        Vose's algorithm takes `O(n)` time to initialize, requires `O(n)` memory,
        and generates samples in constant time.

        References
        ----------
        .. [1] Walker, A. J. (1977) "An efficient method for generating discrete
           random variables with general distributions". *ACM Transactions on
           Mathematical Software, 3(3)*, 253-256.

        .. [2] Vose, M. D. (1991) "A linear algorithm for generating random numbers
           with a given distribution". *IEEE Trans. Softw. Eng., 9*, 972-974.

        .. [3] Schwarz, K (2011) "Darts, dice, and coins: sampling from a discrete
           distribution". http://www.keithschwarz.com/darts-dice-coins/

        Parameters
        ----------
        probs : :py:class:`ndarray <numpy.ndarray>` of length `(N,)`
            A list of probabilities of the `N` outcomes in the sample space.
            `probs[i]` returns the probability of outcome `i`.
        log : bool
            Whether the probabilities in `probs` are in logspace. Default is
            False.
        with_replacement : bool
            Whether to generate samples with or without replacement. Default is
            True.
        """
        if not isinstance(probs, np.ndarray):
            probs = np.array(probs)

        self.log = log
        self.N = len(probs)
        self.probs = probs
        self.with_replacement = with_replacement

        alias = np.zeros(self.N)
        prob = np.zeros(self.N)
        scaled_probs = self.probs + np.log(self.N) if log else self.probs * self.N

        selector = scaled_probs < 0 if log else scaled_probs < 1
        small, large = np.where(selector)[0].tolist(), np.where(~selector)[0].tolist()

        while len(small) and len(large):
            l, g = small.pop(), large.pop()

            alias[l] = g
            prob[l] = scaled_probs[l]

            if log:
                pg = np.log(np.exp(scaled_probs[g]) + np.exp(scaled_probs[l]) - 1)
            else:
                pg = scaled_probs[g] + scaled_probs[l] - 1

            scaled_probs[g] = pg
            to_small = pg < 0 if log else pg < 1
            if to_small:
                small.append(g)
            else:
                large.append(g)

        while len(large):
            prob[large.pop()] = 0 if log else 1

        while len(small):
            prob[small.pop()] = 0 if log else 1

        self.prob_table = prob
        self.alias_table = alias

    def __call__(self, n_samples=1):
        """
        Generate random draws from the `probs` distribution over integers in
        [0, N).

        Parameters
        ----------
        n_samples: int
            The number of samples to generate. Default is 1.

        Returns
        -------
        sample : :py:class:`ndarray <numpy.ndarray>` of shape `(n_samples,)`
            A collection of draws from the distribution defined by `probs`.
            Each sample is an int in the range `[0, N)`.
        """
        return self.sample(n_samples)

    def sample(self, n_samples=1):
        """
        Generate random draws from the `probs` distribution over integers in
        [0, N).

        Parameters
        ----------
        n_samples: int
            The number of samples to generate. Default is 1.

        Returns
        -------
        sample : :py:class:`ndarray <numpy.ndarray>` of shape `(n_samples,)`
            A collection of draws from the distribution defined by `probs`.
            Each sample is an int in the range `[0, N)`.
        """
        ixs = np.random.randint(0, self.N, n_samples)
        p = np.exp(self.prob_table[ixs]) if self.log else self.prob_table[ixs]
        flips = np.random.binomial(1, p)
        samples = [ix if f else self.alias_table[ix] for ix, f in zip(ixs, flips)]

        # do recursive rejection sampling to sample without replacement
        if not self.with_replacement:
            unique = list(set(samples))
            while len(samples) != len(unique):
                n_new = len(samples) - len(unique)
                samples = unique + self.sample(n_new).tolist()
                unique = list(set(samples))

        return np.array(samples, dtype=int)


#######################################################################
#                                Dict                                 #
#######################################################################


class Dict(dict):
    def __init__(self, encoder=None):
        """
        A dictionary subclass which returns the key value if it is not in the
        dict.

        Parameters
        ----------
        encoder : function or None
            A function which is applied to a key before adding / retrieving it
            from the dictionary. If None, the function defaults to the
            identity. Default is None.
        """
        super(Dict, self).__init__()
        self._encoder = encoder
        self._id_max = 0

    def __setitem__(self, key, value):
        if self._encoder is not None:
            key = self._encoder(key)
        elif not isinstance(key, Hashable):
            key = tuple(key)
        super(Dict, self).__setitem__(key, value)

    def _encode_key(self, key):
        D = super(Dict, self)
        enc_key = self._encoder(key)
        if D.__contains__(enc_key):
            val = D.__getitem__(enc_key)
        else:
            val = self._id_max
            D.__setitem__(enc_key, val)
            self._id_max += 1
        return val

    def __getitem__(self, key):
        self._key = copy.deepcopy(key)
        if self._encoder is not None:
            return self._encode_key(key)
        elif not isinstance(key, Hashable):
            key = tuple(key)
        return super(Dict, self).__getitem__(key)

    def __missing__(self, key):
        return self._key
