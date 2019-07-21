import numpy as np

import scipy

from sklearn.neighbors import BallTree as sk_BallTree
from sklearn.metrics.pairwise import rbf_kernel as sk_rbf
from sklearn.metrics.pairwise import linear_kernel as sk_linear
from sklearn.metrics.pairwise import polynomial_kernel as sk_poly


from .distance_metrics import euclidean
from .kernels import LinearKernel, PolynomialKernel, RBFKernel
from .data_structures import BallTree

#######################################################################
#                               Kernels                               #
#######################################################################


def test_linear_kernel():
    while True:
        N = np.random.randint(1, 100)
        M = np.random.randint(1, 100)
        C = np.random.randint(1, 1000)

        X = np.random.rand(N, C)
        Y = np.random.rand(M, C)

        mine = LinearKernel()(X, Y)
        gold = sk_linear(X, Y)

        np.testing.assert_almost_equal(mine, gold)
        print("PASSED")


def test_polynomial_kernel():
    while True:
        N = np.random.randint(1, 100)
        M = np.random.randint(1, 100)
        C = np.random.randint(1, 1000)
        gamma = np.random.rand()
        d = np.random.randint(1, 5)
        c0 = np.random.rand()

        X = np.random.rand(N, C)
        Y = np.random.rand(M, C)

        mine = PolynomialKernel(gamma=gamma, d=d, c0=c0)(X, Y)
        gold = sk_poly(X, Y, gamma=gamma, degree=d, coef0=c0)

        np.testing.assert_almost_equal(mine, gold)
        print("PASSED")


def test_radial_basis_kernel():
    while True:
        N = np.random.randint(1, 100)
        M = np.random.randint(1, 100)
        C = np.random.randint(1, 1000)
        gamma = np.random.rand()

        X = np.random.rand(N, C)
        Y = np.random.rand(M, C)

        # sklearn (gamma) <-> mine (sigma) conversion:
        # gamma = 1 / (2 * sigma^2)
        # sigma = np.sqrt(1 / 2 * gamma)

        mine = RBFKernel(sigma=np.sqrt(1 / (2 * gamma)))(X, Y)
        gold = sk_rbf(X, Y, gamma=gamma)

        np.testing.assert_almost_equal(mine, gold)
        print("PASSED")


#######################################################################
#                          Distance Metrics                           #
#######################################################################


def test_euclidean():
    while True:
        N = np.random.randint(1, 100)
        x = np.random.rand(N)
        y = np.random.rand(N)
        mine = euclidean(x, y)
        theirs = scipy.spatial.distance.euclidean(x, y)
        np.testing.assert_almost_equal(mine, theirs)
        print("PASSED")


#######################################################################
#                           Data Structures                           #
#######################################################################


def test_ball_tree():
    while True:
        N = np.random.randint(2, 100)
        M = np.random.randint(2, 100)
        k = np.random.randint(1, N)
        ls = np.min([np.random.randint(1, 10), N - 1])

        X = np.random.rand(N, M)
        BT = BallTree(leaf_size=ls, metric=euclidean)
        BT.fit(X)

        x = np.random.rand(M)
        mine = BT.nearest_neighbors(k, x)
        assert len(mine) == k

        mine_neighb = np.array([n.key for n in mine])
        mine_dist = np.array([n.distance for n in mine])

        sort_ix = np.argsort(mine_dist)
        mine_dist = mine_dist[sort_ix]
        mine_neighb = mine_neighb[sort_ix]

        sk = sk_BallTree(X, leaf_size=ls)
        theirs_dist, ind = sk.query(x.reshape(1, -1), k=k)
        sort_ix = np.argsort(theirs_dist.flatten())

        theirs_dist = theirs_dist.flatten()[sort_ix]
        theirs_neighb = X[ind.flatten()[sort_ix]]

        for i in range(len(theirs_dist)):
            np.testing.assert_almost_equal(mine_neighb[i], theirs_neighb[i])
            np.testing.assert_almost_equal(mine_dist[i], theirs_dist[i])

        print("PASSED")
