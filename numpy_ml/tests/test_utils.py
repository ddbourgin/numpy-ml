# flake8: noqa
import numpy as np

import scipy
import networkx as nx

from sklearn.neighbors import BallTree as sk_BallTree
from sklearn.metrics.pairwise import rbf_kernel as sk_rbf
from sklearn.metrics.pairwise import linear_kernel as sk_linear
from sklearn.metrics.pairwise import polynomial_kernel as sk_poly


from numpy_ml.utils.distance_metrics import euclidean
from numpy_ml.utils.kernels import LinearKernel, PolynomialKernel, RBFKernel
from numpy_ml.utils.data_structures import BallTree
from numpy_ml.utils.graphs import (
    DiGraph,
    UndirectedGraph,
    Edge,
    random_unweighted_graph,
    random_DAG,
)

#######################################################################
#                               Kernels                               #
#######################################################################


def test_linear_kernel(N=1):
    np.random.seed(12345)
    i = 0
    while i < N:
        N = np.random.randint(1, 100)
        M = np.random.randint(1, 100)
        C = np.random.randint(1, 1000)

        X = np.random.rand(N, C)
        Y = np.random.rand(M, C)

        mine = LinearKernel()(X, Y)
        gold = sk_linear(X, Y)

        np.testing.assert_almost_equal(mine, gold)
        print("PASSED")
        i += 1


def test_polynomial_kernel(N=1):
    np.random.seed(12345)
    i = 0
    while i < N:
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
        i += 1


def test_radial_basis_kernel(N=1):
    np.random.seed(12345)
    i = 0
    while i < N:
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
        i += 1


#######################################################################
#                          Distance Metrics                           #
#######################################################################


def test_euclidean(N=1):
    np.random.seed(12345)
    i = 0
    while i < N:
        N = np.random.randint(1, 100)
        x = np.random.rand(N)
        y = np.random.rand(N)
        mine = euclidean(x, y)
        theirs = scipy.spatial.distance.euclidean(x, y)
        np.testing.assert_almost_equal(mine, theirs)
        print("PASSED")
        i += 1


#######################################################################
#                           Data Structures                           #
#######################################################################


def test_ball_tree(N=1):
    np.random.seed(12345)
    i = 0
    while i < N:
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

        for j in range(len(theirs_dist)):
            np.testing.assert_almost_equal(mine_neighb[j], theirs_neighb[j])
            np.testing.assert_almost_equal(mine_dist[j], theirs_dist[j])

        print("PASSED")
        i += 1


#######################################################################
#                               Graphs                                #
#######################################################################


def from_networkx(G_nx):
    """Convert a networkx graph to my graph representation"""
    V = list(G_nx.nodes)
    edges = list(G_nx.edges)
    is_weighted = "weight" in G_nx[edges[0][0]][edges[0][1]]

    E = []
    for e in edges:
        if is_weighted:
            E.append(Edge(e[0], e[1], G_nx[e[0]][e[1]]["weight"]))
        else:
            E.append(Edge(e[0], e[1]))

    return DiGraph(V, E) if nx.is_directed(G_nx) else UndirectedGraph(V, E)


def to_networkx(G):
    """Convert my graph representation to a networkx graph"""
    G_nx = nx.DiGraph() if G.is_directed else nx.Graph()
    V = list(G._V2I.keys())
    G_nx.add_nodes_from(V)

    for v in V:
        fr_i = G._V2I[v]
        edges = G._G[fr_i]

        for edge in edges:
            G_nx.add_edge(edge.fr, edge.to, weight=edge._w)
    return G_nx


def test_all_paths(N=1):
    np.random.seed(12345)
    i = 0
    while i < N:
        p = np.random.rand()
        directed = np.random.rand() < 0.5
        G = random_unweighted_graph(n_vertices=5, edge_prob=p, directed=directed)

        nodes = G._I2V.keys()
        G_nx = to_networkx(G)

        # for each graph, test all_paths for all pairs of start and end
        # vertices. note that graph is not guaranteed to be connected, so many
        # paths will be empty
        for s_i in nodes:
            for e_i in nodes:
                if s_i == e_i:
                    continue

                paths = G.all_paths(s_i, e_i)
                paths_nx = nx.all_simple_paths(G_nx, source=s_i, target=e_i, cutoff=10)

                paths = sorted(paths)
                paths_nx = sorted(list(paths_nx))

                for p1, p2 in zip(paths, paths_nx):
                    np.testing.assert_array_equal(p1, p2)

                print("PASSED")
                i += 1


def test_random_DAG(N=1):
    np.random.seed(12345)
    i = 0
    while i < N:
        p = np.random.uniform(0.25, 1)
        n_v = np.random.randint(5, 50)

        G = random_DAG(n_v, p)
        G_nx = to_networkx(G)

        assert nx.is_directed_acyclic_graph(G_nx)
        print("PASSED")
        i += 1


def test_topological_ordering(N=1):
    np.random.seed(12345)
    i = 0
    while i < N:
        p = np.random.uniform(0.25, 1)
        n_v = np.random.randint(5, 10)

        G = random_DAG(n_v, p)
        G_nx = to_networkx(G)

        if nx.is_directed_acyclic_graph(G_nx):
            topo_order = G.topological_ordering()

            #  test topological order
            seen_it = set()
            for n_i in topo_order:
                seen_it.add(n_i)
                assert any([c_i in seen_it for c_i in G.get_neighbors(n_i)]) == False

            print("PASSED")
            i += 1


def test_is_acyclic(N=1):
    np.random.seed(12345)
    i = 0
    while i < N:
        p = np.random.rand()
        directed = np.random.rand() < 0.5
        G = random_unweighted_graph(n_vertices=10, edge_prob=p, directed=True)
        G_nx = to_networkx(G)

        assert G.is_acyclic() == nx.is_directed_acyclic_graph(G_nx)
        print("PASSED")
        i += 1
