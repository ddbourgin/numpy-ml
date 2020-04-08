from abc import ABC, abstractmethod
from collections import defaultdict
from itertools import combinations, permutations

import numpy as np

#######################################################################
#                          Graph Components                           #
#######################################################################


class Edge(object):
    def __init__(self, fr, to, w=None):
        """
        A generic directed edge object.

        Parameters
        ----------
        fr: int
            The id of the vertex the edge goes from
        to: int
            The id of the vertex the edge goes to
        w: float, :class:`Object` instance, or None
            The edge weight, if applicable. If weight is an arbitrary Object it
            must have a method called 'sample' which takes no arguments and
            returns a random sample from the weight distribution. If `w` is
            None, no weight is assumed. Default is None.
        """
        self.fr = fr
        self.to = to
        self._w = w

    def __repr__(self):
        return "{} -> {}, weight: {}".format(self.fr, self.to, self._w)

    @property
    def weight(self):
        return self._w.sample() if hasattr(self._w, "sample") else self._w

    def reverse(self):
        """Reverse the edge direction"""
        return Edge(self.t, self.f, self.w)


#######################################################################
#                             Graph Types                             #
#######################################################################


class Graph(ABC):
    def __init__(self, V, E):
        self._I2V = {i: v for i, v in zip(range(len(V)), V)}
        self._V2I = {v: i for i, v in zip(range(len(V)), V)}
        self._G = {i: set() for i in range(len(V))}
        self._V = V
        self._E = E

        self._build_adjacency_list()

    def __getitem__(self, v_i):
        return self.get_neighbors(v_i)

    def get_index(self, v):
        """Get the internal index for a given vetex"""
        return self._V2I[v]

    def get_vertex(self, v_i):
        """Get the original vertex from a given internal index"""
        return self._I2V[v_i]

    @property
    def vertices(self):
        return self._V

    @property
    def indices(self):
        return list(range(len(self.vertices)))

    @property
    def edges(self):
        return self._E

    def get_neighbors(self, v_i):
        """
        Return the internal indices of the vertices reachable from the vertex
        with index `v_i`.
        """
        return [self._V2I[e.to] for e in self._G[v_i]]

    def to_matrix(self):
        """Return an adjacency matrix representation of the graph"""
        adj_mat = np.zeros((len(self._V), len(self._V)))
        for e in self.edges:
            fr, to = self._V2I[e.fr], self._V2I[e.to]
            adj_mat[fr, to] = 1 if e.weight is None else e.weight
        return adj_mat

    def to_adj_dict(self):
        """Return an adjacency dictionary representation of the graph"""
        adj_dict = defaultdict(lambda: list())
        for e in self.edges:
            adj_dict[e.fr].append(e)
        return adj_dict

    def path_exists(self, s_i, e_i):
        """
        Check whether a path exists from vertex index `s_i` to `e_i`.

        Parameters
        ----------
        s_i: Int
            The interal index of the start vertex
        e_i: Int
            The internal index of the end vertex

        Returns
        -------
        path_exists : Boolean
            Whether or not a valid path exists between `s_i` and `e_i`.
        """
        queue = [(s_i, [s_i])]
        while len(queue):
            c_i, path = queue.pop(0)
            nbrs_not_on_path = set(self.get_neighbors(c_i)) - set(path)

            for n_i in nbrs_not_on_path:
                queue.append((n_i, path + [n_i]))
                if n_i == e_i:
                    return True
        return False

    def all_paths(self, s_i, e_i):
        """
        Find all simple paths between `s_i` and `e_i` in the graph.

        Notes
        -----
        Uses breadth-first search. Ignores all paths with repeated vertices.

        Parameters
        ----------
        s_i: Int
            The interal index of the start vertex
        e_i: Int
            The internal index of the end vertex

        Returns
        -------
        complete_paths : list of lists
            A list of all paths from `s_i` to `e_i`. Each path is represented
            as a list of interal vertex indices.
        """
        complete_paths = []
        queue = [(s_i, [s_i])]

        while len(queue):
            c_i, path = queue.pop(0)
            nbrs_not_on_path = set(self.get_neighbors(c_i)) - set(path)

            for n_i in nbrs_not_on_path:
                if n_i == e_i:
                    complete_paths.append(path + [n_i])
                else:
                    queue.append((n_i, path + [n_i]))

        return complete_paths

    @abstractmethod
    def _build_adjacency_list(self):
        pass


class DiGraph(Graph):
    def __init__(self, V, E):
        """
        A generic directed graph object.

        Parameters
        ----------
        V : list
            A list of vertex IDs.
        E : list of :class:`Edge <numpy_ml.utils.graphs.Edge>` objects
            A list of directed edges connecting pairs of vertices in ``V``.
        """
        super().__init__(V, E)
        self.is_directed = True
        self._topological_ordering = []

    def _build_adjacency_list(self):
        """Encode directed graph as an adjancency list"""
        # assumes no parallel edges
        for e in self.edges:
            fr_i = self._V2I[e.fr]
            self._G[fr_i].add(e)

    def reverse(self):
        """Reverse the direction of all edges in the graph"""
        return DiGraph(self.vertices, [e.reverse() for e in self.edges])

    def topological_ordering(self):
        """
        Returns a (non-unique) topological sort / linearization of the nodes
        IFF the graph is acyclic, otherwise returns None.

        Notes
        -----
        A topological sort is an ordering on the nodes in `G` such that for every
        directed edge :math:`u \\rightarrow v` in the graph, `u` appears before
        `v` in the ordering.  The topological ordering is produced by ordering
        the nodes in `G` by their DFS "last visit time," from greatest to
        smallest.

        This implementation follows a recursive, DFS-based approach [1]_ which
        may break if the graph is very large. For an iterative version, see
        Khan's algorithm [2]_ .

        References
        ----------
        .. [1] Tarjan, R. (1976), Edge-disjoint spanning trees and depth-first
           search, *Acta Informatica, 6 (2)*: 171–185.
        .. [2] Kahn, A. (1962), Topological sorting of large networks,
           *Communications of the ACM, 5 (11)*: 558–562.

        Returns
        -------
        ordering : list or None
            A topoligical ordering of the vertex indices if the graph is a DAG,
            otherwise None.
        """
        ordering = []
        visited = set()

        def dfs(v_i, path=None):
            """A simple DFS helper routine"""
            path = set([v_i]) if path is None else path
            for nbr_i in self.get_neighbors(v_i):
                if nbr_i in path:
                    return True  # cycle detected!
                elif nbr_i not in visited:
                    visited.add(nbr_i)
                    path.add(nbr_i)
                    is_cyclic = dfs(nbr_i, path)
                    if is_cyclic:
                        return True

            # insert to the beginning of the ordering
            ordering.insert(0, v_i)
            path -= set([v_i])
            return False

        for s_i in self.indices:
            if s_i not in visited:
                visited.add(s_i)
                is_cyclic = dfs(s_i)

                if is_cyclic:
                    return None

        return ordering

    def is_acyclic(self):
        """Check whether the graph contains cycles"""
        return self.topological_ordering() is not None


class UndirectedGraph(Graph):
    def __init__(self, V, E):
        """
        A generic undirected graph object.

        Parameters
        ----------
        V : list
            A list of vertex IDs.
        E : list of :class:`Edge <numpy_ml.utils.graphs.Edge>` objects
            A list of edges connecting pairs of vertices in ``V``. For any edge
            connecting vertex `u` to vertex `v`, :class:`UndirectedGraph
            <numpy_ml.utils.graphs.UndirectedGraph>` will assume that there
            exists a corresponding edge connecting `v` to `u`, even if this is
            not present in `E`.
        """
        super().__init__(V, E)
        self.is_directed = False

    def _build_adjacency_list(self):
        """Encode undirected, unweighted graph as an adjancency list"""
        # assumes no parallel edges
        # each edge appears twice as (u,v) and (v,u)
        for e in self.edges:
            fr_i = self._V2I[e.fr]
            to_i = self._V2I[e.to]

            self._G[fr_i].add(e)
            self._G[to_i].add(e.reverse())


#######################################################################
#                          Graph Generators                           #
#######################################################################


def random_unweighted_graph(n_vertices, edge_prob=0.5, directed=False):
    """
    Generate an unweighted Erdős-Rényi random graph [*]_.

    References
    ----------
    .. [*] Erdős, P. and Rényi, A. (1959). On Random Graphs, *Publ. Math. 6*, 290.

    Parameters
    ----------
    n_vertices : int
        The number of vertices in the graph.
    edge_prob : float in [0, 1]
        The probability of forming an edge between two vertices. Default is
        0.5.
    directed : bool
        Whether the edges in the graph should be directed. Default is False.

    Returns
    -------
    G : :class:`Graph` instance
        The resulting random graph.
    """
    vertices = list(range(n_vertices))
    candidates = permutations(vertices, 2) if directed else combinations(vertices, 2)

    edges = []
    for (fr, to) in candidates:
        if np.random.rand() <= edge_prob:
            edges.append(Edge(fr, to))

    return DiGraph(vertices, edges) if directed else UndirectedGraph(vertices, edges)


def random_DAG(n_vertices, edge_prob=0.5):
    """
    Create a 'random' unweighted directed acyclic graph by pruning all the
    backward connections from a random graph.

    Parameters
    ----------
    n_vertices : int
        The number of vertices in the graph.
    edge_prob : float in [0, 1]
        The probability of forming an edge between two vertices in the
        underlying random graph, before edge pruning. Default is 0.5.

    Returns
    -------
    G : :class:`Graph` instance
        The resulting DAG.
    """
    G = random_unweighted_graph(n_vertices, edge_prob, directed=True)

    # prune edges to remove backwards connections between vertices
    G = DiGraph(G.vertices, [e for e in G.edges if e.fr < e.to])

    # if we pruned away all the edges, generate a new graph
    while not len(G.edges):
        G = random_unweighted_graph(n_vertices, edge_prob, directed=True)
        G = DiGraph(G.vertices, [e for e in G.edges if e.fr < e.to])
    return G
