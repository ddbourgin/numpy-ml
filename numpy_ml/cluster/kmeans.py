"""An implementation of kmeans clustering (hard, soft)"""

import numpy as np
from sklearn.preprocessing import normalize

class KMeans:
    def __init__(self, X_train, cluster_method="hard", n_clusters=5, beta = 1.0):
        r"""
        Kmean implementation.

        Parameters
        ----------
        X_train : array
                 The input matrix with dimensions (number of data x number of attributes).

        cluster_method : {'hard', 'soft'}
            Whether to cluster using hard or soft clustering. Default is hard.

        n_clusters : int
            number of clusters.

        beta: float
            span of a radial basis kernel
        """
        self.cluster_method = cluster_method
        self.X_train = X_train
        self.n_clusters = n_clusters

        self.centroids = None
        self.assignments = None
        
        if self.cluster_method == 'soft':
            self.beta = beta # Use beta only in soft clustering
            self.centroids, self.assignments = self._kmeans_soft()

        elif self.cluster_method == 'hard':
            self.centroids, self.assignments = self._kmeans_hard()


    def _is_converged(self, prev, cur):
        r"""
        Check for convergence by validating if the centroid or assignments have stopped changing across iterations.

        Parameters:
        -----------
        prev : array
            The input could be the centroids or assignment based on the chosen implemtation details (past iteration).
        cur : array
            The input could be the centroids or assignment based on the chosen implemtation details (current iteration).

        Returns
        -------
        True: if convergence is reached.
        False: if convergence is not reached.        
        """
        return np.allclose(prev,cur)


    def _kmeans_hard(self):
        r"""
        hard clustering: The "vanilla" kmean clustering assigns every data point to a single cluster.

        Returns
        -------
        centroids : array
                    The centroid matrix with dimensions (number of centroids x number of attributes).
        assignments : array
                    The assignments vector with dimensions (number of data).
        """
        size_of_data = self.X_train.shape[0]
        centroid_indexes = np.random.choice(size_of_data, self.n_clusters, replace=False)
        centroids = np.take(self.X_train, centroid_indexes, axis=0)
        assignments = [-1] * size_of_data
        n_dims = self.X_train.shape[1]
        iteration, max_iteration = 0, 100 
        prev_weight_vec = np.zeros(n_dims)
        weight_list = [prev_weight_vec, centroids] 
     
        while not self._is_converged(weight_list[-2], weight_list[-1]) or iteration < max_iteration:
            centroids = weight_list[-1]

            # update cluster assignments
            for i, x_val in enumerate(self.X_train):
                min_distance = 1000000000
                for k, mu_val in enumerate(centroids):

                    dist = np.linalg.norm(x_val - mu_val)

                    if dist < min_distance:                
                        min_distance = dist
                        assignments[i] = k

            # update centroids
            set_labels = range(self.n_clusters)
            # filter by labels
            for label in set_labels:
                filter_indices = np.where(np.array(assignments) == label)[0]
                count_per_label = len(filter_indices)
                if count_per_label > 0:
                    filter_xdata = np.take(self.X_train, filter_indices, axis=0)
                    centroids[label, :] = np.mean(filter_xdata, axis=0)
            weight_list.append(centroids)
            weight_list.pop(0)
            iteration = iteration + 1
        return centroids, assignments


    def _kmeans_soft(self):
        r"""
        Soft clustering: In this implementation, which is the modification of the kmean hard  clustering algorithm, we assigns every data point a degree of membership in the cluster assignment. Hence, we give a probability distribution.

        Parameters:
        -----------
        beta: float
            span of a radial basis kernel

        Returns
        -------
        centroids : array
                    The centroid matrix with dimensions (number of centroids x number of attributes).
        assignments : array
                    The assignments vector with dimensions (number of data x number of centroids). This is the probability distribution of the membership of each data point in the cluster.
        """
        size_of_data = self.X_train.shape[0]
        centroid_indexes = np.random.choice(size_of_data, self.n_clusters, replace=False)
        centroids = np.take(self.X_train, centroid_indexes, axis=0)
        assignments = -1 * np.ones((size_of_data, self.n_clusters))
        n_dims = self.X_train.shape[1]
        iteration, max_iteration = 0, 100 
        tol = 0.00001 # prevent division by zero
        prev_weight_vec = np.zeros(n_dims)
        weight_list = [prev_weight_vec, centroids]  

        while not self._is_converged(weight_list[-2], weight_list[-1]) or iteration < max_iteration:
            centroids = weight_list[-1]

            # update cluster assignments
            for i, x_val in enumerate(self.X_train):
                for k, mu_val in enumerate(centroids):
                    dist = np.linalg.norm(x_val - mu_val)
                    weight = np.exp(-dist / self.beta)
                    assignments[i][k] = weight
            # normalize assignment matrix
            assignments = normalize(assignments, axis=1, norm='l1')

            # update centroids
            set_labels = range(self.n_clusters)
            for label in set_labels:
                numerator = np.zeros((1, n_dims))
                denominator = 0
                for k, x_val in enumerate(self.X_train):
                    numerator += x_val * assignments[k][label] # weight contribution across data input
                    denominator += assignments[k][label]
                curr = (numerator + tol) / (denominator + tol)
                centroids[label, :] = curr

            weight_list.append(centroids)
            weight_list.pop(0)
            iteration = iteration + 1
        return centroids, assignments


    def get_centroids(self):
        r"""
        Get the centroids

        Returns
        -------
        centroids : array
            The centroid matrix with dimensions (number of centroids x number of attributes).
        """
        return self.centroids


    def get_assignments(self):
        r"""
        Get the assignment of data points into their clusters

        Returns
        -------
        assignments : array
                    The assignments vector with dimensions (number of data).
        """
        assignments = self.assignments
        if self.cluster_method == 'soft':
            assignments = np.argmax(self.assignments, axis=1)
        return assignments


    def get_proba(self):
        r"""
        Get the probability distribution of clusters assignments
        Note: only when doing soft-clustering

        Returns
        -------
        assignments : array (return None if hard clustering, array otherwise)
                    The assignments vector with dimensions (number of data x number of centroids). This is the probability distribution of the membership of each data point in the cluster.
        """
        assignments = None
        if self.cluster_method == 'soft':
            assignments = self.assignments
        return assignments
