# flake8: noqa
import numpy as np
from sklearn import datasets
from sklearn.cluster import KMeans as origKMeans
from sklearn.datasets.samples_generator import make_blobs
from sklearn.metrics.cluster import fowlkes_mallows_score
from sklearn.metrics import davies_bouldin_score
from numpy_ml.cluster.kmeans import KMeans

def test_kmeans():
    seed = 12345
    np.random.seed(seed)
    n_clusters=4
    # loading the dataset
    orig_num_of_samples, orig_num_of_features = 3000, 300
    X, y_true = make_blobs(n_samples=orig_num_of_samples, centers=n_clusters, n_features = orig_num_of_features,
                           cluster_std=0.50, random_state=seed)

    # K-Means scikit version (Hard clustering)
    kmeans =  origKMeans(n_clusters=n_clusters, random_state=seed).fit(X)
    # cluster labels as gold standard
    gold_labels = kmeans.labels_

    # Test the dimensions of the parameters
    print ("Hard Clustering")
    km_hard = KMeans(X, cluster_method="hard", n_clusters=n_clusters)

    num_of_samples, num_of_features = km_hard.get_centroids().shape
    assert (num_of_samples, num_of_features) in [(orig_num_of_features, n_clusters), (n_clusters, orig_num_of_features)], "mismatch in assignment probability"

    num_of_samples = len(km_hard.get_assignments())
    assert (num_of_samples == orig_num_of_samples), "mismatch in assignment size"

    # Comparing our clustering algorithm to the Gold standard
    mallows_score = fowlkes_mallows_score(gold_labels, km_hard.get_assignments())
    print("Mallow score: {} | values closer to 1 indicates better clustering".format(mallows_score))

    print ("Soft Clustering")
    # Use the default value for the beta variable
    km_soft = KMeans(X, cluster_method="soft", n_clusters=n_clusters)

    # Only call during soft clustering
    num_of_samples, num_of_features = km_soft.get_proba().shape
    assert (num_of_samples, num_of_features) in [(orig_num_of_samples, n_clusters), (n_clusters, orig_num_of_samples)], "mismatch in assignment probability"

    num_of_samples, num_of_features = km_soft.get_centroids().shape
    assert (num_of_samples, num_of_features) in [(orig_num_of_features, n_clusters), (n_clusters, orig_num_of_features)], "mismatch in assignment probability"

    num_of_samples = len(km_soft.get_assignments())
    assert (num_of_samples == orig_num_of_samples), "mismatch in assignment size"

    # Comparing our clustering algorithm to the Gold standard
    mallows_score = fowlkes_mallows_score(gold_labels, km_soft.get_assignments())
    print("Mallow score: {} | values closer to 1 indicates better clustering".format(mallows_score))