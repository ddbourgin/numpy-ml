import numpy as np
from numpy_ml.linear_models.lm import LinearRegression
from sklearn import datasets
from sklearn.datasets.samples_generator import make_blobs
from sklearn.linear_model import LinearRegression as OriginalLinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

def test_linear_model(N=1):
    seed = 12345
    np.random.seed(seed)
    n_clusters=4
    # loading the dataset
    orig_num_of_samples, orig_num_of_features = 3000, 300
    X, y = make_blobs(n_samples=orig_num_of_samples, centers=n_clusters, n_features = orig_num_of_features, cluster_std=0.50, random_state=seed)
    X_train, X_update, y_train, y_update = train_test_split(X, y, test_size=0.2, random_state=seed)

    # Ground truth
    orig_lreg_model = OriginalLinearRegression()
    orig_lreg_model.fit(X, y)

    # Our model
    olr = LinearRegression()
    olr.fit(X_train, y_train)

    ## update our model
    for x_new, y_new in zip(X_update, y_update):
        x_new = x_new.reshape((1, orig_num_of_features))
        y_new = y_new
        olr.update(x_new, y_new)

    # Evaluating
    X_test, y_test = make_blobs(n_samples=orig_num_of_samples, centers=n_clusters, n_features = orig_num_of_features, cluster_std=0.50, random_state=seed+2)

    y_pred_orig = orig_lreg_model.predict (X_test)
    y_pred_online = olr.predict (X_test)

    r2score_orig = r2_score(y_test, y_pred_orig) 
    r2score_online = r2_score(y_test, y_pred_online) 
    print ("Both score must be similar between scikit: {} and our own online implementation: {}".format(r2score_orig, r2score_online))

