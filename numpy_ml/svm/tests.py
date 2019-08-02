import warnings
warnings.filterwarnings('ignore')
import numpy as np
import random

from svm import SVM

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.datasets.samples_generator import make_blobs
from sklearn.model_selection import train_test_split


def test_SVM():
    i = 1
    np.random.seed(12345)
    while True:
        X, Y = make_blobs(  # generate dataset
            n_samples=np.random.randint(2, 100), 
            n_features=np.random.randint(2, 100),
            centers=2, random_state=i, 
        )
        X, X_test, Y, Y_test = train_test_split(X, Y, test_size=0.3, random_state=i)
        if 0 not in Y or 1 not in Y:  # ignore split error(train/test data only 1 class)
            continue
        # generate param
        C = random.uniform(0.1, 0.9)
        max_iter = random.uniform(50, 500)
        kernel = np.random.choice(["linear", "rbf"])
        tol = random.uniform(0.000001, 0.1)
        # fit and predict
        clf1 = SVC(C=C, max_iter=max_iter, kernel=kernel, tol=tol)
        clf1.fit(X, Y)
        pred1 = clf1.predict(X_test)
        clf2 = SVM(C=C, max_iter=max_iter, kernel=kernel, tol=tol)
        clf2.fit(X, Y)
        pred2 = clf2.predict(X_test)
        # judge
        err_msg = "ERROR {0} {1}".format(accuracy_score(Y_test, pred1), accuracy_score(Y_test, pred2))
        assert accuracy_score(Y_test, pred1) == accuracy_score(Y_test, pred2), err_msg
        print("PASSED")
