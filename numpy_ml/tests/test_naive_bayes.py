import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

from sklearn import naive_bayes

from numpy_ml.naive_bayes.naive_bayes import GaussianNB

def test_GaussianNB():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

    NB = GaussianNB()
    NB.fit(X_train, y_train)

    probs=NB.predict(X_test)
    pred = np.argmax(probs, 1)
    accuracy = sum(pred==y_test)/X_test.shape[0]

    sklearn_NB = naive_bayes.GaussianNB()
    sklearn_NB.fit(X_train, y_train)

    sk_pred=sklearn_NB.predict(X_test)
    sk_accuracy = sum(sk_pred==y_test)/X_test.shape[0]


    try:
        np.testing.assert_almost_equal(accuracy, sk_accuracy)
        print("\Accuracies are equal")
    except AssertionError as e:
        print("\Accuracies are not  equal:\n{}".format(e))



