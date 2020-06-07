# flake8: noqa
import numpy as np

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.datasets import make_regression, make_blobs
from sklearn.model_selection import train_test_split

from numpy_ml.trees.gbdt import GradientBoostedDecisionTree
from numpy_ml.trees.dt import DecisionTree, Node, Leaf
from numpy_ml.trees.rf import RandomForest
from numpy_ml.utils.testing import random_tensor


def clone_tree(dtree):
    children_left = dtree.tree_.children_left
    children_right = dtree.tree_.children_right
    feature = dtree.tree_.feature
    threshold = dtree.tree_.threshold
    values = dtree.tree_.value

    def grow(node_id):
        l, r = children_left[node_id], children_right[node_id]
        if l == r:
            return Leaf(values[node_id].argmax())
        n = Node(None, None, (feature[node_id], threshold[node_id]))
        n.left = grow(l)
        n.right = grow(r)
        return n

    node_id = 0
    root = Node(None, None, (feature[node_id], threshold[node_id]))
    root.left = grow(children_left[node_id])
    root.right = grow(children_right[node_id])
    return root


def compare_trees(mine, gold):
    clone = clone_tree(gold)
    mine = mine.root

    def test(mine, clone):
        if isinstance(clone, Node) and isinstance(mine, Node):
            assert mine.feature == clone.feature, "Node {} not equal".format(depth)
            np.testing.assert_allclose(mine.threshold, clone.threshold)
            test(mine.left, clone.left, depth + 1)
            test(mine.right, clone.right, depth + 1)
        elif isinstance(clone, Leaf) and isinstance(mine, Leaf):
            np.testing.assert_allclose(mine.value, clone.value)
            return
        else:
            raise ValueError("Nodes at depth {} are not equal".format(depth))

    depth = 0
    ok = True
    while ok:
        if isinstance(clone, Node) and isinstance(mine, Node):
            assert mine.feature == clone.feature
            np.testing.assert_allclose(mine.threshold, clone.threshold)
            test(mine.left, clone.left, depth + 1)
            test(mine.right, clone.right, depth + 1)
        elif isinstance(clone, Leaf) and isinstance(mine, Leaf):
            np.testing.assert_allclose(mine.value, clone.value)
            return
        else:
            raise ValueError("Nodes at depth {} are not equal".format(depth))


def test_DecisionTree(N=1):
    i = 1
    np.random.seed(12345)
    while i <= N:
        n_ex = np.random.randint(2, 100)
        n_feats = np.random.randint(2, 100)
        max_depth = np.random.randint(1, 5)

        classifier = np.random.choice([True, False])
        if classifier:
            # create classification problem
            n_classes = np.random.randint(2, 10)
            X, Y = make_blobs(
                n_samples=n_ex, centers=n_classes, n_features=n_feats, random_state=i
            )
            X, X_test, Y, Y_test = train_test_split(X, Y, test_size=0.3, random_state=i)

            # initialize model
            def loss(yp, y):
                return 1 - accuracy_score(yp, y)

            criterion = np.random.choice(["entropy", "gini"])
            mine = DecisionTree(
                classifier=classifier, max_depth=max_depth, criterion=criterion
            )
            gold = DecisionTreeClassifier(
                criterion=criterion,
                max_depth=max_depth,
                splitter="best",
                random_state=i,
            )
        else:
            # create regeression problem
            X, Y = make_regression(n_samples=n_ex, n_features=n_feats, random_state=i)
            X, X_test, Y, Y_test = train_test_split(X, Y, test_size=0.3, random_state=i)

            # initialize model
            criterion = "mse"
            loss = mean_squared_error
            mine = DecisionTree(
                criterion=criterion, max_depth=max_depth, classifier=classifier
            )
            gold = DecisionTreeRegressor(
                criterion=criterion, max_depth=max_depth, splitter="best"
            )

        print("Trial {}".format(i))
        print("\tClassifier={}, criterion={}".format(classifier, criterion))
        print("\tmax_depth={}, n_feats={}, n_ex={}".format(max_depth, n_feats, n_ex))
        if classifier:
            print("\tn_classes: {}".format(n_classes))

        # fit 'em
        mine.fit(X, Y)
        gold.fit(X, Y)

        # get preds on training set
        y_pred_mine = mine.predict(X)
        y_pred_gold = gold.predict(X)

        loss_mine = loss(y_pred_mine, Y)
        loss_gold = loss(y_pred_gold, Y)

        # get preds on test set
        y_pred_mine_test = mine.predict(X_test)
        y_pred_gold_test = gold.predict(X_test)

        loss_mine_test = loss(y_pred_mine_test, Y_test)
        loss_gold_test = loss(y_pred_gold_test, Y_test)

        try:
            np.testing.assert_almost_equal(loss_mine, loss_gold)
            print("\tLoss on training: {}".format(loss_mine))
        except AssertionError as e:
            print("\tTraining losses not equal:\n{}".format(e))

        try:
            np.testing.assert_almost_equal(loss_mine_test, loss_gold_test)
            print("\tLoss on test: {}".format(loss_mine_test))
        except AssertionError as e:
            print("\tTest losses not equal:\n{}".format(e))
        i += 1


def test_RandomForest(N=1):
    np.random.seed(12345)
    i = 1
    while i <= N:
        n_ex = np.random.randint(2, 100)
        n_feats = np.random.randint(2, 100)
        n_trees = np.random.randint(2, 100)
        max_depth = np.random.randint(1, 5)

        classifier = np.random.choice([True, False])
        if classifier:
            # create classification problem
            n_classes = np.random.randint(2, 10)
            X, Y = make_blobs(
                n_samples=n_ex, centers=n_classes, n_features=n_feats, random_state=i
            )
            X, X_test, Y, Y_test = train_test_split(X, Y, test_size=0.3, random_state=i)

            # initialize model
            def loss(yp, y):
                return 1 - accuracy_score(yp, y)

            # initialize model
            criterion = np.random.choice(["entropy", "gini"])
            mine = RandomForest(
                classifier=classifier,
                n_feats=n_feats,
                n_trees=n_trees,
                criterion=criterion,
                max_depth=max_depth,
            )
            gold = RandomForestClassifier(
                n_estimators=n_trees,
                max_features=n_feats,
                criterion=criterion,
                max_depth=max_depth,
                bootstrap=True,
            )
        else:
            # create regeression problem
            X, Y = make_regression(n_samples=n_ex, n_features=n_feats, random_state=i)
            X, X_test, Y, Y_test = train_test_split(X, Y, test_size=0.3, random_state=i)

            # initialize model
            criterion = "mse"
            loss = mean_squared_error
            mine = RandomForest(
                criterion=criterion,
                n_feats=n_feats,
                n_trees=n_trees,
                max_depth=max_depth,
                classifier=classifier,
            )
            gold = RandomForestRegressor(
                n_estimators=n_trees,
                max_features=n_feats,
                criterion=criterion,
                max_depth=max_depth,
                bootstrap=True,
            )

        print("Trial {}".format(i))
        print("\tClassifier={}, criterion={}".format(classifier, criterion))
        print("\tmax_depth={}, n_feats={}, n_ex={}".format(max_depth, n_feats, n_ex))
        if classifier:
            print("\tn_classes: {}".format(n_classes))

        # fit 'em
        mine.fit(X, Y)
        gold.fit(X, Y)

        # get preds
        y_pred_mine = mine.predict(X)
        y_pred_gold = gold.predict(X)

        loss_mine = loss(y_pred_mine, Y)
        loss_gold = loss(y_pred_gold, Y)

        # get preds on test set
        y_pred_mine_test = mine.predict(X_test)
        y_pred_gold_test = gold.predict(X_test)

        loss_mine_test = loss(y_pred_mine_test, Y_test)
        loss_gold_test = loss(y_pred_gold_test, Y_test)

        try:
            np.testing.assert_almost_equal(loss_mine, loss_gold)
            print("\tLoss on training: {}".format(loss_mine))
        except AssertionError as e:
            print("\tTraining losses not equal:\n{}".format(e))

        try:
            np.testing.assert_almost_equal(loss_mine_test, loss_gold_test)
            print("\tLoss on test: {}".format(loss_mine_test))
        except AssertionError as e:
            print("\tTest losses not equal:\n{}".format(e))

        print("PASSED")
        i += 1


def test_gbdt(N=1):
    np.random.seed(12345)
    i = 1
    while i <= N:
        n_ex = np.random.randint(2, 100)
        n_feats = np.random.randint(2, 100)
        n_trees = np.random.randint(2, 100)
        max_depth = np.random.randint(1, 5)

        classifier = np.random.choice([True, False])
        if classifier:
            # create classification problem
            n_classes = np.random.randint(2, 10)
            X, Y = make_blobs(
                n_samples=n_ex, centers=n_classes, n_features=n_feats, random_state=i
            )
            X, X_test, Y, Y_test = train_test_split(X, Y, test_size=0.3, random_state=i)

            # initialize model
            def loss(yp, y):
                return 1 - accuracy_score(yp, y)

            # initialize model
            criterion = np.random.choice(["entropy", "gini"])
            mine = GradientBoostedDecisionTree(
                n_iter=n_trees,
                classifier=classifier,
                max_depth=max_depth,
                learning_rate=0.1,
                loss="crossentropy",
                step_size="constant",
            )
            gold = RandomForestClassifier(
                n_estimators=n_trees,
                max_features=n_feats,
                criterion=criterion,
                max_depth=max_depth,
                bootstrap=True,
            )
        else:
            # create regeression problem
            X, Y = make_regression(n_samples=n_ex, n_features=n_feats, random_state=i)
            X, X_test, Y, Y_test = train_test_split(X, Y, test_size=0.3, random_state=i)

            # initialize model
            criterion = "mse"
            loss = mean_squared_error
            mine = GradientBoostedDecisionTree(
                n_iter=n_trees,
                max_depth=max_depth,
                classifier=classifier,
                learning_rate=0.1,
                loss="mse",
                step_size="constant",
            )
            gold = RandomForestRegressor(
                n_estimators=n_trees,
                max_features=n_feats,
                criterion=criterion,
                max_depth=max_depth,
                bootstrap=True,
            )

        print("Trial {}".format(i))
        print("\tClassifier={}, criterion={}".format(classifier, criterion))
        print("\tmax_depth={}, n_feats={}, n_ex={}".format(max_depth, n_feats, n_ex))
        if classifier:
            print("\tn_classes: {}".format(n_classes))

        # fit 'em
        mine.fit(X, Y)
        gold.fit(X, Y)

        # get preds
        y_pred_mine = mine.predict(X)
        y_pred_gold = gold.predict(X)

        loss_mine = loss(y_pred_mine, Y)
        loss_gold = loss(y_pred_gold, Y)

        # get preds on test set
        y_pred_mine_test = mine.predict(X_test)
        y_pred_gold_test = gold.predict(X_test)

        loss_mine_test = loss(y_pred_mine_test, Y_test)
        loss_gold_test = loss(y_pred_gold_test, Y_test)

        try:
            np.testing.assert_almost_equal(loss_mine, loss_gold)
            print("\tLoss on training: {}".format(loss_mine))
        except AssertionError as e:
            print("\tTraining losses not equal:\n{}".format(e))

        try:
            np.testing.assert_almost_equal(loss_mine_test, loss_gold_test)
            print("\tLoss on test: {}".format(loss_mine_test))
        except AssertionError as e:
            print("\tTest losses not equal:\n{}".format(e))

        print("PASSED")
        i += 1
