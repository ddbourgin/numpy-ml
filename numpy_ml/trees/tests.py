import numpy as np

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.datasets import make_regression
from sklearn.datasets.samples_generator import make_blobs
from sklearn.model_selection import train_test_split

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# https://seaborn.pydata.org/generated/seaborn.set_context.html
# https://seaborn.pydata.org/generated/seaborn.set_style.html
import seaborn as sns

sns.set_style("white")
sns.set_context("paper", font_scale=0.9)

from .gbdt import GradientBoostedDecisionTree
from .dt import DecisionTree, Node, Leaf
from .rf import RandomForest


def random_tensor(shape, standardize=False):
    eps = np.finfo(float).eps
    offset = np.random.randint(-300, 300, shape)
    X = np.random.rand(*shape) + offset

    if standardize:
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + eps)
    return X


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


def test_DecisionTree():
    i = 1
    np.random.seed(12345)
    while True:
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


def test_RandomForest():
    np.random.seed(12345)
    i = 1
    while True:
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


def test_gbdt():
    np.random.seed(12345)
    i = 1
    while True:
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
                classifier=classifier,
                n_trees=n_trees,
                max_depth=max_depth,
                learning_rate=0.1,
                loss="crossentropy",
                step_size="constant",
                split_criterion=criterion,
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
                n_trees=n_trees,
                max_depth=max_depth,
                classifier=classifier,
                learning_rate=0.1,
                loss="mse",
                step_size="constant",
                split_criterion=criterion,
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


def plot():
    fig, axes = plt.subplots(4, 4)
    fig.set_size_inches(10, 10)
    for ax in axes.flatten():
        n_ex = 100
        n_trees = 50
        n_feats = np.random.randint(2, 100)
        max_depth_d = np.random.randint(1, 100)
        max_depth_r = np.random.randint(1, 10)

        classifier = np.random.choice([True, False])
        if classifier:
            # create classification problem
            n_classes = np.random.randint(2, 10)
            X, Y = make_blobs(n_samples=n_ex, centers=n_classes, n_features=2)
            X, X_test, Y, Y_test = train_test_split(X, Y, test_size=0.3)
            n_feats = min(n_feats, X.shape[1])

            # initialize model
            def loss(yp, y):
                return accuracy_score(yp, y)

            # initialize model
            criterion = np.random.choice(["entropy", "gini"])
            mine = RandomForest(
                classifier=classifier,
                n_feats=n_feats,
                n_trees=n_trees,
                criterion=criterion,
                max_depth=max_depth_r,
            )
            mine_d = DecisionTree(
                criterion=criterion, max_depth=max_depth_d, classifier=classifier
            )
            mine_g = GradientBoostedDecisionTree(
                n_trees=n_trees,
                max_depth=max_depth_d,
                classifier=classifier,
                learning_rate=1,
                loss="crossentropy",
                step_size="constant",
                split_criterion=criterion,
            )

        else:
            # create regeression problem
            X, Y = make_regression(n_samples=n_ex, n_features=1)
            X, X_test, Y, Y_test = train_test_split(X, Y, test_size=0.3)
            n_feats = min(n_feats, X.shape[1])

            # initialize model
            criterion = "mse"
            loss = mean_squared_error
            mine = RandomForest(
                criterion=criterion,
                n_feats=n_feats,
                n_trees=n_trees,
                max_depth=max_depth_r,
                classifier=classifier,
            )
            mine_d = DecisionTree(
                criterion=criterion, max_depth=max_depth_d, classifier=classifier
            )
            mine_g = GradientBoostedDecisionTree(
                n_trees=n_trees,
                max_depth=max_depth_d,
                classifier=classifier,
                learning_rate=1,
                loss="mse",
                step_size="adaptive",
                split_criterion=criterion,
            )

        # fit 'em
        mine.fit(X, Y)
        mine_d.fit(X, Y)
        mine_g.fit(X, Y)

        # get preds on test set
        y_pred_mine_test = mine.predict(X_test)
        y_pred_mine_test_d = mine_d.predict(X_test)
        y_pred_mine_test_g = mine_g.predict(X_test)

        loss_mine_test = loss(y_pred_mine_test, Y_test)
        loss_mine_test_d = loss(y_pred_mine_test_d, Y_test)
        loss_mine_test_g = loss(y_pred_mine_test_g, Y_test)

        if classifier:
            entries = [
                ("RF", loss_mine_test, y_pred_mine_test),
                ("DT", loss_mine_test_d, y_pred_mine_test_d),
                ("GB", loss_mine_test_g, y_pred_mine_test_g),
            ]
            (lbl, test_loss, preds) = entries[np.random.randint(3)]
            ax.set_title("{} Accuracy: {:.2f}%".format(lbl, test_loss * 100))
            for i in np.unique(Y_test):
                ax.scatter(
                    X_test[preds == i, 0].flatten(),
                    X_test[preds == i, 1].flatten(),
                    #  s=0.5,
                )
        else:
            X_ax = np.linspace(
                np.min(X_test.flatten()) - 1, np.max(X_test.flatten()) + 1, 100
            ).reshape(-1, 1)
            y_pred_mine_test = mine.predict(X_ax)
            y_pred_mine_test_d = mine_d.predict(X_ax)
            y_pred_mine_test_g = mine_g.predict(X_ax)

            ax.scatter(X_test.flatten(), Y_test.flatten(), c="b", alpha=0.5)
            #  s=0.5)
            ax.plot(
                X_ax.flatten(),
                y_pred_mine_test_g.flatten(),
                #  linewidth=0.5,
                label="GB".format(n_trees, n_feats, max_depth_d),
                color="red",
            )
            ax.plot(
                X_ax.flatten(),
                y_pred_mine_test.flatten(),
                #  linewidth=0.5,
                label="RF".format(n_trees, n_feats, max_depth_r),
                color="cornflowerblue",
            )
            ax.plot(
                X_ax.flatten(),
                y_pred_mine_test_d.flatten(),
                #  linewidth=0.5,
                label="DT".format(max_depth_d),
                color="yellowgreen",
            )
            ax.set_title(
                "GB: {:.1f} / RF: {:.1f} / DT: {:.1f} ".format(
                    loss_mine_test_g, loss_mine_test, loss_mine_test_d
                )
            )
            ax.legend()
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
    plt.savefig("plot.png", dpi=300)
    plt.close("all")
