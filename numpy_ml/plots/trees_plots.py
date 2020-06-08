# flake8: noqa
import numpy as np

from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.datasets import make_blobs, make_regression
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

# https://seaborn.pydata.org/generated/seaborn.set_context.html
# https://seaborn.pydata.org/generated/seaborn.set_style.html
import seaborn as sns

sns.set_style("white")
sns.set_context("paper", font_scale=0.9)

from numpy_ml.trees import GradientBoostedDecisionTree, DecisionTree, RandomForest


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
