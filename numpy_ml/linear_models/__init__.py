"""A module containing assorted linear models."""

from .ridge import RidgeRegression
from .glm import GeneralizedLinearModel
from .logistic import LogisticRegression
from .bayesian_regression import (
    BayesianLinearRegressionKnownVariance,
    BayesianLinearRegressionUnknownVariance,
)
from .naive_bayes import GaussianNBClassifier
from .linear_regression import LinearRegression
