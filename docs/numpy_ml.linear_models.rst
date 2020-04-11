Linear models
#############

The linear models module contains several popular instances of the generalized linear model (GLM).

.. raw:: html

   <h2>Linear Regression</h2>

The simple linear regression model is

.. math::

    \mathbf{y} = \mathbf{bX} + \mathbf{\epsilon}

where

.. math::

    \epsilon \sim \mathcal{N}(0, \sigma^2 I)

In probabilistic terms this corresponds to

.. math::

    \mathbf{y} - \mathbf{bX}  &\sim  \mathcal{N}(0, \sigma^2 I) \\
    \mathbf{y} \mid \mathbf{X}, \mathbf{b}  &\sim  \mathcal{N}(\mathbf{bX}, \sigma^2 I)

The loss for the model is simply the squared error between the model
predictions and the true values:

.. math::

    \mathcal{L} = ||\mathbf{y} - \mathbf{bX}||_2^2

The MLE for the model parameters **b** can be computed in closed form via
the normal equation:

.. math::

    \mathbf{b}_{MLE} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}

where :math:`(\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top` is known
as the pseudoinverse / Moore-Penrose inverse.

**Models**

- :class:`~numpy_ml.linear_models.LinearRegression`

.. raw:: html

   <h2>Ridge Regression</h2>

Ridge regression uses the same simple linear regression model but adds an
additional penalty on the `L2`-norm of the coefficients to the loss function.
This is sometimes known as Tikhonov regularization.

In particular, the ridge model is still simply

.. math::

    \mathbf{y} = \mathbf{bX} + \mathbf{\epsilon}

where

.. math::

    \epsilon \sim \mathcal{N}(0, \sigma^2 I)

except now the error for the model is calcualted as

.. math::

    \mathcal{L} = ||\mathbf{y} - \mathbf{bX}||_2^2 + \alpha ||\mathbf{b}||_2^2

The MLE for the model parameters **b** can be computed in closed form via
the adjusted normal equation:

.. math::

    \mathbf{b}_{MLE} = (\mathbf{X}^\top \mathbf{X} + \alpha I)^{-1} \mathbf{X}^\top \mathbf{y}

where :math:`(\mathbf{X}^\top \mathbf{X} + \alpha I)^{-1}
\mathbf{X}^\top` is the pseudoinverse / Moore-Penrose inverse adjusted for
the `L2` penalty on the model coefficients.

**Models**

- :class:`~numpy_ml.linear_models.RidgeRegression`

.. raw:: html

   <h2>Bayesian Linear Regression</h2>

In its general form, Bayesian linear regression extends the simple linear
regression model by introducing priors on model parameters b and/or the
error variance :math:`\sigma^2`.

The introduction of a prior allows us to quantify the uncertainty in our
parameter estimates for b by replacing the MLE point estimate in simple
linear regression with an entire posterior *distribution*, :math:`p(b \mid X, y,
\sigma)`, simply by applying Bayes rule:

.. math::

    p(b \mid X, y) = \frac{ p(y \mid X, b) p(b \mid \sigma) }{p(y \mid X)}

We can also quantify the uncertainty in our predictions :math:`y^*` for some new
data :math:`X^*` with the posterior predictive distribution:

.. math::

    p(y^* \mid X^*, X, Y) = \int_{b} p(y^* \mid X^*, b) p(b \mid X, y) db

Depending on the choice of prior it may be impossible to compute an
analytic form for the posterior / posterior predictive distribution. In
these cases, it is common to use approximations, either via MCMC or
variational inference.

.. raw:: html

   <h4>Known variance</h4>

--------------------------------

If we happen to already know the error variance :math:`\sigma^2`, the conjugate
prior on `b` is Gaussian. A common parameterization is:

.. math::

    b | \sigma, b_V  \sim  \mathcal{N}(b_{mean}, \sigma^2 b_V)

where :math:`b_{mean}`, :math:`\sigma` and :math:`b_V` are hyperparameters. Ridge
regression is a special case of this model where :math:`b_{mean}` = 0,
:math:`\sigma` = 1 and :math:`b_V = I` (ie., the prior on `b` is a zero-mean,
unit covariance Gaussian).

Due to the conjugacy of the above prior with the Gaussian likelihood, there
exists a closed-form solution for the posterior over the model
parameters:

.. math::

    A  &=  (b_V^{-1} + X^\top X)^{-1} \\
    \mu_b  &=  A b_V^{-1} b_{mean} + A X^\top y \\
    \text{cov}_b  &=  \sigma^2 A \\

The model posterior is then

.. math::

    b \mid X, y  \sim  \mathcal{N}(\mu_b, \text{cov}_b)

We can also compute a closed-form solution for the posterior predictive distribution as
well:

.. math::

    y^* \mid X^*, X, Y \sim \mathcal{N}(X^* \mu_b, \ \ X^* \text{cov}_b X^{* \top} + I)

where :math:`X^*` is the matrix of new data we wish to predict, and :math:`y^*`
are the predicted targets for those data.

**Models**

- :class:`~numpy_ml.linear_models.BayesianLinearRegressionKnownVariance`


.. raw:: html

   <h4>Unknown variance</h4>

--------------------------------

If *both* b and the error variance :math:`\sigma^2` are unknown, the
conjugate prior for the Gaussian likelihood is the Normal-Gamma
distribution (univariate likelihood) or the Normal-Inverse-Wishart
distribution (multivariate likelihood).

    **Univariate**

    .. math::

        b, \sigma^2  &\sim  \text{NG}(b_{mean}, b_{V}, \alpha, \beta) \\
        \sigma^2  &\sim  \text{InverseGamma}(\alpha, \beta) \\
        b \mid \sigma^2  &\sim  \mathcal{N}(b_{mean}, \sigma^2 b_{V})

    where :math:`\alpha, \beta, b_{V}`, and :math:`b_{mean}` are
    parameters of the prior.

    **Multivariate**

    .. math::

        b, \Sigma  &\sim  \mathcal{NIW}(b_{mean}, \lambda, \Psi, \rho) \\
        \Sigma  &\sim  \mathcal{W}^{-1}(\Psi, \rho) \\
        b \mid \Sigma  &\sim  \mathcal{N}(b_{mean}, \frac{1}{\lambda} \Sigma)

    where :math:`b_{mean}, \lambda, \Psi`, and :math:`\rho` are
    parameters of the prior.


Due to the conjugacy of the above priors with the Gaussian likelihood,
there exists a closed-form solution for the posterior over the model
parameters:

.. math::

    B  &=  y - X b_{mean} \\
    \text{shape}  &=  N + \alpha \\
    \text{scale}  &=  \frac{1}{\text{shape}} (\alpha \beta + B^\top (X b_V X^\top + I)^{-1} B) \\

where

.. math::

    \sigma^2 \mid X, y  &\sim  \text{InverseGamma}(\text{shape}, \text{scale}) \\
    A  &=  (b_V^{-1} + X^\top X)^{-1} \\
    \mu_b  &=  A b_V^{-1} b_{mean} + A X^\top y \\
    \text{cov}_b  &=  \sigma^2 A

The model posterior is then

.. math::

    b | X, y, \sigma^2 \sim \mathcal{N}(\mu_b, \text{cov}_b)

We can also compute a closed-form solution for the posterior predictive distribution:

.. math::

    y^* \mid X^*, X, Y \sim \mathcal{N}(X^* \mu_b, \ X^* \text{cov}_b X^{* \top} + I)

**Models**

- :class:`~numpy_ml.linear_models.BayesianLinearRegressionUnknownVariance`

.. toctree::
   :maxdepth: 2
   :hidden:

   numpy_ml.linear_models.lm
