Linear models
#############

.. raw:: html

   <h2>Ordinary and Weighted Linear Least Squares</h2>

In weighted linear least-squares regression (WLS), a real-valued target
:math:`y_i`, is modeled as a linear combination of covariates
:math:`\mathbf{x}_i` and model coefficients **b**:

.. math::

    y_i = \mathbf{b}^\top \mathbf{x}_i + \epsilon_i

In the above equation, :math:`\epsilon_i \sim \mathcal{N}(0, \sigma_i^2)` is a
normally distributed error term with variance :math:`\sigma_i^2`. Ordinary
least squares (OLS) is a special case of this model where the variance is fixed
across all examples, i.e., :math:`\sigma_i = \sigma_j \ \forall i,j`. The
maximum likelihood model parameters, :math:`\hat{\mathbf{b}}_{WLS}`, are those
that minimize the weighted squared error between the model predictions and the
true values:

.. math::

    \mathcal{L} = ||\mathbf{W}^{0.5}(\mathbf{y} - \mathbf{bX})||_2^2

where :math:`\mathbf{W}` is a diagonal matrix of the example weights. In OLS,
:math:`\mathbf{W}` is the identity matrix. The maximum likelihood estimate for
the model parameters can be computed in closed-form using the normal equations:

.. math::

    \hat{\mathbf{b}}_{WLS} =
        (\mathbf{X}^\top \mathbf{WX})^{-1} \mathbf{X}^\top \mathbf{Wy}


**Models**

- :class:`~numpy_ml.linear_models.LinearRegression`

.. raw:: html

   <h2>Ridge Regression</h2>

Ridge regression uses the same simple linear regression model but adds an
additional penalty on the `L2`-norm of the coefficients to the loss function.
This is sometimes known as Tikhonov regularization.

In particular, the ridge model is the same as the OLS model:

.. math::

    \mathbf{y} = \mathbf{bX} + \mathbf{\epsilon}

where :math:`\epsilon \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})`,
except now the error for the model is calculated as

.. math::

    \mathcal{L} = ||\mathbf{y} - \mathbf{bX}||_2^2 + \alpha ||\mathbf{b}||_2^2

The MLE for the model parameters **b** can be computed in closed form via
the adjusted normal equation:

.. math::

    \hat{\mathbf{b}}_{Ridge} =
        (\mathbf{X}^\top \mathbf{X} + \alpha \mathbf{I})^{-1} \mathbf{X}^\top \mathbf{y}

where :math:`(\mathbf{X}^\top \mathbf{X} + \alpha \mathbf{I})^{-1}
\mathbf{X}^\top` is the pseudoinverse / Moore-Penrose inverse adjusted for
the `L2` penalty on the model coefficients.

**Models**

- :class:`~numpy_ml.linear_models.RidgeRegression`

.. raw:: html

   <h2>Bayesian Linear Regression</h2>

In its general form, Bayesian linear regression extends the simple linear
regression model by introducing priors on model parameters *b* and/or the
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

    p(y^* \mid X^*, X, Y) = \int_{b} p(y^* \mid X^*, b) p(b \mid X, y) \ \text{d}b

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

    b | \sigma, V  \sim  \mathcal{N}(\mu, \sigma^2 V)

where :math:`\mu`, :math:`\sigma` and :math:`V` are hyperparameters. Ridge
regression is a special case of this model where :math:`\mu = 0`,
:math:`\sigma = 1` and :math:`V = I` (i.e., the prior on *b* is a zero-mean,
unit covariance Gaussian).

Due to the conjugacy of the above prior with the Gaussian likelihood, there
exists a closed-form solution for the posterior over the model
parameters:

.. math::

    A  &=  (V^{-1} + X^\top X)^{-1} \\
    \mu_b  &=  A V^{-1} \mu + A X^\top y \\
    \Sigma_b  &=  \sigma^2 A \\

The model posterior is then

.. math::

    b \mid X, y  \sim  \mathcal{N}(\mu_b, \Sigma_b)

We can also compute a closed-form solution for the posterior predictive distribution as
well:

.. math::

    y^* \mid X^*, X, Y \sim \mathcal{N}(X^* \mu_b, \ \ X^* \Sigma X^{* \top} + I)

where :math:`X^*` is the matrix of new data we wish to predict, and :math:`y^*`
are the predicted targets for those data.

**Models**

- :class:`~numpy_ml.linear_models.BayesianLinearRegressionKnownVariance`


.. raw:: html

   <h4>Unknown variance</h4>

--------------------------------

If *both* *b* and the error variance :math:`\sigma^2` are unknown, the
conjugate prior for the Gaussian likelihood is the Normal-Gamma
distribution (univariate likelihood) or the Normal-Inverse-Wishart
distribution (multivariate likelihood).

    **Univariate**

    .. math::

        b, \sigma^2  &\sim  \text{NG}(\mu, V, \alpha, \beta) \\
        \sigma^2  &\sim  \text{InverseGamma}(\alpha, \beta) \\
        b \mid \sigma^2  &\sim  \mathcal{N}(\mu, \sigma^2 V)

    where :math:`\alpha, \beta, V`, and :math:`\mu` are parameters of the
    prior.

    **Multivariate**

    .. math::

        b, \Sigma  &\sim  \mathcal{NIW}(\mu, \lambda, \Psi, \rho) \\
        \Sigma  &\sim  \mathcal{W}^{-1}(\Psi, \rho) \\
        b \mid \Sigma  &\sim  \mathcal{N}(\mu, \frac{1}{\lambda} \Sigma)

    where :math:`\mu, \lambda, \Psi`, and :math:`\rho` are
    parameters of the prior.


Due to the conjugacy of the above priors with the Gaussian likelihood,
there exists a closed-form solution for the posterior over the model
parameters:

.. math::

    B  &=  y - X \mu \\
    \text{shape}  &=  N + \alpha \\
    \text{scale}  &=  \frac{1}{\text{shape}} (\alpha \beta + B^\top (X V X^\top + I)^{-1} B) \\

where

.. math::

    \sigma^2 \mid X, y  &\sim  \text{InverseGamma}(\text{shape}, \text{scale}) \\
    A  &=  (V^{-1} + X^\top X)^{-1} \\
    \mu_b  &=  A V^{-1} \mu + A X^\top y \\
    \Sigma_b  &=  \sigma^2 A

The model posterior is then

.. math::

    b | X, y, \sigma^2 \sim \mathcal{N}(\mu_b, \Sigma_b)

We can also compute a closed-form solution for the posterior predictive distribution:

.. math::

    y^* \mid X^*, X, Y \sim \mathcal{N}(X^* \mu_b, \ X^* \Sigma_b X^{* \top} + I)

**Models**

- :class:`~numpy_ml.linear_models.BayesianLinearRegressionUnknownVariance`

.. raw:: html

   <h2>Naive Bayes Classifier</h2>

The naive Bayes model assumes the features of a training example
:math:`\mathbf{x}` are mutually independent given the example label :math:`y`:

.. math::

    P(\mathbf{x}_i \mid y_i) = \prod_{j=1}^M P(x_{i,j} \mid y_i)

where :math:`M` is the rank of the :math:`i^{th}` example :math:`\mathbf{x}_i`
and :math:`y_i` is the label associated with the :math:`i^{th}` example.

Combining this conditional independence assumption with a simple application of
Bayes' theorem gives the naive Bayes classification rule:

.. math::

    \hat{y} &= \arg \max_y P(y \mid \mathbf{x}) \\
            &= \arg \max_y  P(y) P(\mathbf{x} \mid y) \\
            &= \arg \max_y  P(y) \prod_{j=1}^M P(x_j \mid y)

The prior class probability :math:`P(y)` can be specified in advance or
estimated empirically from the training data.

**Models**

- :class:`~numpy_ml.linear_models.GaussianNBClassifier`

.. raw:: html

   <h2>Generalized Linear Model</h2>

The generalized linear model (GLM) assumes that each target/dependent variable
:math:`y_i` in target vector :math:`\mathbf{y} = (y_1, \ldots, y_n)`, has been
drawn independently from a pre-specified distribution in the exponential family
with unknown mean :math:`\mu_i`. The GLM models a (one-to-one, continuous,
differentiable) function, *g*, of this mean value as a linear combination of
the model parameters :math:`\mathbf{b}` and observed covariates,
:math:`\mathbf{x}_i` :

.. math::

    g(\mathbb{E}[y_i \mid \mathbf{x}_i]) =
        g(\mu_i) = \mathbf{b}^\top \mathbf{x}_i

where *g* is known as the link function.  The choice of link function is
informed by the instance of the exponential family the target is drawn from.

**Models**

- :class:`~numpy_ml.linear_models.GeneralizedLinearModel`

.. toctree::
   :maxdepth: 2
   :hidden:

   numpy_ml.linear_models.lm
