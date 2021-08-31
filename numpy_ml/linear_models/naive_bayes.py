import numpy as np


class GaussianNBClassifier:
    def __init__(self):
        """
        A naive Bayes classifier for real-valued data.

        Notes
        -----
        The naive Bayes model assumes the features of each training example
        :math:`\mathbf{x}` are mutually independent given the example label
        :math:`y`:

        .. math::

            P(\mathbf{x}_i \mid y_i) = \prod_{j=1}^M P(x_{i,j} \mid y_i)

        where :math:`M` is the rank of the `i`th example :math:`\mathbf{x}_i`
        and :math:`y_i` is the label associated with the `i`th example.

        Combining the conditional independence assumption with a simple
        application of Bayes' theorem gives the naive Bayes classification
        rule:

        .. math::

            \hat{y} &= \arg \max_y P(y \mid \mathbf{x}) \\
                    &= \arg \max_y  P(y) P(\mathbf{x} \mid y) \\
                    &= \arg \max_y  P(y) \prod_{j=1}^M P(x_j \mid y)

        In the final expression, the prior class probability :math:`P(y)` can
        be specified in advance or estimated empirically from the training
        data.

        In the Gaussian version of the naive Bayes model, the feature
        likelihood is assumed to be normally distributed for each class:

        .. math::

            \mathbf{x}_i \mid y_i = c, \theta \sim \mathcal{N}(\mu_c, \Sigma_c)

        where :math:`\theta` is the set of model parameters: :math:`\{\mu_1,
        \Sigma_1, \ldots, \mu_K, \Sigma_K\}`, :math:`K` is the total number of
        unique classes present in the data, and the parameters for the Gaussian
        associated with class :math:`c`, :math:`\mu_c` and :math:`\Sigma_c`
        (where :math:`1 \leq c \leq K`), are estimated via MLE from the set of
        training examples with label :math:`c`.
        """
        pass

    # Separate the dataset into a subset of data for each class

    def separate_classes(self, X, y):
        """
        Separates the dataset in to a subset of data for each class.
        Parameters:
        ------------
        X- array, list of features
        y- list, target
        Returns:
        A dictionary with y as keys and assigned X as values.
        """
        separated_classes = {}
        for i in range(len(X)):
            feature_values = X[i]
            class_name = y[i]
            if class_name not in separated_classes:
                separated_classes[class_name] = []
            separated_classes[class_name].append(feature_values)
        return separated_classes

    # Standard deviation and mean are required for the (Gaussian) distribution function

    def stat_info(self, X):
        """
        Calculates standard deviation and mean of features.
        Parameters:
        ------------
        X- array , list of features
        Returns:
        A dictionary with STD and Mean as keys and assigned features STD and Mean as values.
        """
        for feature in zip(*X):
            yield {
                'std' : np.std(feature),
                'mean' : np.mean(feature)
            }

    def fit(self, X, y):
        """
        Fit the model parameters via maximum likelihood.

        Notes
        -----
        The model parameters are stored in the :py:attr:`class_summary` attribute.
        The following keys are present:

        prior_proba :py:class:`ndarray <numpy.ndarray>` of shape `(K,)`
            Prior probability of each of the `K` label classes, estimated
            empirically from the training data
        summary : Dictionary having both the keys and the values of 
            py:class:`ndarray <numpy.ndarray>` of shape `(K, M)`
            Feature means for each of the `K` label classes along with the
            Feature STDs for each of the `K` label classes

        Parameters
        ----------
        X :py:class:`ndarray <numpy.ndarray>` of shape `(N, M)`
            A dataset consisting of `N` examples, each of dimension `M`
        y :py:class:`ndarray <numpy.ndarray>` of shape `(N,)`
            The class label for each of the `N` examples in `X`

        Returns
        -------
        Dictionary with the prior probability, mean, and standard deviation of each class
        """

        separated_classes = self.separate_classes(X, y)
        self.class_summary = {}

        for class_name, feature_values in separated_classes.items():
            self.class_summary[class_name] = {
                'prior_proba': len(feature_values)/len(X),
                'summary': [i for i in self.stat_info(feature_values)],
            }
        return self.class_summary

    # Gaussian distribution function

    def distribution(self, x, mean, std):
        """
        Holds the computation for the Gaussian Distribution Function

        Parameters
        ----------
        x:  type: float, value of feature 'x'
        mean:   type: float, the mean value of feature 'x'
        stdev:  type: float, the standard deviation of feature 'x'

        Returns
        --------
        f:  A float value of Normal Probability
        """

        exponent = np.exp(-((x-mean)**2 / (2*std**2)))
        f = exponent / (np.sqrt(2*np.pi)*std)
        return f

    # Required predict method, to predict the class

    def predict(self, X):
        """
        Use the trained classifier to predict the class label for each example
        in **X**.

        Parameters
        ----------
        X: :py:class:`ndarray <numpy.ndarray>` of shape `(N, M)`
            A dataset of `N` examples, each of dimension `M`

        Returns
        -------
        MAPs : :py:class:`ndarray <numpy.ndarray>` of shape `(N)`
            The predicted class labels for each example in `X`
        """

        # Using Maximum a posteriori (MAP) probability

        MAPs = []

        for row in X:
            joint_proba = {}
            
            for class_name, features in self.class_summary.items():
                total_features =  len(features['summary'])
                likelihood = 1

                for idx in range(total_features):
                    feature = row[idx]
                    mean = features['summary'][idx]['mean']
                    stdev = features['summary'][idx]['std']
                    normal_proba = self.distribution(feature, mean, stdev)
                    likelihood *= normal_proba
                prior_proba = features['prior_proba']
                joint_proba[class_name] = prior_proba * likelihood

            MAP = max(joint_proba, key= joint_proba.get)
            MAPs.append(MAP)

        return MAPs

    # Calculate the model's accuracy

    def accuracy(self, y_test, y_pred):
        """
        Calculates model's accuracy using label comparison

        Parameters
        ------------
        y_test: :py:class:`ndarray <numpy.ndarray>` of shape `(N,)`
            The true class label for each of the `N` examples in `X`
        y_pred: :py:class:`ndarray <numpy.ndarray>` of shape `(N,)`
            The predicted class label for each of the `N` examples in `X`

        Returns
        --------
        acc:    A number between 0-1, representing the percentage of correct predictions.
            The accuracy of the GaussianNB model using numpy-ml environment
        """

        true_true = 0
        for y_t, y_p in zip(y_test, y_pred):
            if y_t == y_p:
                true_true += 1 
        acc = true_true / len(y_test)
        return acc