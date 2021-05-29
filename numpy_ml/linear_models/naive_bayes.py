import numpy as np 

class GaussianNB():
    """
    Gaussian Naive Bayes 

    Assume each class conditional feature distribution is
    independent and estimate the mean and variance from the
    training data

    Parameters
    ----------
    epsilon: float
        a value that add to variance to prevent numerical error
    
    Attributes
    ----------
    num_class : ndarray of shape (n_classes,)
        count of each class in the training sample

    mean: ndarray of shape (n_classes,)
            mean of each variance
    
    sigma: ndarray of shape (n_classes,)
        variance of each class
    
    prior :  ndarray of shape (n_classes,)
            probability of each class

    """
    def __init__(self,eps=1e-6):
        self.eps = eps 
    
    def fit(self,X,y):
        """
        Train the model with X,y

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Input data
        y: ndarray of shape (n_samples,)
            Target
        
        returns
        --------
        self: object
        """
        
        self.n_sample, self.n_features = X.shape
        self.labels = np.unique(y)
        self.n_classes = len(self.labels)

        self.mean = np.zeros((self.n_classes,self.n_features))
        self.sigma = np.zeros((self.n_classes,self.n_features))
        self.prior = np.zeros((self.n_classes,))

        for i in range(self.n_classes):
            X_c = X[y==i,:]

            self.mean[i,:] = np.mean(X_c,axis=0)
            self.sigma[i,:] = np.var(X_c,axis=0) + self.eps
            self.prior[i] = X_c.shape[0]/self.n_sample

        return self

    def predict(self,X):
        """
        used the trained model to generate prediction

        Parameters
        ---------
        X: ndarray of shape (n_samples, n_features)
            Input data

        returns
        -------
        probs : ndarray of shape (n_samples, n_classes)
                The model predictions for each items in X to be in each class
        """

        probs = np.zeros((X.shape[0],self.n_classes))
        for i in range(self.n_classes):
            probs[:,i] = self.prob(X,self.mean[i,:],self.sigma[i,:],self.prior[i])


        return probs

    def prob(self,X,mean,sigma,prior):
        """
        compute the joint log likelihood of data based on gaussian distribution

        X: ndarray of shape (n_samples, n_features)
            Input data

        mean: ndarray of shape (n_classes,)
            mean of each variance
        
        sigma: ndarray of shape (n_classes,)
            variance of each class
        
        prior :  ndarray of shape (n_classes,)
                probability of each class

        returns
        -------
        joint_log_likelihood : ndarry of shape (n_samples,)
            joint log likelihood of data
        
        """

        prob = -self.n_features / 2 * np.log(2 * np.pi) - 0.5 * np.sum(
            np.log(sigma )
            )
        prob -= 0.5 * np.sum(np.power(X -mean, 2) / (sigma), 1)

        joint_log_likelihood = prior + prob
        return joint_log_likelihood









    
