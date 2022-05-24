
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class SVRDemandPrediction(BaseEstimator):
    """ Deep Learning - Long Short Term Memory

    A class to train an LSTM model and use the model to predict electric demand.

    Parameters
    ----------
    param : str, default='demo_param'
        A parameter used for demonstation of how to pass and store paramters.

    Examples
    --------
    >>> from peaktk.demand_prediction.lstm_dp import LSTMDemandPrediction
    >>> X = ...
    >>> y = ...
    >>> estimator = SVRDemandPrediction()
    >>> estimator.fit(X_train, y_train)
    SVRDemandPrediction()
    >>> estimator.predict(X_test)
    [...]
    """

    def __init__(self, kernel='rbf', C=1.0, epsilon=0.2, coef0=0.0, tol=1e-3, gamma='auto'):
        """Init function. Create DL_LSTM object with ..... attribute."""
        self.model = None
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        self.coef0 = coef0
        self.tol = tol
        self.gamma = gamma

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit(self, X, y):
        """A reference implementation of a fitting function.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).
        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y, accept_sparse=True)
        self.is_fitted_ = True

        X_shape = X.shape

        # set default params
        # defaultKwargs = { 'epochs': 500, 'batch_size': 72, 'verbose': 0 }
        # kwargs = { **defaultKwargs, **kwargs }

        self.model = SVR(kernel=self.kernel, C=self.C, epsilon=self.epsilon, coef0=self.coef0, tol=self.tol, gamma=self.gamma)

        history = self.model.fit(X, y)
        # `fit` should always return `self`
        return self

    def predict(self, X):
        """ A reference implementation of a predicting function.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        Returns
        -------
        y : ndarray, shape (n_samples,)
            Returns an array of ones.
        """
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')
        return self.model.predict(X)



