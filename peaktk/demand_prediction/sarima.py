
# reference:
# https://machinelearningmastery.com/how-to-grid-search-sarima-model-hyperparameters-for-time-series-forecasting-in-python/
# grid search sarima hyperparameters for monthly mean temp dataset
import numpy as np
from math import sqrt
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
from warnings import catch_warnings
from warnings import filterwarnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from pandas import read_csv
import matplotlib.pyplot as plt 

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class SARIMA(BaseEstimator):
    """ SARIMA - Seasonal AutoRegressive Integrated Moving Average

    A class to train a SARIMA model and use the model to predict electric demand.

    Parameters
    ----------
    param : str, default='demo_param'
        A parameter used for demonstation of how to pass and store paramters.

    Examples
    --------
    >>> from peaktk.demand_prediction.sarima import SARIMA
    >>> X = ...
    >>> y = ...
    >>> estimator = SARIMA()
    >>> estimator.fit(X_train, y_train)
    SARIMA()
    >>> estimator.predict(X_test)
    [...]
    """

    def __init__(self, order=(1,1,2), sorder=(1,1,2,7), trend='c', enforce_stationarity=False, enforce_invertibility=False):
        """Init function. Create SARIMA object with one attribute."""
        self.model = None
        self.order = order
        self.sorder = sorder
        self.trend = trend
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility


    def fit(self, X):
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
        self.model = SARIMAX(X, order=self.order, seasonal_order=self.sorder, trend=self.trend, enforce_stationarity=self.enforce_stationarity, enforce_invertibility=self.enforce_invertibility)
        
        self.model_fit = self.model.fit(disp=False, method='powell')

        self.len_X = len(X)

        self.is_fitted_ = True


        # `fit` should always return `self`
        return self

    def predict(self, number_of_day=1):
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
        check_is_fitted(self, 'is_fitted_')

        yhat = self.model_fit.predict(start=self.len_X, end=self.len_X + number_of_day - 1)
        return yhat

    def fit_predict(self, X):
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
        y = []
        for x in X:
            self.fit(x)
            y_hat = self.predict()[0]
            y.append(y_hat)
        return np.array(y)




    # # one-step sarima forecast
    # def sarima_forecast(history, config):
    #   order, sorder, trend = config
    #   # define model
    #   model = SARIMAX(history, order=order, seasonal_order=sorder, trend=trend, enforce_stationarity=False, enforce_invertibility=False)
    #   # fit model
    #   model_fit = model.fit(disp=False, method='powell')
    #   # make one step forecast
    #   yhat = model_fit.predict(len(history), len(history))
    #   return yhat[0]
     
    # # walk-forward validation for univariate data
    # def walk_forward_validation(train, test, cfg):
    #   predictions = []
    #   # seed history with training dataset
    #   history = [x for x in train]
    #   # step over each time-step in the test set
    #   for i in range(len(test)):
    #       # fit model and make forecast for history
    #       yhat = sarima_forecast(history, cfg)
    #       # store forecast in list of predictions
    #       predictions.append(yhat)
    #       # add actual observation to history for the next loop
    #       history.append(test[i])
    #   # estimate prediction error
    #   error = mean_squared_error(actual, predicted)
    #   print(error)
    #   print(predictions)
    #   return error, predictions
 
    # # score a model, return None on failure
    # def score_model(train, test, cfg, debug=False):
    #   result = None
    #   # convert config to a key
    #   key = str(cfg)
    #   # show all warnings and fail on exception if debugging
    #   if debug:
    #       result = walk_forward_validation(train, test, cfg)
    #   else:
    #       # one failure during model validation suggests an unstable config
    #       try:
    #           # never show warnings when grid searching, too noisy
    #           with catch_warnings():
    #               filterwarnings("ignore")
    #               result = walk_forward_validation(train, test, cfg)
    #       except:
    #           error = None
    #   # check for an interesting result
    #   if result is not None:
    #       print(' > Model[%s] %.3f' % (key, result))
    #   return (key, result)
 
    # # grid search configs
    # def grid_search(train, test, cfg_list):

    #   scores = [score_model(train, test, cfg) for cfg in cfg_list]
    #   # remove empty results
    #   scores = [r for r in scores if r[1] != None]
    #   # sort configs by error, asc
    #   scores.sort(key=lambda tup: tup[1])
    #   return scores
 
    # # create a set of sarima configs to try
    # def sarima_configs(seasonal=[0]):
    #   models = list()
    #   # define config lists
    #   p_params = [0, 1, 2]
    #   d_params = [0, 1]
    #   q_params = [0, 1, 2]
    #   t_params = ['n','c','t','ct']
    #   P_params = [0, 1, 2]
    #   D_params = [0, 1]
    #   Q_params = [0, 1, 2]
    #   m_params = seasonal
    #   # create config instances
    #   for p in p_params:
    #       for d in d_params:
    #           for q in q_params:
    #               for t in t_params:
    #                   for P in P_params:
    #                       for D in D_params:
    #                           for Q in Q_params:
    #                               for m in m_params:
    #                                   cfg = [(p,d,q), (P,D,Q,m), t]
    #                                   models.append(cfg)
    #   return models

