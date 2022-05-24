
"""
Write comment about this class
"""
from keras import regularizers
from keras.optimizers import adam_v2
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class LSTMDemandPrediction(BaseEstimator):
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
    >>> estimator = LSTMDemandPrediction()
    >>> estimator.fit(X_train, y_train)
    LSTMDemandPrediction()
    >>> estimator.predict(X_test)
    [...]
    """

    def __init__(self, breg=0, dropout=0.115, lr=0.005, recurrent='hard_sigmoid', RS=False, f_bias=True, bs=True, decay=0.001):
        """Init function. Create DL_LSTM object with ..... attribute."""
        self.model = None
        self.breg = breg
        self.dropout = dropout
        self.lr = lr
        self.recurrent = recurrent
        self.RS = RS
        self.f_bias = f_bias
        self.bs = bs
        self.decay = decay

    def fit(self, X, y, **kwargs):
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
        # X, y = check_X_y(X, y, accept_sparse=True)

        X_shape = X.shape

        # set default params
        defaultKwargs = { 'epochs': 100, 'batch_size': 72, 'verbose': 0 }
        kwargs = { **defaultKwargs, **kwargs }

        self.model = Sequential()
        self.model.add(LSTM(32,input_shape=(X_shape[1], X_shape[2]),kernel_initializer='glorot_uniform', 
            bias_regularizer = regularizers.l2(self.breg),return_sequences=True,
            recurrent_activation = self.recurrent , return_state = self.RS, unit_forget_bias = self.f_bias, use_bias = self.bs))
        self.model.add(LSTM(16, dropout = self.dropout, return_sequences = False, recurrent_activation = self.recurrent,
            return_state = self.RS,unit_forget_bias = self.f_bias, use_bias = self.bs))
        self.model.add(Dense(1))

        adam = adam_v2.Adam(learning_rate=self.lr, decay=self.decay)
        self.model.compile(loss='mae', optimizer=adam)

        history = self.model.fit(X, y, epochs=kwargs["epochs"], batch_size=kwargs["batch_size"], verbose=kwargs["verbose"], shuffle=False)
        self.is_fitted_ = True
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
        # X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')
        return self.model.predict(X)



