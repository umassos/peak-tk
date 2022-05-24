
import numpy as np

from keras import regularizers
from keras.optimizers import adam_v2
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class LSTMPeakHourModel(BaseEstimator):
    """ Peak Hours of the Day Prediction - Long Short Term Memory

    A class to train an LSTM model and use the model to predict N peak hours of the day.

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
    def __init__(self, num_peak_hours, breg=0.002, dropout=0.4, lr=0.0006, recurrent='sigmoid', RS=False, f_bias=True, bs=True, decay=0.0005, kernel='glorot_uniform'):
        self.model = None
        self.history = None
        self.pred_demand = None
        self.breg = breg
        self.dropout = dropout
        self.lr = lr
        self.recurrent = recurrent
        self.RS = RS
        self.f_bias = f_bias
        self.bs = bs
        self.decay = decay
        self.kernel = kernel
        self.num_peak_hours = num_peak_hours

    def create_model(self, X_shape):
        # LSTM Model (Amee version)
        # Amee's

        if len(X_shape) != 3:
            print("Error: Input shape has to be in 3D array")
            return
            
        # kernel = 'glorot_uniform'
        # breg = 0.002
        # dropout = 0.4
        # lr = 0.0006
        # recurrent = 'sigmoid'
        # RS = False
        # f_bias = True
        # bs = True
        # decay = 0.0005

        model = Sequential()
        model.add(LSTM(100,input_shape=(X_shape[1], X_shape[2]),kernel_initializer=self.kernel
                       , bias_regularizer = regularizers.l2(self.breg),return_sequences=True,recurrent_activation = self.recurrent
                       , return_state = self.RS,unit_forget_bias = self.f_bias, use_bias = self.bs))
        model.add(LSTM(90, dropout = self.dropout, return_sequences = True,recurrent_activation = self.recurrent
                       ,return_state = self.RS,unit_forget_bias = self.f_bias, use_bias = self.bs))
        model.add(LSTM(80, dropout = self.dropout, return_sequences = True,recurrent_activation = self.recurrent
                       ,return_state = self.RS,unit_forget_bias = self.f_bias, use_bias = self.bs))
        model.add(LSTM(70,recurrent_activation = self.recurrent, use_bias = self.bs))
        model.add(Dense(24))

        optimizer = adam_v2.Adam(self.lr, self.decay)
        model.compile(optimizer, loss='mae')
    
        return model


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

        # set default params
        defaultKwargs = { 'epochs': 100, 'batch_size': 72, 'verbose': 0 }
        kwargs = { **defaultKwargs, **kwargs }

        # X, y = check_X_y(X, y, accept_sparse=True)
        self.is_fitted_ = True

        self.model = self.create_model(X.shape)
        history = self.model.fit(X, y, epochs=kwargs["epochs"], batch_size=kwargs["batch_size"], verbose=kwargs["verbose"], shuffle=False)
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
        self.pred_demand = self.model.predict(X)
        result_list = []
        for d in self.pred_demand:
            # find peak
            idx = np.argpartition(d, -self.num_peak_hours)[-self.num_peak_hours:]
            result = np.array([False]*24)
            result[idx] = True
            result_list.append(result)
        return np.array(result_list)



