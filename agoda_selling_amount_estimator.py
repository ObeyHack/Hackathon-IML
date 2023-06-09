from __future__ import annotations
from typing import NoReturn

import sklearn.ensemble
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, log_loss
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.datasets import make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from agoda_cancellation_estimator import AgodaCancellationEstimator
from base_estimator import BaseEstimator
import numpy as np


class AgodaSellingAmountEstimator(BaseEstimator):
    """
    An estimator for solving the Agoda Cancellation challenge
    """

    def __init__(self) -> AgodaSellingAmountEstimator:
        """
        Instantiate an estimator for solving the Agoda Cancellation challenge

        Parameters
        ----------


        Attributes
        ----------

        """
        super().__init__()
        self.model = None
        self.cancel_estimator = None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit an estimator for given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Notes
        -----

        """
        self.cancel_estimator = AgodaCancellationEstimator().fit(X, y.apply(lambda x: 1 if x >= 0 else 0))
        self.model = LinearRegression().fit(X, y)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """

        y_classify = self.cancel_estimator.predict(X)
        y_pred = self.model.predict(X)
        y_pred[y_classify == 0] = -1
        return y_pred


    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under loss function
        """

        return np.sqrt(mean_squared_error(y, self.predict(X)))
