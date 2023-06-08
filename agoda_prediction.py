import Preprocess_data
from base_estimator import BaseEstimator
from agoda_cancellation_estimator import AgodaCancellationEstimator
from utils import split_train_test
import numpy as np
import pandas as pd


def load_data(filename: str, is_cancellation=True):
    data = pd.read_csv(filename)
    if "Test" in filename or "test" in filename:
        if is_cancellation:
            X_test, _, h_booking_id = Preprocess_data.preprocess_for_cancellation(data, None)
        else:
            X_test, _, h_booking_id = Preprocess_data.preprocess_for_cost(data, None)
        return X_test, None, h_booking_id

    if is_cancellation:
        X = data.drop('cancellation_datetime', axis=1)
        y = data['cancellation_datetime']
        X_train, y_train, _ = Preprocess_data.preprocess_for_cancellation(X, y)

    else:
        X = data.drop('original_selling_amount', axis=1)
        y = data['original_selling_amount']
        X_train, y_train, _ = Preprocess_data.preprocess_for_cost(X, y)
    return X_train, pd.Series(y_train), None


def evaluate_and_export(estimator: BaseEstimator,  X: np.ndarray, h_booking_id_save: np.ndarray, filename: str,
                        is_cancellation=True):
    """
    Export to specified file the prediction results of given estimator on given testset.

    File saved is in csv format with a single column named 'predicted_values' and n_samples rows containing
    predicted values.

    Parameters
    ----------
    estimator: BaseEstimator or any object implementing predict() method as in BaseEstimator (for example sklearn)
        Fitted estimator to use for prediction

    X: ndarray of shape (n_samples, n_features)
        Test design matrix to predict its responses

    filename:
        path to store file at

    """
    name = "cancellation" if is_cancellation else "predicted_selling_amount"
    pd.DataFrame({"ID": h_booking_id_save, name: estimator.predict(X)}).to_csv(filename, index=False)

