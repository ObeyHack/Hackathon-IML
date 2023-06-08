import numpy as np


def F1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate F1 score

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values

    Returns
    -------
    F1 Score of given predictions
    """
    true_predictions = y_true[y_true == y_pred]
    false_predictions = y_true[y_true != y_pred]
    TP = float(len(true_predictions[true_predictions == 1]))
    FP = float(len(false_predictions[false_predictions == 1]))
    FN = float(len(false_predictions[false_predictions == 0]))
    return TP / (TP + 0.5 * (FP + FN))
