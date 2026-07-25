import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from ml.model import compute_model_metrics, inference, train_model


# TODO: implement the first test. Change the function name and input as needed
def test_train_model():
    """
    Test that train_model returns a trained RandomForestClassifier.
    """
    X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    y_train = np.array([0, 0, 1, 1])

    model = train_model(X_train, y_train)

    assert isinstance(model, RandomForestClassifier)


# TODO: implement the second test. Change the function name and input as needed
def test_inference():
    """
    Test that inference returns one prediction for each input row.
    """
    X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    y_train = np.array([0, 0, 1, 1])

    model = train_model(X_train, y_train)

    preds = inference(model, X_train)

    assert len(preds) == len(X_train)


# TODO: implement the third test. Change the function name and input as needed
def test_compute_model_metrics():
    """
    Test that precision, recall, and F1 are valid values.
    """
    y = np.array([0, 1, 1, 0])
    preds = np.array([0, 1, 1, 0])

    precision, recall, fbeta = compute_model_metrics(y, preds)

    assert precision == pytest.approx(1.0)
    assert recall == pytest.approx(1.0)
    assert fbeta == pytest.approx(1.0)
