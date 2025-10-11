import numpy as np

def evaluate_rmse(Y: np.ndarray, predictions: np.ndarray) -> float:
    ''''
    This function evaluates the root square mean error of regression predictions.
    '''

    rmse = (1/len(Y) * np.sum((Y - predictions)**2))**0.5

    return rmse