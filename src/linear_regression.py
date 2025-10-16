import numpy as np
import numpy.linalg as la


def find_theta_linear(X: np.ndarray, y: np.ndarray) -> float:
    '''
    Locating the slope (theta parameter) in the most simple case.
    Assumes the line passes through origin.

    Returns only the value of the slope
    '''

    try:
        theta = 1 / float(np.dot(X, X)) * np.dot(X, y)
    except ZeroDivisionError:
        raise ZeroDivisionError('Not allowed for this project.')
        
    return float(theta)


def find_theta_multinomial(X: np.ndarray, y: np.ndarray, N: int) -> np.ndarray:
    '''
    Locating the theta parameters in any degree polynomial.
    The polynomial is not fixed at origin.

    Returns an np.ndarray with parameter values (starting from constant term c_0 up until
    last term c_n of nth degree polynomial)
    '''

    if N <= 0 or type(N) != int:
        return 'Degree 0 is not possible'
    
    else:
        feature_matrix = np.vander(X, N=(N+1), increasing=True)
        theta = la.inv(feature_matrix.T @ feature_matrix) @ feature_matrix.T @ y

    return theta


def find_variance(X: np.ndarray, y: np.ndarray, theta: float) -> float:
    '''
    This function finds variance in ONLY the very simple case of scenarios
    '''

    try:
        variance = np.sum((y - theta * X)**2) / len(X)
    except ZeroDivisionError:
        raise ZeroDivisionError('Not allowed data of size 0.')
    
    return variance


def predict(X: np.ndarray, theta: float | np.ndarray) -> np.ndarray:
    '''
    Prediction algorithm based on parameter values and known X attributes.
    Works in any polynomial case.

    Outputs the prediction vector y.
    '''

    if type(theta) == float:
        predictions = X * theta
    
    else:
        feature_matrix = np.vander(X, N=len(theta), increasing=True)
        predictions = feature_matrix @ theta

    return predictions


# variance = find_variance(X, y, theta)
# print(variance)