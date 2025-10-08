import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt


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

def evaluate_rmse(Y: np.ndarray, predictions: np.ndarray) -> float:
    ''''
    This function evaluates the root square mean error of regression predictions.
    '''

    rmse = (1/len(Y) * np.sum((Y - predictions)**2))**0.5

    return rmse


def generate_linear_data(n_samples=100, noise=5, slope=1, intercept=0, random_state=42):
    '''y = ax + b + noise'''
    np.random.seed(random_state)
    X = np.linspace(0, 10, n_samples)
    y = slope * X + intercept + np.random.randn(n_samples) * noise
    return X, y


def plot_data(X, y, predictions: list[np.ndarray] = np.ndarray([])):

    
    X = np.array(X).flatten()
    
    # Scatter plot of data points
    plt.figure(figsize=(8, 6))
    plt.scatter(X, y, color='blue', alpha=0.7, label="Data")

    if len(predictions) > 0:
        if type(predictions) == np.ndarray:
            plt.plot(X, predictions, color='red', label='Fitted line')

        else:
            for pred in predictions:
                plt.plot(X, pred, color='red', label='Fitted line')
    
    # Styling
    plt.xlabel("X")
    plt.ylabel("y")
    plt.title("Synthetic Linear Data")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

X, y = generate_linear_data(n_samples=200, noise=5)

theta = find_theta_linear(X, y)
predictions = predict(X, theta)
print(theta)

theta_extended = find_theta_multinomial(X, y, 1)
predictions_extended = predict(X, theta_extended)
print(theta_extended)

plot_data(X, y, [predictions, predictions_extended])


rmse = evaluate_rmse(y, predictions)
rmse_extended = evaluate_rmse(y, predictions_extended)
print(rmse, rmse_extended)

# variance = find_variance(X, y, theta)
# print(variance)