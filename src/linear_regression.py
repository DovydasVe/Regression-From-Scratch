import numpy as np
import numpy.linalg as la


def find_parameters(X: np.ndarray, y: np.ndarray):
    try:
        theta = 1 / float(np.dot(X, X)) * np.dot(X, y)
    except:
        print('Error')
        
    return theta


def generate_linear_data(n_samples=100, noise=5, slope=1, intercept=0, random_state=42):
    '''y = ax + b + noise'''
    np.random.seed(random_state)
    X = np.linspace(0, 10, n_samples)
    y = slope * X + intercept + np.random.randn(n_samples) * noise
    return X, y

X, y = generate_linear_data(n_samples=200, noise=5)
print(find_parameters(X, y))