import numpy as np
import pandas as pd

def generate_linear_data(n_samples=100, noise=5, a=1, b=0, random_state=42):
    '''y = ax + b + noise'''
    np.random.seed(random_state)
    X = np.linspace(0, 10, n_samples)
    y = a * X + b + np.random.randn(n_samples) * noise
    return X, y

def generate_quadratic_data(n_samples=100, noise=10, a=1, b=0, c=0, random_state=42):
    '''y = ax^2 + bx + c + noise'''
    np.random.seed(random_state)
    X = np.linspace(0, 10, n_samples)
    y = a * (X ** 2) + b * X + c + np.random.randn(n_samples) * noise
    return X, y

def generate_cubic_data(n_samples=100, noise=15, a=1, b=0, c=0, d=0, random_state=42):
    '''y = ax^3 + bx^2 + cx + d + noise'''
    np.random.seed(random_state)
    X = np.linspace(0, 10, n_samples)
    y = a * (X ** 3) + b * (X ** 2) + c * X + d + np.random.randn(n_samples) * noise
    return X, y
