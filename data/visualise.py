import numpy as np
import matplotlib.pyplot as plt

def plot_data(X, y, predictions: list[np.ndarray] = []):
    X = np.array(X).flatten()
    
    # Scatter plot of data points
    plt.figure(figsize=(8, 6))
    plt.scatter(X, y, color='blue', alpha=0.7, label="Data")

    # Drawing regression lines
    if len(predictions) > 0:
        if type(predictions) == np.ndarray:
            plt.plot(X, predictions, color='green', label='Fitted line')

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