import matplotlib.pyplot as plt
import numpy as np


def plot_centered_bounds(y, upper_bound, lower_bound):
    """
    Plot centered bounds and observed values.

    Parameters:
      y : array-like
          The observed values.
      upper_bound : array-like
          The upper bounds.
      lower_bound : array-like
          The lower bounds.
    """
    y_centered = y - (upper_bound + lower_bound) / 2
    upper_bound_centered = upper_bound - (upper_bound + lower_bound) / 2
    lower_bound_centered = lower_bound - (upper_bound + lower_bound) / 2
    # Sort by the length of the bounds
    bound_len = upper_bound - lower_bound
    sorter = np.argsort(bound_len)
    y_centered = y_centered[sorter]
    upper_bound = upper_bound_centered[sorter]
    lower_bound = lower_bound_centered[sorter]

    plt.figure(figsize=(8, 6))
    plt.scatter(range(len(y)), y_centered, color='orange', s=15)
    plt.plot(range(len(y)), lower_bound, color='blue')
    plt.plot(range(len(y)), upper_bound, color='blue')
    plt.fill_between(range(len(y)), lower_bound, upper_bound, color='blue', alpha=0.1)
    plt.xlabel("Ordered Samples")
    plt.ylabel("Centered Values and Prediction Intervals")
    plt.legend(loc='best')
    plt.grid(True, color='gray', alpha=0.1)
    plt.show()

