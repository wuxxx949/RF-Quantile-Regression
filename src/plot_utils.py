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
    cnt = 0
    for idx in range(len(y)):
        if y[idx] < lower_bound[idx] or y[idx] > upper_bound[idx]:
            cnt += 1

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
    plt.text(0.01, 0.99, f"pct of pts outside of bound: {cnt / len(y):.2%}", transform=plt.gca().transAxes,
             verticalalignment='top', horizontalalignment='left', fontsize=12)
    plt.xlabel("Ordered Samples")
    plt.ylabel("Centered Values and Prediction Intervals")
    plt.legend(loc='best')
    plt.grid(True, color='gray', alpha=0.1)
    plt.show()


def pred_vs_actual(actual, prediction):
    """
    Plot predicted vs actual values.

    Parameters:
      actual : array-like
          The observed values.
      prediction : array-like
    """
   # plot median prediction vs actual
    min_val = min(min(prediction), min(actual))
    max_val = max(max(prediction), max(actual))
    plt.figure(figsize=(8, 6))
    plt.scatter(prediction, actual, color='orange', s=15, label='Observed')
    # add 45 degree line
    plt.plot([min_val, max_val], [min_val, max_val], color='blue', linestyle='--')
    plt.xlabel("Predicted Values")
    plt.ylabel("Actual Values")
    plt.title("Predicted vs Actual Values")
    plt.show()