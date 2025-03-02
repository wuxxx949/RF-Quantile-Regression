import numpy as np

def qantile_loss(y_true: np.array, y_pred: np.array, quantile: float) -> float:
    """
    Calculate the quantile loss for a given quantile.

    Parameters:
    y_true (np.array): True values.
    y_pred (np.array): Predicted values.
    quantile (float): The quantile to calculate the loss for.

    Returns:
    float: The quantile loss.
    """
    error = np.abs(y_true - y_pred)
    coefficient = np.array([quantile] * len(y_true))
    coefficient[y_true <= y_pred] = 1 - quantile
    return np.mean(coefficient * error)


def mape(y_true: np.array, y_pred: np.array) -> float:
    """
    Calculate the mean absolute percentage error (MAPE).

    Parameters:
    y_true (np.array): True values.
    y_pred (np.array): Predicted values.

    Returns:
    float: The MAPE.
    """
    return np.mean(np.abs((y_true - y_pred) / y_true))


def mse(y_true: np.array, y_pred: np.array):

    """
    Calculate the mean squared error (MSE).
    Parameters:
    y_true (np.array): True values.
    y_pred (np.array): Predicted values.
    Returns:
    float: The MSE.
    """
    return np.mean((y_true - y_pred) ** 2)


if __name__ == "__main__":
    y_true = np.array([0.1, 4, 3, 4, 5])
    y_pred = np.array([1, 2, 3, 4, 5])
    quantile = 0.9
    print(qantile_loss(y_true, y_pred, quantile))