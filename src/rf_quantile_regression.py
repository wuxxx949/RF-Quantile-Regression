from rfgap import RFGAP

import numpy as np
import matplotlib.pyplot as plt


def weighted_quantile_rfgap(values, quantiles, sample_weight=None, values_sorted=False):
    """
    Compute weighted quantiles

    Parameters:
      values : array-like
          The data values.
      quantiles : float or array-like
          Desired quantile(s) in the range [0, 1].
      sample_weight : array-like, optional
          Weights for each data point. If None, each data point is given equal weight.
      values_sorted : bool, optional
          If True, assumes that `values` is already sorted.

    Returns:
      A single quantile value or an array of quantile values, each taken directly from the data.
    """
    values = np.array(values)

    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    quantiles = np.atleast_1d(quantiles)

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    # Compute cumulative weights and normalize to range [0, 1]
    cum_weights = np.cumsum(sample_weight)
    total_weight = cum_weights[-1]
    cum_weights_norm = cum_weights / total_weight

    # For each quantile, select the first data point where the cumulative weight exceeds the quantile
    quantile_values = []
    for q in quantiles:
        idx = np.searchsorted(cum_weights_norm, q, side='left')
        quantile_values.append(values[idx])

    # Return a scalar if only one quantile was provided, otherwise an array.
    return quantile_values[0] if len(quantile_values) == 1 else np.array(quantile_values)


def predit_bounds(x: np.array, y: np.array, coverage: float, **kwargs):
    """Predict bounds using RFGAP

    Args:
        x (np.array): array-like, shape (n_samples, n_features)
        y (np.array): array-like, shape (n_samples,)
        coverage (float): coverage level for the prediction interval
        **kwargs: additional arguments for RFGAP

    Returns:
        upper_bound (np.array): array-like, shape (n_samples,)
        lower_bound (np.array): array-like, shape (n_samples,)
        rf (RFGAP): fitted RFGAP model
    """
    if coverage <= 0 or coverage >= 1:
        raise ValueError("coverage must be between 0 and 1")
    rf = RFGAP(
        prediction_type="regression",
        **kwargs
        )
    rf.fit(x, y)

    proximities = rf.get_proximities()

    upper_bounds = np.zeros(len(y))
    lower_bounds = np.zeros(len(y))
    medians = np.zeros(len(y))

    for idx in range(len(y)):
        proximity = proximities[idx].toarray()[0]
        upper = weighted_quantile_rfgap(
            values=y,
            quantiles=1 - (1 - coverage) / 2,
            sample_weight=proximity,
            values_sorted=False
            )
        lower = weighted_quantile_rfgap(
            values=y,
            quantiles=(1 - coverage) / 2,
            sample_weight=proximity
            )
        median = weighted_quantile_rfgap(
            values=y,
            quantiles=0.5,
            sample_weight=proximity
            )
        upper_bounds[idx] = upper
        lower_bounds[idx] = lower
        medians[idx] = median

    return upper_bounds, medians, lower_bounds, rf


if __name__ == "__main__":
    import openml
    boston = openml.datasets.get_dataset(531)  # 531 is the dataset ID
    prediction_type = 'regression'

    # Convert to a Pandas DataFrame
    x, y, _, _ = boston.get_data(target=boston.default_target_attribute)


    rf = RFGAP(prediction_type=prediction_type,
               n_estimators=500,
               max_depth=6,
               min_samples_split=5
               )
    rf.fit(x, y)

    proximities = rf.get_proximities()


    median_prediction = np.zeros(len(y))
    for idx in range(len(y)):
        proximity = proximities[idx].toarray()[0]
        median = weighted_quantile_rfgap(
            values=y,
            quantiles=0.5,
            sample_weight=proximity
            )
        median_prediction[idx] = median

    sorter = np.argsort(median_prediction)

    # plot median prediction vs actual
    plt.figure(figsize=(8, 6))
    plt.scatter(median_prediction[sorter], y[sorter], color='orange', s=15, label='Observed')

    # add 45 degree line
    plt.plot([0, 50], [0, 50], color='blue')
