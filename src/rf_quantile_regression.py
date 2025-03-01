from rfgap import RFGAP

import numpy as np
import matplotlib.pyplot as plt

def weighted_quantile_actual_point(values, quantiles, sample_weight=None, values_sorted=False):
    """
    Compute weighted quantiles using actual data points (no interpolation).

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

    # # Example usage:
    # data = np.array([10, 20, 30, 40, 50])
    # weights = np.array([1, 2, 3, 4, 5])

    # # Compute the weighted median using an actual data point
    # median_actual = weighted_quantile_actual_point(data, 0.5, sample_weight=weights)
    # print("Weighted median (actual data point):", median_actual)

    # # Compute an arbitrary quantile, say 0.3 quantile
    # quantile_30_actual = weighted_quantile_actual_point(data, 0.3, sample_weight=weights)
    # print("Weighted 0.3 quantile (actual data point):", quantile_30_actual)
    y = y.to_numpy()
    # sorter = np.argsort(y)
    upper_bound = np.zeros(len(y))
    lower_bound = np.zeros(len(y))
    y_centered = np.zeros(len(y))

    for idx in range(len(y)):
        proximity = proximities[idx].toarray()[0]
        upper = weighted_quantile_actual_point(
            values=y,
            quantiles=0.975,
            sample_weight=proximity
            )
        lower = weighted_quantile_actual_point(
            values=y,
            quantiles=0.025,
            sample_weight=proximity
            )
        mean_bound = (upper + lower) / 2
        upper_bound[idx] = upper - mean_bound
        lower_bound[idx] = lower - mean_bound
        y_centered[idx] = y[idx] - mean_bound

    # Sort by the length of the bounds
    bound_len = upper_bound - lower_bound
    sorter = np.argsort(bound_len)
    y_centered = y_centered[sorter]

    plt.figure(figsize=(8, 6))

    # Scatter the observed values (orange points)
    plt.scatter(range(len(y)), y_centered, color='orange', s=15)

    # Plot the upper and lower bounds (blue lines)
    plt.plot(range(len(y)), lower_bound[sorter], color='blue')
    plt.plot(range(len(y)), upper_bound[sorter], color='blue')

    # Fill between the bounds (light blue shading)
    plt.fill_between(range(len(y)), lower_bound[sorter], upper_bound[sorter], color='blue', alpha=0.1)

    # Label axes, title, etc.
    plt.xlabel("Ordered Samples")
    plt.ylabel("Observed Values and Prediction Intervals")
    plt.legend(loc='best')
    plt.grid(True, color='gray', alpha=0.1)
    plt.show()

    median_prediction = np.zeros(len(y))
    for idx in range(len(y)):
        proximity = proximities[idx].toarray()[0]
        median = weighted_quantile_actual_point(
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


    cnt = 0
    for idx in range(len(y)):
        if y[idx] < lower_bound[idx] or y[idx] > upper_bound[idx]:
            cnt += 1

    print(cnt / len(y))


    upper_bound = np.zeros(len(y))
    lower_bound = np.zeros(len(y))
    y_lst = y

    for idx in range(len(y)):
        proximity = proximities[idx].toarray()[0]
        upper = weighted_quantile_actual_point(
            values=y,
            quantiles=0.975,
            sample_weight=proximity
            )
        lower = weighted_quantile_actual_point(
            values=y,
            quantiles=0.025,
            sample_weight=proximity
            )
        upper_bound[idx] = upper
        lower_bound[idx] = lower

    y_centered = np.zeros(len(y))
    upper_bound_centered = np.zeros(len(y))
    lower_bound_centered = np.zeros(len(y))


    plot_centered_bounds(y, upper_bound, lower_bound)