import numpy as np
from rfgap import RFGAP

from plot_utils import plot_centered_bounds
from rf_quantile_regression import predit_bounds, weighted_quantile_rfgap

if __name__ == "__main__":
    import openml
    boston = openml.datasets.get_dataset(531)  # 531 is the dataset ID
    prediction_type = 'regression'

    # Convert to a Pandas DataFrame
    x, y, _, _ = boston.get_data(target=boston.default_target_attribute)

    upper_bound, lower_bound, rf = predit_bounds(
        x, y, coverage=0.95, n_estimators=500, max_depth=6
        )

    plot_centered_bounds(y, upper_bound, lower_bound)
