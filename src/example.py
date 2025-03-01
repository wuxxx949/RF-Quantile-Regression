import numpy as np
from rfgap import RFGAP

from plot_utils import plot_centered_bounds
from rf_quantile_regression import weighted_quantile_rfgap

if __name__ == "__main__":
    import openml
    boston = openml.datasets.get_dataset(531)  # 531 is the dataset ID
    prediction_type = 'regression'

    # Convert to a Pandas DataFrame
    x, y, _, _ = boston.get_data(target=boston.default_target_attribute)
    rf = RFGAP(prediction_type=prediction_type,
               n_estimators=500,
               max_depth=6,
               )
    rf.fit(x, y)

    proximities = rf.get_proximities()

    upper_bound = np.zeros(len(y))
    lower_bound = np.zeros(len(y))

    for idx in range(len(y)):
        proximity = proximities[idx].toarray()[0]
        upper = weighted_quantile_rfgap(
            values=y,
            quantiles=0.975,
            sample_weight=proximity,
            values_sorted=False
            )
        lower = weighted_quantile_rfgap(
            values=y,
            quantiles=0.025,
            sample_weight=proximity
            )
        upper_bound[idx] = upper
        lower_bound[idx] = lower

    plot_centered_bounds(y, upper_bound, lower_bound)


    cnt = 0
    for idx in range(len(y)):
        if y[idx] < lower_bound[idx] or y[idx] > upper_bound[idx]:
            cnt += 1

    print(f"pct of pts outside of bound: {cnt / len(y)}")
