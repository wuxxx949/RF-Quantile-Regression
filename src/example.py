import numpy as np
from sklearn.linear_model import QuantileRegressor

from plot_utils import plot_centered_bounds
from rf_quantile_regression import predit_bounds

if __name__ == "__main__":
    import openml
    boston = openml.datasets.get_dataset(531)  # 531 is the dataset ID

    # Convert to a Pandas DataFrame
    X, y, _, _ = boston.get_data(target=boston.default_target_attribute)
    X['RAD'] = X['RAD'].astype(int)
    X['CHAS'] = X['CHAS'].astype(int)

    upper_bound, lower_bound, _ = predit_bounds(
        X, y, coverage=0.95, n_estimators=500, max_depth=6
        )

    plot_centered_bounds(y, upper_bound, lower_bound)
    print(f'average length of the bounds rf: {np.mean(upper_bound - lower_bound)}')

    # compare with the sklearn implementation
    qr = QuantileRegressor(quantile=0.025, alpha=0)
    lower_bound_lm = qr.fit(X, y).predict(X.to_numpy())
    qr = QuantileRegressor(quantile=0.975, alpha=0)
    upper_bound_lm = qr.fit(X, y).predict(X)
    plot_centered_bounds(y, upper_bound_lm, lower_bound_lm)
    print(f'average length of the bounds lm: {np.mean(upper_bound_lm - lower_bound_lm)}')

    # compare with other prox_method of original
    upper_bound_orig, lower_bound_orig, _ = predit_bounds(
        X, y, prox_method = 'original', coverage=0.95, n_estimators=500, max_depth=6
        )
    plot_centered_bounds(y, upper_bound_orig, lower_bound_orig)
    print(f'average length of the bounds original prox: {np.mean(upper_bound_orig - lower_bound_orig)}')

    # compare with other prox_method of oob
    upper_bound_oob, lower_bound_oob, _ = predit_bounds(
        X, y, prox_method = 'oob', coverage=0.95, n_estimators=500, max_depth=6
        )

    plot_centered_bounds(y, upper_bound_oob, lower_bound_oob)
    print(f'average length of the bounds oob prox: {np.mean(upper_bound_oob - lower_bound_oob)}')

