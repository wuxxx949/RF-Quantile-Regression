from itertools import combinations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble._forest import (_generate_unsampled_indices,
                                      _get_n_samples_bootstrap)


def rf_similarity(
    X: np.array,
    clf: RandomForestClassifier
    ) -> np.array:
    """random forest similarity using out of bag samples

    Args:
        X (np.array): training examples
        clf (RandomForestClassifier): _description_

    Returns:
        np.array: upper triangular matrix counting the frequency of same terminal node
    """
    n_samples = X.shape[0]
    sim_matrix = np.zeros(shape=(n_samples, n_samples))

    n_samples_bootstrap = _get_n_samples_bootstrap(len(y), clf.max_samples)

    for estimator in clf:
        unsampled_indices = _generate_unsampled_indices(
            estimator.random_state,
            n_samples,
            n_samples_bootstrap,
        )
        unsampled_data = X[unsampled_indices, :]
        leaf_indices = estimator.apply(unsampled_data)
        for l in set(leaf_indices):
            same_leaf_indices = unsampled_indices[leaf_indices == l]
            for idx in combinations(same_leaf_indices, 2):
                sim_matrix[idx[0], idx[1]] += 1

    return sim_matrix


if __name__ == '__main__':
    from sklearn.datasets import make_classification
    RANDOM_STATE = 123

    # Generate a binary classification dataset.
    X, y = make_classification(
        n_samples=500,
        n_features=25,
        n_clusters_per_class=1,
        n_informative=15,
        random_state=RANDOM_STATE,
    )

    clf = RandomForestClassifier(
        warm_start=True,
        oob_score=True,
        max_features="sqrt",
        max_samples=300,
        random_state=RANDOM_STATE,
        )

    clf.set_params(n_estimators=200)
    clf.fit(X, y)

    out = rf_similarity(X, clf)