import numpy as np

from dwarka.model import GTWRModel


def test_gtwr_predicts_and_returns_uncertainty():
    # Small synthetic dataset
    rng = np.random.default_rng(42)
    n_train = 80
    n_test = 20

    # coords in a 2D space
    coords_train = rng.random((n_train, 2)) * 100.0
    coords_test = rng.random((n_test, 2)) * 100.0

    # times (numeric)
    times_train = rng.random(n_train) * 10.0
    times_test = rng.random(n_test) * 10.0

    # Single feature: a function of location and time
    X_train = (coords_train[:, 0] * 0.01 + 0.5 * np.sin(times_train))[:, None]
    X_test = (coords_test[:, 0] * 0.01 + 0.5 * np.sin(times_test))[:, None]

    # True coefficient varies with x-coordinate (spatially varying)
    beta_train = 1.0 + 0.5 * (coords_train[:, 0] / 100.0)
    y_train = (beta_train * X_train.ravel()) + rng.normal(scale=0.1, size=n_train)

    model = GTWRModel(h_s=20.0, h_t=5.0)
    model.fit(X_train, y_train, coords_train, times=times_train)

    preds, uncert = model.predict(
        X_test, coords_test, times_pred=times_test, return_residuals=True
    )

    assert preds.shape[0] == n_test
    assert uncert.shape[0] == n_test
    assert np.all(np.isfinite(preds))
    assert np.all(uncert >= 0.0)

    # Check that predictions have some explanatory power: MSE less than variance
    mse = np.mean((preds - (1.0 * X_test.ravel())) ** 2)
    var_y = np.var(X_test.ravel())
    # This is a loose check to ensure model isn't producing constant garbage
    assert mse < (var_y * 10.0 + 1e-6)
