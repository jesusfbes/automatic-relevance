import numpy as np

from sklearn.model_selection import KFold
import GPy


def gp_ard(X, y, n_folds=10,standarize=True):
    """ Function that computes the relevances of some data for some target"""

    if standarize:
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        y = (y - np.mean(y)) / np.std(y)

    kf = KFold(n_splits=n_folds)
    mean_sq_error = []

    n_vars = X.shape[1]

    ard_values = np.zeros((1, n_vars))

    sign_values = np.zeros((1, n_vars))

    for train, test in kf.split(y):
        (X_train, X_test, y_train, y_test) = (X[train],  X[test], y[train], y[test])
        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)

        kernel1 = GPy.kern.Linear(input_dim=n_vars, variances=[1]*n_vars, ARD=True)

        m = GPy.models.GPRegression(X_train, y_train, kernel1)

        m.optimize(messages=False)

        ard_values = ard_values + m.parameters[0].variances

        # Compute sign values
        sign_values = sign_values + \
            np.sign(np.dot(np.dot(np.linalg.pinv(m.kern.K(X_train, X_train)), X_train).T, y_train).T)

        (y_predict, dummy) = m.predict(X_test)

        mean_sq_error.append(np.mean((y_predict-y_test)**2))

    #print("MSE_1=" + str((mean_sq_error)))
    ard_values = ard_values/n_folds

    return ard_values, sign_values, mean_sq_error
