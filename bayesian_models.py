# this file include the gaussian process regression model and random forest model for the bayesian optimization
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

class bayesian_models():
    # this is the set of models which are used in the bayesian optimization
    def __init__(self) -> None:
        pass

    def random_forest(self, X, y):
        # this is the random forest model which is used in the bayesian optimization
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        return rf

    def gaussian_process(self, X, y):
        # this is the gaussian process model which is used in the bayesian optimization
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
        gp.fit(X_train, y_train)
        return gp