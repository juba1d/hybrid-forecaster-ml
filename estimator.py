import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class HybridForecaster(BaseEstimator, RegressorMixin):
    # hybrid model
    def __init__(self, lgbm_params=None):
        # default params
        self.lgbm_params = lgbm_params if lgbm_params else {'n_estimators': 100, 'learning_rate': 0.1}
        self.trend_model_ = LinearRegression()
        self.residual_model_ = LGBMRegressor(**self.lgbm_params)

    def fit(self, X, y):
        # data check
        X, y = check_X_y(X, y)
        
        # trend ber koro
        self.trend_model_.fit(X, y)
        trend_line = self.trend_model_.predict(X)
        
        # baki part
        leftover = y - trend_line
        
        # residual shikhbo
        self.residual_model_.fit(X, leftover)
        
        self.is_fitted_ = True
        return self

    def predict(self, X):
        # shob thik?
        check_is_fitted(self)
        X = check_array(X)
        
        # result jog koro
        p1 = self.trend_model_.predict(X)
        p2 = self.residual_model_.predict(X)
        
        return p1 + p2