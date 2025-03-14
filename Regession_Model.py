import numpy as np
import pandas as pd
import sklearn.linear_model as linear_model
import sklearn.metrics as metrics

class RegressionModel:

    def __init__(self, train_x, train_y, test_x, test_y):
        
        self.train_x = train_x
        self.train_y = train_y.ravel()
        self.test_x = test_x
        self.test_y = test_y.ravel()

    
    def fit_model_and_predict(self, model_name):
        
        if model_name == 'linear_regression':
            reg = linear_model.LinearRegression()
        elif model_name == 'ridge':
            reg = linear_model.Ridge()
        elif model_name == 'lasso':
            reg = linear_model.Lasso()
        elif model_name == 'elastic_net':
            reg = linear_model.ElasticNet()
        else:
            return 'Invalid model name! Please enter a valid model.'
        
        reg.fit(self.train_x, self.train_y)
        return reg.predict(self.test_x)

    
    def eval_fit(self, pred_y, metric_name):
        
        if metric_name == 'mae':
            return metrics.mean_absolute_error(self.test_y, pred_y)
        elif metric_name == 'mse':
            return metrics.mean_squared_error(self.test_y, pred_y)
        elif metric_name == 'rmse':
            return metrics.root_mean_squared_error(self.test_y, pred_y)
        elif metric_name == 'r2':
            return metrics.r2_score(self.test_y, pred_y)
        elif metric_name == 'exp_var':
            return metrics.explained_variance_score(self.test_y, pred_y)
        elif metric_name == 'corr':
            corr_matrix = np.corrcoef(self.test_y, pred_y)
            return corr_matrix[0, 1]
        elif metric_name == 'all':
            mae = metrics.mean_absolute_error(self.test_y, pred_y)
            mse = metrics.mean_squared_error(self.test_y, pred_y)
            rmse = metrics.root_mean_squared_error(self.test_y, pred_y)
            r2 = metrics.r2_score(self.test_y, pred_y)
            exp_var = metrics.explained_variance_score(self.test_y, pred_y)
            corr_matrix = np.corrcoef(self.test_y, pred_y)
            corr_coef = corr_matrix[0, 1]
            metric_list = {'mae': mae, 'mse': mse, 'rmse': rmse, 'r2': r2, 'exp_var': exp_var, 'corr_coef': corr_coef}
            key = metric_list.keys()
            val = metric_list.values()
            return list(zip(key, val))
        else:
            return 'Invalid evaluation metric! Please enter valid metric.'