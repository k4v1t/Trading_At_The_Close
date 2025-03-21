import numpy as np
import pandas as pd
import sklearn.linear_model as linear_model
import sklearn.metrics as metrics

class RegressionModel:

    def __init__(self, train_x, train_y, test_x, test_y):
        
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.reg = None
    
    def fit_model(self, model_name, reg_alpha = 1.0, reg_l1_ratio = 0.5):

        if model_name == 'linear_regression':
            self.reg = linear_model.LinearRegression()
        elif model_name == 'ridge':
            self.reg = linear_model.Ridge(alpha=reg_alpha)
        elif model_name == 'lasso':
            self.reg = linear_model.Lasso(alpha=reg_alpha)
        elif model_name == 'elastic_net':
            self.reg = linear_model.ElasticNet(alpha=reg_alpha, l1_ratio=reg_l1_ratio)
        else:
            return 'Invalid model name! Please enter a valid model.'
        
        self.reg.fit(self.train_x, self.train_y)

    def predict_model(self, train_or_test):

        if train_or_test == 'train':
            x_var = self.train_x
        elif train_or_test == 'test':
            x_var = self.test_x
        else:
            return 'Invalid data for prediction! Please fix and re-run.'
        
        return self.reg.predict(x_var)

    
    def eval_fit(self, pred_y, metric_name, train_or_test):
        
        if train_or_test == 'train':
            y_var = self.train_y
        elif train_or_test == 'test':
            y_var = self.test_y
        else:
            return 'Invalid data for prediction! Please fix and re-run.'

        if metric_name == 'mae':
            return metrics.mean_absolute_error(y_var, pred_y)
        elif metric_name == 'mse':
            return metrics.mean_squared_error(y_var, pred_y)
        elif metric_name == 'rmse':
            return metrics.root_mean_squared_error(y_var, pred_y)
        elif metric_name == 'r2':
            return metrics.r2_score(y_var, pred_y)
        elif metric_name == 'exp_var':
            return metrics.explained_variance_score(y_var, pred_y)
        elif metric_name == 'corr':
            corr_matrix = np.corrcoef(y_var, pred_y)
            return corr_matrix[0, 1]
        elif metric_name == 'all':
            mae = metrics.mean_absolute_error(y_var, pred_y)
            mse = metrics.mean_squared_error(y_var, pred_y)
            rmse = metrics.root_mean_squared_error(y_var, pred_y)
            r2 = metrics.r2_score(y_var, pred_y)
            exp_var = metrics.explained_variance_score(y_var, pred_y)
            corr_matrix = np.corrcoef(y_var, pred_y)
            corr_coef = corr_matrix[0, 1]
            metric_list = {'mae': mae, 'mse': mse, 'rmse': rmse, 'r2': r2, 'exp_var': exp_var, 'corr_coef': corr_coef}
            key = metric_list.keys()
            val = metric_list.values()
            return list(zip(key, val))
        else:
            return 'Invalid evaluation metric! Please enter valid metric.'