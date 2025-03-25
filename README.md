# Trading_At_The_Close
Optiver Kaggle Challenge

The code in this repository attempts to predict the stock prices within the NASDAQ index using the data provided by Optiver. The competition overview and the data provided can be found on Kaggle:
https://www.kaggle.com/competitions/optiver-trading-at-the-close/overview

There are three notebooks in this repository which are described below.

(1) Index_Estimation_PCA.ipynb - This notebook has the code used to estimate the individual stock weights of the index using a PCA approach on the covariance matrix of the individual stock returns. It also then estimates the index returns  based on the reference price, weighted-average price and mid price.
(2) Feature_Generation_v2.ipynb - This notebook takes in the index returns and then estimates the active stock returns per price type (reference, wap or mid). Using this, along with some other metrics to do with the imbalances in the auction book and order book, it also calculates the exponentially-weighted moving averages (EWMAs). These EWMAs are calculated for a range of half-lives to be used in the regression model.
(3) Model_Fit_And_Predict_v2.ipynb - Using the features generated in the file above, this notebook then uses linear regression, ridge regression and lasso regression to predict the stock prices in the test dataset. The alpha hyperparameter for ridge and lasso regression is tuned using a 5-fold cross validation approach. The linear regression model seems to work best, providing a test MAE of 5.24 compared to the baseline MAE of 5.27.

Other models to try at some point:
(1) Elastic Net Regression
(2) LightGBM
(3) XGBoost
