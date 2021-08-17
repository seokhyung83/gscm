import os, random, time
import xgboost
import datetime
import pygam
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#from fbprophet import Prophet
from sklearn.linear_model import LinearRegression as LR
from sklearn.metrics import r2_score
from sklearn.gaussian_process import GaussianProcessRegressor as gpr
from sklearn.gaussian_process.kernels import WhiteKernel, ConstantKernel as C, RBF



y_inven = pd.read_csv('./data/Y_Inven.csv')
x_sales = pd.read_csv('./data/X_Sales.csv')
x_product = pd.read_csv('./data/X_Product.csv')


def dataset(x_dta, case=1, is_train=True):
    x_prev_col = [s for s in x_dta.columns.values if 'Prev' in s]
    x_post_col = [s for s in x_dta.columns.values if 'Post' in s]
    x_var_col = [s for s in x_dta.columns.values if 'Var' in s]

    X_prev_ = x_dta[x_prev_col]
    X_post_ = x_dta[x_post_col]
    X_var_ = x_dta[x_var_col]
    if case ==1 :
        Y_ =  x_dta['Sales']
    else :
        Y_ =  x_dta['Products']

    if is_train :
        X_prev_train = X_prev_[x_dta['YEAR']==2020]
        X_post_train = X_post_[x_dta['YEAR']==2020]
        X_var_train = X_var_[x_dta['YEAR']==2020]
        Y_train = Y_[x_dta['YEAR']==2020]
    else:
        X_prev_train = X_prev_[x_dta['YEAR']==2021]
        X_post_train = X_post_[x_dta['YEAR']==2021]
        X_var_train = X_var_[x_dta['YEAR']==2021]
        Y_train = Y_[x_dta['YEAR']==2021]
    
    return Y_train, X_prev_train, X_post_train, X_var_train    




model_sales=xgboost.XGBRegressor(n_estimators=50, learning_rate=0.08, gamma=0, subsample=0.55, colsample_bytree=1, max_depth=5)#, tree_method='gpu_hist', gpu_id=0)
model_products=xgboost.XGBRegressor(n_estimators=50, learning_rate=0.08, gamma=0, subsample=0.55, colsample_bytree=1, max_depth=5)#, tree_method='gpu_hist', gpu_id=0)


Y_train, prev_X_train, post_X_train, var_X_train = dataset(x_sales, 1)

model_sales.fit(var_X_train, Y_train)
sales_prev = model_sales.predict(var_X_train)

Y_test, _, _, var_X_test = dataset(x_sales, 1, False)
sales_hat = model_sales.predict(var_X_test)
Y_sales_hat = np.concatenate((sales_prev, sales_hat))
Y_sales = np.concatenate((Y_train, Y_test))



Y_train, prev_X_train, post_X_train, var_X_train = dataset(x_product,2 )

model_products.fit(var_X_train, Y_train)
products_prev = model_products.predict(var_X_train)

Y_test, _, _, var_X_test = dataset(x_product, 2, False)
products_hat = model_products.predict(var_X_test)
Y_products_hat = np.concatenate((products_prev, products_hat))
Y_products = np.concatenate((Y_train, Y_test))


plt.figure(figsize=(30,20))
plt.subplot(2,1,1)
plt.title("Sales Trend & Prediction")
plt.plot(Y_sales, color='red', marker='o')
plt.plot(Y_sales_hat, color='blue', linestyle='dashed', marker='.')
plt.axvline(43, c='k')
plt.subplot(2,1,2)
plt.title("Products Trend & Prediction")
plt.plot(Y_products, color='red', marker='o')
plt.plot(Y_products_hat, color='blue', linestyle='dashed', marker='.')
plt.axvline(43, c='k')
plt.savefig('./result1.png')














# %%
