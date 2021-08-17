import os, random, time
import xgboost
import datetime
import pygam
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization
from fbprophet import Prophet
from sklearn.linear_model import LinearRegression as LR
from sklearn.metrics import r2_score

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

def run_model_(model_, trX_, trY_, teX_, teY_):

    model_.fit(trX_, trY_)
    hat_prev_ = model_.predict(trX_)
    hat_  = model_.predict(teX_)
    Y_hat_ = np.concatenate((hat_prev_, hat_))
    Y_     = np.concatenate((trY_, teY_))
    real_ = np.mean(1- np.abs(trY_ - hat_prev_) / np.abs(trY_)) * 100
    fcst_ = np.mean(1- np.abs(teY_ - hat_     ) / np.abs(teY_)) * 100    
    return real_, fcst_, Y_hat_, Y_, hat_prev_, hat_

Y_sales_train, sales_prev_X_train, sales_post_X_train, sales_var_X_train = dataset(x_sales, 1)
Y_sales_test , sales_prev_X_test , sales_post_X_test , sales_var_X_test  = dataset(x_sales, 1, False)
Y_product_train, product_prev_X_train, product_post_X_train, product_var_X_train = dataset(x_product, 2)
Y_product_test , product_prev_X_test , product_post_X_test , product_var_X_test  = dataset(x_product, 2, False)
sales_var_col = [s for s in sales_var_X_train.columns.values if 'Var' in s]
product_var_col = [s for s in product_var_X_train.columns.values if 'Var' in s]

param_bound = {'alpha' : (0.9,0.99) , 'm_n_esitmator' : (10, 100), 'm_lr' : (0.01, 0.5), 'm_subsample' : (0.3, 0.9), 'm_max_depth' : (2,10), 'col_k' : (1,8)}
def product_opt(alpha, m_n_esitmator, m_lr, m_subsample, m_max_depth, col_k):
    weight_mat = list(map(lambda x : alpha**x if x > 0 else 1, range(0,8)))
    
    product_var_X_train1 = np.multiply(product_var_X_train, np.tile([weight_mat], product_var_X_train.shape[0]).reshape(product_var_X_train.shape[0], -1)).copy()
    product_var_X_test1 = np.multiply(product_var_X_test, np.tile([weight_mat], product_var_X_test.shape[0]).reshape(product_var_X_test.shape[0], -1)).copy()
    model_product=xgboost.XGBRegressor(n_estimators=round(m_n_esitmator), learning_rate=m_lr, gamma=0, subsample=m_subsample, colsample_bytree=1, max_depth=round(m_max_depth))#, tree_method='gpu_hist', gpu_id=0)
    
    real_product, fcst_product, product_Y_, product_Y_hat_, product_prev, product_hat = run_model_(model_product, 
                                                                                     product_var_X_train1[product_var_col[:round(col_k)]], Y_product_train,
                                                                                     product_var_X_test1[product_var_col[:round(col_k)]], Y_product_test)
    return fcst_product
    #print(" Sales Mean Average => Train :  %f5 / Test : %f5"%(real_sale, fcst_sale))   


product_optimizer = BayesianOptimization(f=product_opt, pbounds=param_bound, verbose=2, random_state=1)
product_optimizer.maximize(init_points=10, n_iter=100)



