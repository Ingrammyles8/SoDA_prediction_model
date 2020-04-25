# social distancing regressions/machine learning
"""
Created on Fri Apr 17 16:38:51 2020

@author: mai2125
"""

import numpy as np
import pandas as pd
import soc_dist_presets as ps
import soc_dist_graphs as sg
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import metrics
from sklearn.preprocessing import StandardScaler



def merge_dfs(df_list, col="states"):
    '''
    Merges dataframes using an inner join based on specified col
    '''
    ind = 0
    for df in df_list:
        if ind == 0:
            df_final = df
        else:
            df_final = pd.merge(df_final, df, on=col)
        ind += 1
            
    return df_final

dataset = merge_dfs([ps.census_data, ps.mobi_df, ps.states_data, ps.cvd_state])


def scatter_plot_features(df, feature_name):
    '''
    Used for exploring features to determine which ones are correlated with
    SoDA score
    '''
    df.plot(x="SoDA Score", y=feature_name, kind="scatter")
    return

#scatter_plot_features(dataset, 'Persons in poverty, percent')

'''
RIDGE REGRESSION FUNCTIONS
(use m_dataset.iloc[:, -1] in presets for y and all other columns for x)
'''
def multi_ridge_regression(X_in, y_in):
    '''
    Performs a multivariate ridge regression given X features and y outcome
    variable 
    
    Input: dataframe of X features and Series of y outcome variables
    
    Output: dataframe of beta coefficients, and the ridge regressor
    '''
    # Ridge regression 

    # standardizes X values for the ridge regression
    scaler = StandardScaler()
    X = X_in.values
    y = y_in.values
    X_std = scaler.fit_transform(X)
    
    # formulates the training and tests set for each variable
    X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=.2, random_state=0)
    
    # ridge regression algorithm
    regressor = linear_model.RidgeCV(alphas=[.1, 1, 10])
    regressor = regressor.fit(X_std, y)
    regressor.alpha_
    
    # coefficient dataframe and y prediction results of the ridge regression
    coeff_df = pd.DataFrame(regressor.coef_, X_in.columns, columns=["Coefficients"])
    y_pred = regressor.predict(X_test)
    df = pd.DataFrame({"Actual":y_test, "Predicted":y_pred})
    
    print("Ridge regression score: " + str(regressor.score(X_std, y)))
    print("Ridge regression: " + str(metrics.explained_variance_score(y_test, y_pred)))
    print("Ridge regression: " + str(metrics.mean_squared_error(y_test, y_pred)))
    
    return coeff_df, df, regressor
    


def single_ridge_regression(X_in, y_in):
    '''
    Performs a single variable ridge regression given X feature and y outcome
    variable
    
    Input: Series of X feature and Series of y outcome variables
    
    Output: dataframe of beta coefficients, and the ridge regressor
    '''
    # Ridge regression using only cumulative number of COVID cases

    # standardizes X values for the ridge regression
    scaler = StandardScaler()
    X = X_in.values.reshape(-1, 1)
    y = y_in.values.reshape(-1, 1)
    X_std = scaler.fit_transform(X)
    
    # formulates the training and tests set for each variable
    X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=.2, random_state=0)

    # ridge regression algorithm
    regressor = linear_model.RidgeCV(alphas=[.1, 1, 10])
    regressor = regressor.fit(X_std, y)
    regressor.alpha_
    
    
    # coefficient dataframe and y prediction results of the ridge regression
    y_pred = regressor.predict(X_test)
    df = pd.DataFrame({"Actual":y_test.flatten(), "Predicted":y_pred.flatten()})
    # shows model performance
    print("Ridge regression score: " + str(regressor.score(X_std, y)))
    print("Ridge regression explained_variance: " + str(metrics.explained_variance_score(y_test, y_pred)))
    print("Ridge regression MSE: " + str(metrics.mean_squared_error(y_test, y_pred)))
    
    return regressor, df

def rr_coef_df(X, y):
    '''
    Performs gives dataframe of beta coefficients from 
    single variable ridge regressions for every variable in X
    
    Input: Dataframe of X features and Series of y outcome variables
    
    Output: dataframe of beta coefficients
    '''
    # initalize coefficient list
    coef_list = []
    # iterates over columnns of X and takes single variable ridge regressions
    for col in X.columns:
        regressor = single_ridge_regression(X.loc[:, col], y)[0]
        coef_list.append(regressor.coef_[0][0])
    # turns coef_list into coef_df
    coef_df = pd.DataFrame(coef_list, X.columns, columns=["Coefficients"])
    
    return coef_df


def predict_all_factors():
    '''
    Using ridge regression to predict SoDA levels using all factors in m_dataset
    
    Output: df of model results vs actual results
    '''
    pred_all_factors = multi_ridge_regression(ps.m_dataset.iloc[:, 1:-1], 
                                                                  ps.m_dataset.iloc[:, -1])[1]
    return pred_all_factors
    
def predict_cov_only():
    '''
    Using ridge regression to predict SoDA levels using all factors in m_dataset
    
    Output: df of model results vs actual results
    '''
    pred_cov = single_ridge_regression(ps.m_dataset.loc[:,"Cumulative COVID cases (current)"], 
                                                    ps.m_dataset.iloc[:, -1])[1]
    return pred_cov

def get_coef_df(save=False):
    '''
    Using ridge regression to predict SoDA levels using only number of COVID
    cases
    
    Output: df of model results vs actual results
    '''
    coef_df = rr_coef_df(ps.m_dataset.iloc[:, 1:-1], ps.m_dataset.iloc[:, -1])
    if save == True:
        coef_df.to_csv(ps.outputs/"ridge_regression_coef.csv")
    return coef_df

def predict_top_x_factors(x):
    '''
    Using ridge regression to predict SoDA levels using top x most influential 
    factors
    
    Input: x int
    
    Output: df of model results vs actual results
    '''
    coef_df = get_coef_df()
    m_dataset = ps.m_dataset[list(abs(coef_df).sort_values(by="Coefficients", ascending=False).index)]
    pred_top = multi_ridge_regression(m_dataset.iloc[:, :x], ps.m_dataset.iloc[:, -1])[1]
    return pred_top
    
def comp_pred_df(save=False):
    '''
    Merges all prediction results into one dataframe to analyzed in R
    
    Output: merged df of all model results
    '''
    
    all_df = predict_all_factors()
    cov_df = predict_cov_only()
    top_df = predict_top_x_factors(11)
    
    m_df = merge_dfs([all_df, cov_df, top_df], col="Actual")
    m_df = m_df.rename(columns={"Predicted":"Predicted Top Features", 
                         "Predicted_y":"Predicted COVID Count Only", 
                         "Predicted_x":"Predicted Al;l"})
    if save == True:
        m_df.to_csv(ps.outputs/"prediction_results.csv")
    return m_df
    








