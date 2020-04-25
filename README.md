# DSI CUEPID Competition

# Ridge Regression Model for Predicting State Social Distancing Adherence Rates 

This Ridge regression model aims to provide insights into what socioeconomic and clinical demographics contribute to statewide mobility prectices and social distancing adherence (SoDA). Because of the colinearity of socioeconomic data, ridge regression was used to counteract this colinearity and provide a more accurate model. The model then uses these socioecomic clinical factors to predict the SoDA scores of states. The purpose of this algorithm was to highlight possible disparities in social distancing and anticipate which states would be slower to come out of social distancing protocols despite federal guidance. 

## Getting Started

Download the code, data, and dump folders onto your local machine. 
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

## Prerequisites

Python 3.3.6 and above
R 1.1 and above
Python Packages necessary:

```
Python
----
Numpy
Pandas
Pathlib
sklearn
Matplotlib.pyplot
R
----
ggplot
```

## TO RUN
* run python soc_dist_regress.py to the performance of the model under different factors
* To see the prediction values and the coefficients use comp_pred_df(save=True) and get_coef_df(save=True), respectively

## Description of Code and Data

### US_Demographics 
Location of tables for the us demographics data used as model features.
* master_soc_dist_dataset_trim.csv contains all of the data used in the most recent version of the ridge regression model
* All other files contain tables used to make master_soc_dist and the links to where the data was extracted

### Code

soc_dist_regress.py: Executes the ridge regression model, generates dataframes for beta coefficients and model results

* comp_pred_df Gives the prediction values of three variations of the model and the model score of each variation. The three variations include: all features used simultaneously in a multivariate ridge regression, only total COVID count per state in a single variable ridge regression, and the top 11 most impactful fatures determined by the absolute value of the coefficients found by get_coef_df.
* get_coef_df: Creates the dataframe of the beta coeffients by doing single variable ridge regressions for each variable in ps.m_dataset
* scatter_plot_features: Used to vizualize the features vs. SoDA score per state  


soc_dist_presets.py: The logistics of the model and datasheets used in the model. Cleaning of the data was also done in this sheet. Data used includes:
* states demographics
* the number of cumulative COVID cases in a state using CDC data
* the SoDA scores derived by taking absolute value the mean of the percent mobility change from baseline in public areas (parks, recreation areas, retail stores, workplaces, grocery stores, pharmacies, places of transit) averaged over March 16th 2020 to April 11th. The baseline for this data was mobility trends in January and February. Vist https://www.google.com/covid19/mobility/ for more information on the mobility trend data.

soc_dist_graphs.py: functions for visualizing the SoDA data per state, the mobility trends per state, and model performance results

ridge_regression.R: functions for visualizing the beta coefficients of the single ridge regressions of each variable

## Built With

Python 3.3.6


## Authors

* **Myles Ingram**
* **Ashley Zahabian**


## Acknowledgments

* Google Global Mobility Data
* Census.gov
* Politico
* Statista
* CNN
* CDC
