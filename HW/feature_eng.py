import math
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor


# This function will return true if the input is categorical
def define_cat(dataset, col):
    if dataset[col].dtype == "O":
        return True
    else:
        return False


# This function will return true if the input is numeric
def define_numeric(dataset, col):
    if (dataset[col].dtype == "int64") | (dataset[col].dtype == "float64"):
        return True
    else:
        return False


def predictor_response_plots(dataset, predictor, response):
    if (define_cat(dataset, predictor)) & (define_cat(dataset, response)):
        # Heatmap ( categorical variables vs categorical response)
        cat_freq = pd.crosstab(index=dataset[predictor], columns=dataset[response])
        sns.heatmap(data=cat_freq, annot=True, fmt=".1f")
        plt.show()
    if (define_numeric(dataset, predictor)) & (define_cat(dataset, response)):
        # first convert the dataset to wide format
        sns.kdeplot(
            data=dataset,
            x=predictor,
            hue=response,
            cut=0,
            fill=True,
            common_norm=False,
            alpha=1,
        )
        plt.show()

    if (define_numeric(dataset, predictor)) & (define_numeric(dataset, response)):
        plt.scatter(dataset[predictor], dataset[response])
        plt.show()

    if (define_cat(dataset, predictor)) & (define_numeric(dataset, response)):
        sns.violinplot(x=dataset[predictor], y=dataset[response])
        plt.show()


def regression_model(dataset, predictor, response):
    # when the response is categorical (logistic regression)
    if define_numeric(dataset, predictor) & define_numeric(dataset, response):
        x = dataset[predictor]
        y = dataset[response]
        cons = sm.add_constant(x)
        linear_regression_model = sm.OLS(y, cons)
        linear_regression_model_fitted = linear_regression_model.fit()
        print(print(f"linear regression : {response} "))
        print(linear_regression_model_fitted.summary())
        # Get the stats
        # t_value = round(linear_regression_model_fitted.tvalues[1], 6)
        # p_value = "{:.6e}".format(linear_regression_model_fitted.pvalues[1])

    if define_numeric(dataset, predictor) & define_cat(dataset, response):
        # note : This only works for int binary response, it only takes int 0,1 as response.
        # I could not figure out how to do it for non-int binary response.
        x = dataset[predictor]
        y = dataset[response]
        cons = sm.add_constant(x)
        logistic_regression_model = sm.Logit(y, cons)
        logistic_regression_model_fitted = logistic_regression_model.fit()
        print(print(f"Logistic regression for: {response} "))
        print(logistic_regression_model_fitted.summary())
        # Get the stats
        # t_value = round(logistic_regression_model_fitted.tvalues[1], 6)
        # p_value = "{:.6e}".format(logistic_regression_model_fitted.pvalues[1])
    else:
        print("no model associated")


# Note : I still need to work on this to figure out how to find counts of response in each bin
def diff_mean_response(dataset, predictor):
    # Optimal number of bins
    n_bin = math.ceil(math.sqrt(len(dataset[predictor])))
    # Get bin start points (lower bound)
    bins_list = np.linspace(min(dataset[[predictor]]), max(dataset[predictor]), n_bin)
    # create a new df to store values
    bin_df = pd.DataFrame()
    # calculate bins (lower and upper bound)
    bin_df["bins"] = pd.cut(x=dataset[predictor], bins=bins_list)


def random_forest_var_imp(dataset, response):
    # Note : This only works for int binary response
    rf = RandomForestRegressor(n_estimators=100)
    x = dataset[dataset.select_dtypes([np.number]).columns]
    y = dataset[response]
    rf.fit(x, y)
    sorted_idx = rf.feature_importances_.argsort()
    plt.barh(dataset.columns[sorted_idx], rf.feature_importances_[sorted_idx])
    plt.xlabel("Random Forest Feature Importance")


def main():
    # user input to receive dataset
    dataset = pd.read_csv("/Users/maryam/Downloads/archive/House_Rent_Dataset.csv")
    # specifying the response variable
    response_name = "Rent"
    # Don't include response in other variables, we don't want to plot response vs response.
    col_name = [col for col in dataset.columns if col != response_name]
    # create plots
    # for col in col_name:
    #    predictor_response_plots(dataset, col, response_name)
    # create regression models
    for col in col_name:
        regression_model(dataset, col, response_name)
    # To check bins please run this ( still in progress)
    diff_mean_response(dataset, col_name)
    # variable importance
    # please use if response is binary (0,1)
    # random_forest_var_imp(dataset,reponse_name)


if __name__ == "__main__":
    sys.exit(main())
