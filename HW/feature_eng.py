import math
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.ensemble import RandomForestRegressor


def define_boolean(dataset, col):
    boolean = [0, 1]
    if (isinstance(dataset[col], int) | isinstance(dataset[col], float)) & (
        all(p == boolean for p in dataset[col])
    ):
        return True
    else:
        return False


def define_cat(dataset, col):
    if isinstance(dataset[col], object):
        return True
    else:
        return False


def define_numeric(dataset, col):
    if (isinstance(dataset[col], int) | isinstance(dataset[col], float)) & (
        define_boolean(dataset, col)
    ) == False:
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
    if define_numeric(dataset, predictor) & define_numeric(response):
        x = dataset[predictor]
        y = dataset[response]
        predictor = sm.api.add_constant(x)
        linear_regression_model = sm.api.OLS(y, predictor)
        linear_regression_model_fitted = linear_regression_model.fit()
        print(predictor)
        print(linear_regression_model_fitted.summary())
        # Get the stats
        t_value = round(linear_regression_model_fitted.tvalues[1], 6)
        p_value = "{:.6e}".format(linear_regression_model_fitted.pvalues[1])
    if define_numeric(dataset, predictor) & define_cat(response):
        x = dataset[predictor]
        y = dataset[response]
        predictor = smf.api.add_constant(x)
        Logit_model = smf.api.logit(y, predictor)
        Logit_model_fitted = Logit_model.fit()
        print(predictor)
        print(Logit_model_fitted.summary())
        # Get the stats
        t_value = round(Logit_model_fitted.tvalues[1], 6)
        p_value = "{:.6e}".format(Logit_model_fitted.pvalues[1])
    else:
        print("no model associated")


def diff_mean_response(dataset, predictor, response):
    # determining number of bins
    n_bin = math.ceil(math.sqrt(len(dataset[predictor])))
    bin_width = (max(dataset[predictor]) - min(dataset[predictor])) / n_bin


def random_forest_var_imp(dataset, predictor, response):
    rf = RandomForestRegressor(n_estimators=100)
    x = dataset[predictor]
    y = dataset[response]
    rf.fit(x, y)
    sorted_idx = rf.feature_importances_.argsort()
    plt.barh(dataset.columns[sorted_idx], rf.feature_importances_[sorted_idx])
    plt.xlabel("Random Forest Feature Importance")


def main():
    # user input to receive dataset

    dataset = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    )
    # specifying the response variable
    response_name = " <=50K"
    # Don't include response in other variables, we don't want to plot response vs response.
    col_name = [col for col in dataset.columns if col != response_name]

    for col in col_name:
        predictor_response_plots(dataset, col, response_name)
    # for i in range(1, len(col_name)):
    #  regression_model(col_name[1])


if __name__ == "__main__":
    sys.exit(main())
