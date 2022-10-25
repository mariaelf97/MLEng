import math
import sys
import warnings
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as ss
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix

warnings.simplefilter(action="ignore", category=FutureWarning)
# This function will return true if the input is categorical


def define_cat(dataset, col):
    if dataset[col].dtype == "O":
        return True
    else:
        return False


def predictor_response_plots(dataset, predictor, response):
    predictor_cat = define_cat(predictor)
    response_cat = define_cat(response)
    # res = cat - pred = cat
    if response_cat:
        if predictor_cat:
            # Heatmap ( categorical variables vs categorical response)
            conf_matrix = confusion_matrix(dataset[response], dataset[predictor])
            fig = go.Figure(
                data=go.Heatmap(z=conf_matrix, zmin=0, zmax=conf_matrix.max())
            )
            fig.update_layout(
                title="Categorical Predictor by Categorical Response",
                xaxis_title="Response",
                yaxis_title="Predictor",
            )
            plotly.offline.plot(fig)
        # res = cat - pred = numeric
        else:
            fig = px.histogram(
                dataset,
                x=dataset[predictor],
                color=dataset[response],
                hover_data=dataset.columns,
            )
            fig.update_layout(
                title="Continuous Predictor by Categorical Response",
                xaxis_title="Predictor",
                yaxis_title="Distribution",
            )
            plotly.offline.plot(fig)

    # res = numeric - pred = cat
    else:
        if predictor_cat:
            fig = go.Figure(
                data=go.Violin(
                    y=dataset[response],
                    x=dataset[predictor],
                    box_visible=True,
                    line_color="black",
                    meanline_visible=True,
                    fillcolor="lightseagreen",
                    opacity=0.6,
                    x0=response,
                )
            )
            fig.update_layout(yaxis_zeroline=False)
            plotly.offline.plot(fig)
        # res = numeric - pred = numeric
        else:
            fig = px.scatter(x=dataset[predictor], y=dataset[response], trendline="ols")
            fig.update_layout(
                title="Continuous Response by Continuous Predictor",
                xaxis_title="Predictor",
                yaxis_title="Response",
            )
            plotly.offline.plot(fig)


def regression_model(dataset, predictor, response):
    predictor_cat = define_cat(predictor)
    response_cat = define_cat(response)
    # when the response is categorical (logistic regression)
    if response_cat:
        if predictor_cat:
            print("no model associated")
        else:
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
    else:
        # note : This only works for int binary response, it only takes int 0,1 as response.
        # I could not figure out how to do it for non-int binary response.
        if predictor_cat:
            print("no model associated")
        else:
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


# Note : I still need to work on this to figure out how to find counts of response in each bin
def diff_mean_response(dataset, predictor, response):
    # Optimal number of bins
    n_bin = math.ceil(math.sqrt(len(dataset[predictor])))
    # Get bin start points (lower bound)
    bins_list = np.linspace(min(dataset[predictor]), max(dataset[predictor]), n_bin)
    # create a new df to store values
    bin_df = pd.DataFrame()
    # calculate bins (lower and upper bound)
    bin_df["bins"] = pd.cut(x=dataset[predictor], bins=bins_list)
    binned_pred_resp = pd.concat(
        [bin_df.reset_index(drop=True), dataset[response]], axis=1
    )
    counts = binned_pred_resp.groupby(["bins"]).count()
    mean_response = binned_pred_resp.groupby(["bins"]).mean().fillna(0)
    binned_pred_resp_counts = pd.concat([counts, mean_response], axis=1)
    binned_pred_resp_counts.reset_index(inplace=True)
    binned_pred_resp_counts.columns = ["bins", "counts", "response"]
    binned_pred_resp_counts["bin_median"] = binned_pred_resp_counts.bins.apply(
        lambda x: x.mid
    )
    plt.bar(binned_pred_resp_counts["bin_median"], binned_pred_resp_counts["counts"])
    plt.show()


def random_forest_var_imp(dataset, response):
    # Note : This only works for int binary response
    rf = RandomForestRegressor(n_estimators=100)
    x = dataset[dataset.select_dtypes([np.number]).columns]
    y = dataset[response]
    rf.fit(x, y)
    sorted_idx = rf.feature_importances_.argsort()
    plt.barh(dataset.columns[sorted_idx], rf.feature_importances_[sorted_idx])
    plt.xlabel("Random Forest Feature Importance")


def correlation_matrix(dataset, numeric_columns, categorical_columns):
    # Calculate correlation metrics between numeric - numeric predictors
    corr_mat_numeric = dataset[numeric_columns].corr()
    print(corr_mat_numeric)
    # Calculate correlation metrics between categorical - categorical predictors
    cat_var1 = categorical_columns
    cat_var2 = categorical_columns
    cat_var_prod = list(product(cat_var1, cat_var2, repeat=1))
    df_cat_v1 = dataset[(categorical_columns)].dropna()
    result = []
    for i in cat_var_prod:
        if i[0] != i[1]:
            result.append(
                (
                    i[0],
                    i[1],
                    list(
                        ss.chi2_contingency(
                            pd.crosstab(df_cat_v1[i[0]], df_cat_v1[i[1]])
                        )
                    )[1],
                )
            )
    chi_test_output = pd.DataFrame(result, columns=["var1", "var2", "coeff"])
    print(chi_test_output.pivot(index="var1", columns="var2", values="coeff"))
    # Calculate correlation metrics between continuous - categorical predictors
    categorical_dataset = dataset[categorical_columns]
    dataset_encoded = categorical_dataset.apply(lambda x: pd.factorize(x)[0])
    numeric_dataset = dataset[numeric_columns]
    combined_dataset = pd.concat(
        [numeric_dataset.reset_index(drop=True), dataset_encoded], axis=1
    )
    corr_mat_numeric_cat = combined_dataset.corr()
    print(corr_mat_numeric_cat)


def main():
    # Dataset loading
    dataset = pd.read_csv("/Users/maryam/Downloads/archive/House_Rent_Dataset.csv")
    # specifying the response variable
    response_name = "Rent"
    # Don't include response in other variables, we don't want to plot response vs response.
    col_name = [col for col in dataset.columns if col != response_name]
    predictor_dataset = dataset[col_name]
    # Split dataset on predictors in list between categoricals and continuous
    numeric_columns = predictor_dataset.select_dtypes("number").columns
    categorical_columns = predictor_dataset.select_dtypes("object").columns

    # create plots
    for col in col_name:
        predictor_response_plots(dataset, col, response_name)
    # create regression models
    for col in col_name:
        regression_model(dataset, col, response_name)
    # To check bins please run this ( still in progress)
    diff_mean_response(dataset, response_name)
    # variable importance
    # please use if response is binary (0,1)
    # random_forest_var_imp(dataset,reponse_name)
    correlation_matrix(dataset, numeric_columns, categorical_columns)


if __name__ == "__main__":
    sys.exit(main())
