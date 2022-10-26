import math
import sys
import warnings
from itertools import product

import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as ss
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier


warnings.simplefilter(action="ignore", category=FutureWarning)
# This function will return true if the input is categorical


def define_cat(dataset, col):
    if dataset[col].dtype == "O":
        return True
    elif dataset[col].fillna(0).unique().sum() == 1:
        return True
    else:
        return False


def change_yes_no_to_binary(dataset, col):
    dataset[col] = dataset[col].map({"yes": 1, "no": 0})
    return dataset[col]


def predictor_response_plots(dataset, predictor, response):
    predictor_cat = define_cat(dataset, predictor)
    response_cat = define_cat(dataset, response)
    # res = cat - pred = cat
    if response_cat:
        if predictor_cat:
            cat_freq = pd.crosstab(index=dataset[predictor], columns=dataset[response])
            # conf_matrix = confusion_matrix(dataset[response], dataset[predictor])
            fig = go.Figure(data=go.Heatmap(z=cat_freq, zauto=True))
            fig.update_layout(
                title="Categorical Predictor by Categorical Response",
                xaxis_title="Response=" + response,
                yaxis_title="Predictor=" + predictor,
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
                xaxis_title="Response=" + response,
                yaxis_title="Predictor=" + predictor,
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
            fig.update_layout(
                yaxis_zeroline=False,
                title="Categorical Predictor by Numeric Response",
                xaxis_title="Response=" + response,
                yaxis_title="Predictor=" + predictor,
            )

            plotly.offline.plot(fig)
        # res = numeric - pred = numeric
        else:
            fig = px.scatter(x=dataset[predictor], y=dataset[response], trendline="ols")
            fig.update_layout(
                title="Continuous Response by Continuous Predictor",
                xaxis_title="Response=" + response,
                yaxis_title="Predictor=" + predictor,
            )
            plotly.offline.plot(fig)


def regression_model(dataset, predictor, response):
    predictor_cat = define_cat(dataset, predictor)
    response_cat = define_cat(dataset, response)
    # when the response is categorical (logistic regression)
    if response_cat:
        if ("yes" not in dataset[response].unique()) & (predictor_cat):
            print("no model associated")
        elif ("yes" in dataset[response].unique()) & (predictor_cat):
            print("no model associated")
        elif ("yes" in dataset[response].unique()) & (predictor_cat == False):
            y = change_yes_no_to_binary(dataset, response)
            x = dataset[predictor]
            cons = sm.add_constant(x)
            logistic_regression_model = sm.Logit(y, cons)
            logistic_regression_model_fitted = logistic_regression_model.fit()
            print(print(f"Logistic regression for: {response} "))
            print(logistic_regression_model_fitted.summary())
        else:
            y = dataset[response]
            x = dataset[predictor]
            cons = sm.add_constant(x)
            logistic_regression_model = sm.Logit(y, cons)
            logistic_regression_model_fitted = logistic_regression_model.fit()
            print(print(f"Logistic regression for: {response} "))
            print(logistic_regression_model_fitted.summary())
            # Get the stats
            # t_value = round(logistic_regression_model_fitted.tvalues[1], 6)
            # p_value = "{:.6e}".format(logistic_regression_model_fitted.pvalues[1])

    else:
        # note : This only works for int binary response, it only takes int 0,1 as response.
        # I could not figure out how to do it for non-int binary response.
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
    # count number of data points in each bin
    counts = binned_pred_resp.groupby(["bins"]).count()
    # get the mean value of response in each bin
    mean_response = binned_pred_resp.groupby(["bins"]).mean().fillna(0)
    # merge two dataframes (mean of response and counts per bin)
    binned_pred_resp_counts = pd.concat([counts, mean_response], axis=1)
    binned_pred_resp_counts.reset_index(inplace=True)
    # re-name the columns in the dataframe
    binned_pred_resp_counts.columns = ["bin_interval", "bin_count", "bin_means"]
    # get bin center
    binned_pred_resp_counts["bin_centers"] = binned_pred_resp_counts.bin_interval.apply(
        lambda x: x.mid
    )
    # get population mean
    binned_pred_resp_counts["population_mean"] = binned_pred_resp_counts[
        "bin_means"
    ].mean()
    # get mean squared difference
    binned_pred_resp_counts["mean_squared_diff"] = pow(
        (
            binned_pred_resp_counts["population_mean"]
            - binned_pred_resp_counts["bin_means"]
        ),
        2,
    )
    # get mean squared differences unweighted
    binned_pred_resp_counts["mean_squared_diff_unweighted"] = binned_pred_resp_counts[
        "mean_squared_diff"
    ].sum()
    # population proportion = bin_count/ N (sample size)
    binned_pred_resp_counts["population_proportion"] = (
        binned_pred_resp_counts["bin_count"]
        / binned_pred_resp_counts["bin_count"].sum()
    )
    # weighted mean square difference = mean square difference * population proportion
    binned_pred_resp_counts["mean_squared_diff_weighted"] = (
        binned_pred_resp_counts["mean_squared_diff"]
        * binned_pred_resp_counts["population_proportion"]
    )
    fig = px.bar(binned_pred_resp_counts, x="bin_centers", y="bin_count")
    fig.add_hline(y=binned_pred_resp_counts["response_mean"].mean())
    fig.update_layout(
        title="Difference in mean of response",
        xaxis_title="Predictor=" + predictor,
        yaxis_title="response=" + response,
    )
    fig.show()
    html = binned_pred_resp_counts.to_html()
    text_file = open("mean_of_response.html", "w")
    text_file.write(html)
    text_file.close()


def random_forest_var_imp(dataset, response):
    feature_names = dataset.select_dtypes("number").columns
    forest = RandomForestClassifier(random_state=0)
    forest.fit(dataset[feature_names], dataset[response])
    importances = forest.feature_importances_
    forest_importances = pd.Series(importances, index=feature_names)
    return forest_importances.sort_values(ascending=False)

def get_correlation(dataset,predictor1,predictor2,bias_correction=True, tschuprow=False):
    predictor1_cat= define_cat(dataset,predictor1)
    predictor2_cat= define_cat(dataset,predictor2)
    if predictor1_cat & predictor2_cat:
        corr_coeff = np.nan
        try:
            crosstab_matrix = pd.crosstab(dataset[predictor1], dataset[predictor2])
            n_observations = crosstab_matrix.sum().sum()

            yates_correct = True
            if bias_correction:
                if crosstab_matrix.shape == (2, 2):
                    yates_correct = False

            chi2, _, _, _ = ss.chi2_contingency(
                crosstab_matrix, correction=yates_correct
            )
            phi2 = chi2 / n_observations

            # r and c are number of categories of x and y
            r, c = crosstab_matrix.shape
            if bias_correction:
                phi2_corrected = max(0, phi2 - ((r - 1) * (c - 1)) / (n_observations - 1))
                r_corrected = r - ((r - 1) ** 2) / (n_observations - 1)
                c_corrected = c - ((c - 1) ** 2) / (n_observations - 1)
                if tschuprow:
                    corr_coeff = np.sqrt(
                        phi2_corrected / np.sqrt((r_corrected - 1) * (c_corrected - 1))
                    )
                    return corr_coeff
                corr_coeff = np.sqrt(
                    phi2_corrected / min((r_corrected - 1), (c_corrected - 1))
                )
                return corr_coeff
            if tschuprow:
                corr_coeff = np.sqrt(phi2 / np.sqrt((r - 1) * (c - 1)))
                return corr_coeff
            corr_coeff = np.sqrt(phi2 / min((r - 1), (c - 1)))
            return corr_coeff
        except Exception as ex:
            print(ex)
            if tschuprow:
                warnings.warn("Error calculating Tschuprow's T", RuntimeWarning)
            else:
                warnings.warn("Error calculating Cramer's V", RuntimeWarning)
            return corr_coeff
    elif (predictor1 & predictor2 == False) |  (predictor2 & predictor1 == False):
        categorical_dataset = dataset[categorical_columns]
        dataset_encoded = categorical_dataset.apply(lambda x: pd.factorize(x)[0])
        numeric_dataset = dataset[numeric_columns]
        combined_dataset = pd.concat(
            [numeric_dataset.reset_index(drop=True), dataset_encoded], axis=1
        )
        corr_mat_numeric_cat = combined_dataset.corr()



def correlation_matrix(dataset, numeric_columns, categorical_columns):
    # Calculate correlation metrics between numeric - numeric predictors
    corr_mat_numeric = dataset[numeric_columns].corr()
    corr_mat_numeric = corr_mat_numeric.rename_axis(None).rename_axis(None, axis=1)
    corr_mat_numeric.columns = ['variable1', 'variable2', 'correlation']
    corr_mat_numeric1 = corr_mat_numeric.stack().reset_index()
    return corr_mat_numeric1
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
    #print(chi_test_output.pivot(index="var1", columns="var2", values="coeff"))
    # Calculate correlation metrics between continuous - categorical predictors
    categorical_dataset = dataset[categorical_columns]
    dataset_encoded = categorical_dataset.apply(lambda x: pd.factorize(x)[0])
    numeric_dataset = dataset[numeric_columns]
    combined_dataset = pd.concat(
        [numeric_dataset.reset_index(drop=True), dataset_encoded], axis=1
    )
    corr_mat_numeric_cat = combined_dataset.corr()
    #print(corr_mat_numeric_cat)


def main():
    # Dataset loading
    dataset = pd.read_csv("/home/mahmedi/Downloads/archive(1)/titanic_data.csv")
    dataset = dataset.dropna()
    del dataset["Cabin"]
    del dataset["Name"]
    dataset["binary"] = dataset["Survived"].map({1: "yes", 0: "no"})

    # specifying the response variable
    response_name = "Survived"
    # Don't include response in other variables, we don't want to plot response vs response.
    # col_name = [col for col in dataset.columns if col != response_name]
    # predictor_dataset = dataset[col_name]
    # Split dataset on predictors in list between categoricals and continuous
    # numeric_columns = predictor_dataset.select_dtypes("number").columns
    # categorical_columns = predictor_dataset.select_dtypes("object").columns

    # create plots
    # for col in col_name:
    #    predictor_response_plots(dataset, col, response_name)
    # create regression models
    # for col in col_name:
    #    regression_model(dataset, col, response_name)
    diff_mean_response(dataset, "Age", response_name)
    # variable importance
    # please use if response is binary (0,1)
    # random_forest_var_imp(dataset,reponse_name)
    # correlation_matrix(dataset, numeric_columns, categorical_columns)


if __name__ == "__main__":
    sys.exit(main())
