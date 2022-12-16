import os
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
    # elif dataset[col].fillna(0).unique().sum() == 1:
    #   return True
    else:
        return False


def change_yes_no_to_binary(dataset, col):
    dataset[col] = dataset[col].map({"yes": 1, "no": 0})
    return dataset[col]


def save_html(file, var1, var2, mean_res=False, two_predictors=False):
    html = file.to_html()
    if two_predictors:
        if mean_res:
            fig_file = open(var1 + "-" + var2 + "_mean_response_heatmap.html", "w")
        else:
            fig_file = open(var1 + "-" + var2 + ".html", "w")
    else:
        fig_file = open(var1 + "_mean_response_bargraph.html", "w")

    fig_file.write(html)
    fig_file.close()


def make_clickable(url, name):
    return f'<a href="{url}">{name}</a>'


def predictor_response_plots(dataset, predictor, response):
    predictor_cat = define_cat(dataset, predictor)
    response_cat = define_cat(dataset, response)
    # res = cat - pred = cat
    if response_cat:
        if predictor_cat:
            cat_freq = pd.crosstab(index=dataset[predictor], columns=dataset[response])
            fig = go.Figure(data=go.Heatmap(z=cat_freq, zauto=True))
            fig.update_layout(
                title="Categorical Predictor by Categorical Response",
                xaxis_title="Response=" + response,
                yaxis_title="Predictor=" + predictor,
            )
            save_html(fig, predictor, response)

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

            save_html(fig, predictor, response)

    # res = numeric - pred = cat
    else:
        if predictor_cat:
            fig = go.Figure(
                data=go.Violin(
                    y=dataset[response],
                    x=dataset[predictor],
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

            save_html(fig, predictor, response)
        # res = numeric - pred = numeric
        else:
            fig = px.scatter(x=dataset[predictor], y=dataset[response], trendline="ols")
            fig.update_layout(
                title="Continuous Response by Continuous Predictor",
                xaxis_title="Response=" + response,
                yaxis_title="Predictor=" + predictor,
            )

            save_html(fig, predictor, response)


def regression_model(dataset, predictor, response):
    predictor_cat = define_cat(dataset, predictor)
    response_cat = define_cat(dataset, response)
    # when the response is categorical (logistic regression)
    if response_cat:
        if ("yes" not in dataset[response].unique()) & predictor_cat:
            print("no model associated")
        elif ("yes" in dataset[response].unique()) & predictor_cat:
            print("no model associated")
        elif ("yes" in dataset[response].unique()) & (not predictor_cat):
            y = change_yes_no_to_binary(dataset, response)
            x = dataset[predictor]
            cons = sm.add_constant(x)
            logistic_regression_model = sm.Logit(y, cons)
            logistic_regression_model_fitted = logistic_regression_model.fit()
            t_value = round(logistic_regression_model_fitted.tvalues[1], 6)
            p_value = "{:.6e}".format(logistic_regression_model_fitted.pvalues[1])
            # print(print(f"Logistic regression for: {response} "))
            # print(logistic_regression_model_fitted.summary())
            return "t_value is :" + t_value + "p_value is:" + p_value

        else:
            y = dataset[response]
            x = dataset[predictor]
            cons = sm.add_constant(x)
            logistic_regression_model = sm.Logit(y, cons)
            logistic_regression_model_fitted = logistic_regression_model.fit()
            # print(print(f"Logistic regression for: {response} "))
            # print(logistic_regression_model_fitted.summary())
            # Get the stats
            t_value = round(logistic_regression_model_fitted.tvalues[1], 6)
            p_value = "{:.6e}".format(logistic_regression_model_fitted.pvalues[1])
            return "t_value is :" + t_value + "p_value is:" + p_value

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
            # print(print(f"linear regression : {response} "))
            # print(linear_regression_model_fitted.summary())
            # Get the stats
            t_value = round(linear_regression_model_fitted.tvalues[1], 6)
            p_value = "{:.6e}".format(linear_regression_model_fitted.pvalues[1])
            return "t_value is :" + t_value + "p_value is:" + p_value


def brute_force(dataset, predictor1, predictor2, response, figure=True):
    predictor1_cat = define_cat(dataset, predictor1)
    predictor2_cat = define_cat(dataset, predictor2)
    if predictor1_cat & predictor2_cat:
        bin_df = (
            dataset.groupby([predictor1, predictor2])[response]
            .mean()
            .reset_index(drop=True)
        )
        counts = dataset.groupby([predictor1, predictor2]).size().reset_index(drop=True)
        bin_df_merged = pd.merge(
            bin_df, counts, on=[predictor1, predictor2], how="outer"
        )
        bin_df_merged["population_mean"] = bin_df_merged[response].mean()
        bin_df_merged["mean_squared_diff"] = pow(
            (bin_df_merged["population_mean"] - bin_df_merged[response]),
            2,
        )
        # rename columns
        bin_df_merged.columns = [
            "predictor1",
            "predictor2",
            "response",
            "counts",
            "population_mean",
            "mean_squared_diff",
        ]
        # mean_squared_diff_unweighted = bin_df_merged["mean_squared_diff"].sum() / len(
        #    bin_df_merged.index
        # )
        # population proportion = bin_count/ N (sample size)
        bin_df_merged["population_proportion"] = (
            bin_df_merged["counts"] / bin_df_merged["counts"].sum()
        )
        # weighted mean square difference = mean square difference * population proportion
        bin_df_merged["mean_squared_diff_weighted"] = (
            bin_df_merged["mean_squared_diff"] * bin_df_merged["population_proportion"]
        )
        mean_squared_diff_weighted = bin_df_merged[
            "mean_squared_diff_weighted"
        ].sum() / len(bin_df_merged.index)
        if figure:
            df_wide = bin_df_merged.pivot(
                index="predictor1",
                columns="predictor2",
                values="mean_squared_diff_weighted",
            )
            fig = px.imshow(df_wide)
            save_html(fig, predictor1, predictor2, mean_res=True)

        else:
            return mean_squared_diff_weighted

    elif (not predictor1_cat) & (not predictor2_cat):
        # Optimal number of bins
        n_bin1 = 10
        n_bin2 = 10
        # Get bin start points (lower bound)
        bins_list1 = np.linspace(
            min(dataset[predictor1]), max(dataset[predictor1]), n_bin1
        )
        bins_list2 = np.linspace(
            min(dataset[predictor2]), max(dataset[predictor2]), n_bin2
        )
        # create a new df to store values
        bin_df = pd.DataFrame()
        # calculate bins (lower and upper bound)
        bin_df["bins1"] = pd.cut(x=dataset[predictor1], bins=bins_list1)
        bin_df["bins2"] = pd.cut(x=dataset[predictor2], bins=bins_list2)
        binned_pred_resp = pd.concat(
            [bin_df.reset_index(drop=True), dataset[response]], axis=1
        )
        # count number of data points in each bin
        counts = binned_pred_resp.groupby(["bins1", "bins2"]).count()
        # get the mean value of response in each bin
        mean_response = binned_pred_resp.groupby(["bins1", "bins2"]).mean().fillna(0)
        # merge two dataframes (mean of response and counts per bin)
        binned_pred_resp_counts = pd.concat([counts, mean_response], axis=1)
        binned_pred_resp_counts.reset_index(inplace=True)
        # re-name the columns in the dataframe
        binned_pred_resp_counts.columns = [
            "bins_interval1",
            "bins_interval2",
            "bin_count",
            "bin_means",
        ]
        # get bin center
        binned_pred_resp_counts[
            "bin_centers1"
        ] = binned_pred_resp_counts.bins_interval1.apply(lambda x: x.mid)
        binned_pred_resp_counts[
            "bin_centers2"
        ] = binned_pred_resp_counts.bins_interval2.apply(lambda x: x.mid)
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
        # mean_squared_diff_unweighted = binned_pred_resp_counts["mean_squared_diff"].sum()
        # / len(binned_pred_resp_counts.index)
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
        if figure:
            df_wide = binned_pred_resp_counts.pivot(
                index="bin_centers1",
                columns="bin_centers2",
                values="mean_squared_diff_weighted",
            )
            fig = px.imshow(df_wide)
            save_html(fig, predictor1, predictor2, mean_res=True)

        else:
            return binned_pred_resp_counts["mean_squared_diff_weighted"].sum() / len(
                binned_pred_resp_counts.index
            )

    else:
        if predictor1_cat:
            cat_var = predictor1
            num_var = predictor2
        else:
            cat_var = predictor2
            num_var = predictor1
        n_bin = 10
        # Get bin start points (lower bound)
        bins_list = np.linspace(min(dataset[num_var]), max(dataset[num_var]), n_bin)
        bin_df = pd.DataFrame()
        bin_df["bins"] = pd.cut(x=dataset[num_var], bins=bins_list)
        bin_df["bin_centers"] = bin_df.bins.apply(lambda x: x.mid)
        bin_df["predictor2"] = dataset[cat_var]
        bin_df.columns = ["predictor1", "bin_centers", "predictor2"]
        bin_df_merged = pd.concat([bin_df, dataset[response]], axis=1)
        bin_df_merged.columns = ["predictor1", "bin_centers", "predictor2", "response"]

        binned_pred_resp = (
            bin_df_merged.groupby(["bin_centers", "predictor2"])["response"]
            .mean()
            .reset_index()
        )
        counts = (
            bin_df_merged.groupby(["bin_centers", "predictor2"]).size().reset_index()
        )
        bin_df_merged = pd.merge(
            binned_pred_resp, counts, on=["bin_centers", "predictor2"], how="outer"
        )
        bin_df_merged.columns = ["bin_centers", "predictor2", "response", "counts"]
        bin_df_merged["population_mean"] = bin_df_merged["response"].mean()
        bin_df_merged["mean_squared_diff"] = pow(
            (bin_df_merged["population_mean"] - bin_df_merged["response"]),
            2,
        )

        # population proportion = bin_count/ N (sample size)
        bin_df_merged["population_proportion"] = (
            bin_df_merged["counts"] / bin_df_merged["counts"].sum()
        )
        # weighted mean square difference = mean square difference * population proportion
        bin_df_merged["mean_squared_diff_weighted"] = (
            bin_df_merged["mean_squared_diff"] * bin_df_merged["population_proportion"]
        )
        bin_df_merged["mean_squared_diff_weighted"] = bin_df_merged[
            "mean_squared_diff_weighted"
        ].sum() / len(bin_df_merged.index)
        if figure:
            df_wide = bin_df_merged.pivot(
                index="bin_centers",
                columns="predictor2",
                values="mean_squared_diff_weighted",
            )
            fig = px.imshow(df_wide)
            save_html(fig, predictor1, predictor2, mean_res=True)

        else:
            return bin_df_merged["mean_squared_diff_weighted"].sum() / len(
                bin_df_merged.index
            )


def mean_of_response(dataset, predictor, response, figure=False):
    n_bin1 = 10
    # Get bin start points (lower bound)
    bins_list1 = np.linspace(min(dataset[predictor]), max(dataset[predictor]), n_bin1)
    # create a new df to store values
    bin_df = pd.DataFrame()
    # calculate bins (lower and upper bound)
    bin_df["bins"] = pd.cut(x=dataset[predictor], bins=bins_list1)
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
    binned_pred_resp_counts.columns = [
        "bins_interval1",
        "bin_count",
        "bin_means",
    ]
    # get bin center
    binned_pred_resp_counts[
        "bin_centers"
    ] = binned_pred_resp_counts.bins_interval1.apply(lambda x: x.mid)

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
    if figure:
        fig = px.bar(binned_pred_resp_counts, x="bin_centers", y="bin_count")
        save_html(fig, predictor, "none", two_predictors=False, mean_res=False)

    else:
        return binned_pred_resp_counts["mean_squared_diff_weighted"].sum() / len(
            binned_pred_resp_counts.index
        )


def random_forest_var_imp(dataset, predictor_list, response):
    feature_names = dataset[predictor_list].select_dtypes("number").columns
    forest = RandomForestClassifier(random_state=0)
    forest.fit(dataset[feature_names], dataset[response])
    importances = forest.feature_importances_
    forest_importances = pd.Series(importances, index=feature_names)
    return forest_importances.sort_values(ascending=False)


def get_correlation(
    dataset, predictor1, predictor2, bias_correction=False, tschuprow=True
):
    predictor1_cat = define_cat(dataset, predictor1)
    predictor2_cat = define_cat(dataset, predictor2)
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
                phi2_corrected = max(
                    0, phi2 - ((r - 1) * (c - 1)) / (n_observations - 1)
                )
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
    elif (predictor1_cat & (not predictor2_cat)) | (
        predictor2_cat & (not predictor1_cat)
    ):
        if predictor1_cat:
            categories = dataset[predictor1].to_numpy()
            measurements = dataset[predictor2].to_numpy()
        else:
            categories = dataset[predictor2].to_numpy()
            measurements = dataset[predictor1].to_numpy()

        fcat, _ = pd.factorize(categories)
        cat_num = np.max(fcat) + 1
        y_avg_array = np.zeros(cat_num)
        n_array = np.zeros(cat_num)
        for i in range(0, cat_num):
            cat_measures = measurements[np.argwhere(fcat == i).flatten()]
            n_array[i] = len(cat_measures)
            y_avg_array[i] = np.average(cat_measures)
        y_total_avg = np.sum(np.multiply(y_avg_array, n_array)) / np.sum(n_array)
        numerator = np.sum(
            np.multiply(n_array, np.power(np.subtract(y_avg_array, y_total_avg), 2))
        )
        denominator = np.sum(np.power(np.subtract(measurements, y_total_avg), 2))
        if numerator == 0:
            eta = 0.0
        else:
            eta = np.sqrt(numerator / denominator)
        return eta

    else:
        corr = np.corrcoef(dataset[predictor1], dataset[predictor2])[0, 1]
        return corr


def generate_tables(
    dataset, categorical_columns, numeric_columns, response_name, cat1=True, cat2=True
):
    if cat1 & cat2:
        var_list = list(product(categorical_columns, categorical_columns))
    elif (not cat1) & (not cat2):
        var_list = list(product(numeric_columns, numeric_columns))
    else:
        var_list = list(product(numeric_columns, categorical_columns))
    d = []
    for i in range(0, len(var_list)):
        var1 = var_list[i][0]
        var2 = var_list[i][1]
        corr = get_correlation(
            dataset,
            var_list[i][0],
            var_list[i][1],
            bias_correction=False,
            tschuprow=True,
        )
        predictor_response_plots(dataset, var_list[i][0], response_name)
        plot_link = var_list[i][0] + "-" + var_list[i][1] + ".html"
        d.append(
            {
                "predictor1": var1,
                "predictor2": var2,
                "correlation": corr,
                "plot": f'<a href="{plot_link}">{plot_link}</a>',
            }
        )
    df = pd.DataFrame(d)
    df["correlation_absolute_value"] = abs(df["correlation"])
    df = df.sort_values(by="correlation_absolute_value", ascending=False)
    return df


def generate_cor_mat(long_format_table):
    df_wide = long_format_table.pivot(
        index="predictor1", columns="predictor2", values="correlation"
    )
    fig = px.imshow(df_wide)
    plotly.offline.plot(fig)


def generate_brute_force(
    dataset, categorical_columns, numeric_columns, response_name, cat1=True, cat2=True
):
    if cat1 & cat2:
        var_list = list(product(categorical_columns, categorical_columns))
    elif (not cat1) & (not cat2):
        var_list = list(product(numeric_columns, numeric_columns))
    else:
        var_list = list(product(numeric_columns, categorical_columns))
    d = []
    for i in range(0, len(var_list)):
        var1 = var_list[i][0]
        var2 = var_list[i][1]
        brute_force_value = brute_force(
            dataset, var_list[i][0], var_list[i][1], response_name, figure=False
        )
        brute_force(dataset, var_list[i][0], var_list[i][1], response_name, figure=True)
        plot_link = var1 + "-" + var2 + "_mean_response_heatmap.html"
        d.append(
            {
                "predictor1": var1,
                "predictor2": var2,
                "weighted_mean_response": brute_force_value,
                "plot": f'<a href="{plot_link}">{plot_link}</a>',
            }
        )
    df = pd.DataFrame(d)
    return df


def main():
    # Dataset loading
    from dataset_loader import TestDatasets

    t = TestDatasets()
    dataset = t.get_test_data_set(data_set_name="titanic")[0]
    dataset = dataset.dropna()
    response_name = t.get_test_data_set(data_set_name="titanic")[2]
    predictors_list = [col for col in dataset.columns if col != response_name]
    predictor_dataset = dataset[predictors_list]
    # Split dataset on predictors in list between categoricals and continuous
    numeric_columns = predictor_dataset.select_dtypes("number").columns
    categorical_columns = predictor_dataset.select_dtypes("object").columns
    # generate cat-cat correlation tables
    cat_cat_cor_table = generate_tables(
        dataset,
        categorical_columns,
        numeric_columns,
        response_name,
        cat1=True,
        cat2=True,
    )
    cat_cat_cor_table.to_html("cat_cat_correlation_table.html", escape=False)
    # generate cat-cat correlation matrix
    generate_cor_mat(cat_cat_cor_table)

    # generate num_num correlation tables
    num_num_cor_table = generate_tables(
        dataset,
        categorical_columns,
        numeric_columns,
        response_name,
        cat1=False,
        cat2=False,
    )
    num_num_cor_table.to_html("num_num_correlation_table.html", escape=False)
    # generate num-num correlation matrix
    generate_cor_mat(num_num_cor_table)
    # generate cat_num correlation tables
    cat_num_cor_table = generate_tables(
        dataset,
        categorical_columns,
        numeric_columns,
        response_name,
        cat1=True,
        cat2=False,
    )
    cat_num_cor_table.to_html("cat_num_correlation_table.html", escape=False)
    # generate num-num correlation matrix
    generate_cor_mat(cat_num_cor_table)

    # Brute force for num-num
    num_num_brute_force_table = generate_brute_force(
        dataset,
        categorical_columns,
        numeric_columns,
        response_name,
        cat1=False,
        cat2=False,
    )
    num_num_brute_force_table.to_html("num_num_brute_force_table.html", escape=False)
    # Brute force for cat_num
    cat_num_brute_force_table = generate_brute_force(
        dataset,
        categorical_columns,
        numeric_columns,
        response_name,
        cat1=True,
        cat2=False,
    )
    cat_num_brute_force_table.to_html("cat_num_brute_force_table.html", escape=False)
    # Brute force for cat_cat
    cat_cat_brute_force_table = generate_brute_force(
        dataset,
        categorical_columns,
        numeric_columns,
        response_name,
        cat1=True,
        cat2=True,
    )
    cat_cat_brute_force_table.to_html("cat_cat_brute_force_table.html", escape=False)


if __name__ == "__main__":
    sys.exit(main())
