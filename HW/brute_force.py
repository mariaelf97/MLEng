import numpy as np
import pandas as pd
import plotly.express as px
from define_data_type import define_cat


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
            fig.write_html(
                "output/" + predictor1 + "-" + predictor2 + "_brute_force_plot.html"
            )

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
            fig.write_html(
                "output/" + predictor1 + "-" + predictor2 + "_brute_force_plot.html"
            )

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
            fig.write_html(
                "output/" + predictor1 + "-" + predictor2 + "_brute_force_plot.html"
            )

        else:
            return bin_df_merged["mean_squared_diff_weighted"].sum() / len(
                bin_df_merged.index
            )
