import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def mean_of_response(dataset, predictor, response, figure=False):
    n_bin = 10
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
    binned_pred_resp_counts.columns = [
        "bins_interval",
        "bin_count",
        "bin_means",
    ]
    # get bin center
    binned_pred_resp_counts[
        "bin_centers"
    ] = binned_pred_resp_counts.bins_interval.apply(lambda x: x.mid)

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
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add traces
        fig.add_trace(
            go.Bar(
                x=binned_pred_resp_counts["bin_centers"],
                y=binned_pred_resp_counts["mean_squared_diff"],
                name="Response",
            ),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(
                x=binned_pred_resp_counts["bin_centers"],
                y=binned_pred_resp_counts["population_mean"],
                name="Population",
            ),
            secondary_y=True,
        )
        fig.add_trace(
            go.Scatter(
                x=binned_pred_resp_counts["bin_centers"],
                y=binned_pred_resp_counts["mean_squared_diff"],
                name="Population",
            ),
            secondary_y=True,
        )
        fig.write_html(predictor + "-" + response + "_mean_of_response_plot.html")

    else:
        return binned_pred_resp_counts["mean_squared_diff_weighted"].sum() / len(
            binned_pred_resp_counts.index
        )
