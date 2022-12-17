import sys
from itertools import product

import numpy as np
import pandas as pd
import plotly.express as px
import sqlalchemy
from brute_force import brute_force
from correlation import get_correlation
from mean_of_response import mean_of_response
from model import predict_model, random_forest_var_imp
from predictor_response_plots import predictor_response_plots
from regression import logistic_reg


def main():
    db_user = ""
    db_pass = ""  # pragma: allowlist secret
    db_host = "localhost"
    db_database = "test"
    connect_string = (
        f"mariadb+mariadbconnector://{db_user}:{db_pass}@{db_host}/{db_database}"
    )
    # pragma: allowlist secret

    sql_engine = sqlalchemy.create_engine(connect_string)

    query = """select * from joined_team_batting_pitching_boxscore_diff;"""

    df = pd.read_sql_query(query, sql_engine)

    # replace blank values with NAs
    df = df.mask(df == "")
    # removed batting_go_to_fo_or_ao_home ,batting_go_to_fo_or_ao_away,
    # pitching_go_to_ao_home, pitching_go_to_ao_away
    # batting_go_to_fo_or_ao_diff, pitching_go_to_ao_diff as they have many missing values
    features_to_delete = [
        "batting_go_to_fo_or_ao_home",
        "batting_go_to_fo_or_ao_away",
        "pitching_go_to_ao_home",
        "pitching_go_to_ao_away",
        "batting_go_to_fo_or_ao_diff",
        "pitching_go_to_ao_diff",
    ]
    # remove missing values
    df = df.drop(features_to_delete, axis=1)
    # the remaining 219 missing values can be deleted
    # increased to 573 once replaced missing values with nan
    df = df.dropna()
    # drop features that are not helpful
    df = df.drop(["game_id", "team_id"], axis=1)
    # defining winner in numeric values because the model can't handle categorical ones
    df["winner"] = [1 if x == "H" else 0 for x in df["winner_home_or_away"]]
    del df["winner_home_or_away"]
    response_name = "winner"
    predictors_list = [
        col for col in df.columns if col != [response_name, "local_date"]
    ]

    # removing highly correlated variables
    # correlation matrix
    df_correlation_matrix = df[predictors_list].corr().abs()
    upper_tri = df_correlation_matrix.where(
        np.triu(np.ones(df_correlation_matrix.shape), k=1).astype(np.bool)
    )
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.7)]
    df = df.drop(df.columns[to_drop], axis=1)
    fig = px.imshow(upper_tri)
    fig.write_html("output/correlation_heatmap.html")

    # mean of response, p-value, t-value, predictor response plots
    d = []
    for i in range(0, len(predictors_list)):
        mean_resp_weighted = mean_of_response(
            df, predictors_list[i], response_name, figure=False
        )[0]
        mean_resp_unweighted = mean_of_response(
            df, predictors_list[i], response_name, figure=False
        )[1]
        predictor_response_plots(df, predictors_list[i], response_name)
        mean_of_response(df, predictors_list[i], response_name, figure=True)
        p_value = logistic_reg(df, predictors_list[i], response_name)[0]
        t_value = logistic_reg(df, predictors_list[i], response_name)[1]
        plot_link_mean_of_response = (
            "output/"
            + predictors_list[i]
            + "-"
            + response_name
            + "_mean_of_response_plot.html"
        )
        plot_link_predictor_response = (
            "output/"
            + predictors_list[i]
            + "-"
            + response_name
            + "_predictor_response_plot.html"
        )
        d.append(
            {
                "predictor": predictors_list[i],
                "mean_squared_diff_weighted": mean_resp_weighted,
                "mean_squared_diff": mean_resp_unweighted,
                "p_value": p_value,
                "t_value": t_value,
                "predictor response plot": f'<a href="{plot_link_predictor_response}">Link</a>',
                "mean of response plot": f'<a href="{plot_link_mean_of_response}">Link</a>',
            }
        )
    mean_of_resp_table = pd.DataFrame(d)
    mean_of_resp_table = mean_of_resp_table.sort_values(
        by="mean_squared_diff_weighted", ascending=False
    )
    mean_of_resp_table.to_html("output/mean_of_response_table.html", escape=False)

    # generate pair-wise brute-force and correlation values
    var_list = list(product(predictors_list, predictors_list))
    d = []
    for i in range(0, len(var_list)):
        var1 = var_list[i][0]
        var2 = var_list[i][1]
        corr = get_correlation(
            df,
            var_list[i][0],
            var_list[i][1],
            tschuprow=True,
        )
        brute_force_value = brute_force(
            df, var_list[i][0], var_list[i][1], response_name, figure=False
        )
        # brute_force(df, var_list[i][0], var_list[i][1], response_name, figure=True)
        plot_link = (
            "output/" + var_list[i][0] + "-" + var_list[i][1] + "_brute_force_plot.html"
        )
        d.append(
            {
                "predictor1": var1,
                "predictor2": var2,
                "correlation": corr,
                "brute_force_value": brute_force_value,
                "plot": f'<a href="{plot_link}">Link</a>',
            }
        )
    pair_wise_df = pd.DataFrame(d)
    pair_wise_df["correlation_absolute_value"] = abs(pair_wise_df["correlation"])
    pair_wise_df = pair_wise_df.sort_values(
        by="correlation_absolute_value", ascending=False
    )
    pair_wise_df.to_html("output/pair_wise_variable_table.html", escape=False)

    # variable importance
    rf = pd.DataFrame(random_forest_var_imp(df, predictors_list, response_name))
    rf.to_html("output/+random_forest_variable_importance.html", escape=False)

    # choose features to include in the model
    predictors_to_include = [
        "batting_average_batting_home",
        "pitching_so_to_hr_away",
        "pitching_ab_to_hr_home",
        "batting_ab_to_hr_home",
        "pitching_so_to_hr_home",
        "batting_hr_to_hit_home",
        "batting_average_batting_away",
        "pitching_ab_to_hr_away",
        "batting_go_to_fo_or_ao_home",
        "batting_w_to_sr_away",
        "batting_hr_to_hit_away",
        "batting_groundout_home",
        "pitching_groundout_home",
        "batting_w_to_sr_home",
    ]

    # Random forest model
    predict_model(df, predictors_to_include, response_name)


if __name__ == "__main__":
    sys.exit(main())
