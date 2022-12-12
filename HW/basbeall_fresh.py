import sys
from itertools import product

import pandas as pd
import sqlalchemy
from brute_force import brute_force
from correlation import get_correlation
from mean_of_response import mean_of_response
from predictor_response_plots import predictor_response_plots


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

    query = """select * from joined_team_batting_pitching_boxscore;"""

    df = pd.read_sql_query(query, sql_engine)
    # remove missing values
    df = df.dropna()
    del df["game_id"]
    del df["team_id"]
    df["winner"] = [1 if x == "H" else 0 for x in df["winner_home_or_away"]]
    del df["winner_home_or_away"]
    response_name = "winner"
    predictors_list = [col for col in df.columns if col != response_name]

    # save predictor response plots as html output
    for predictor in predictors_list:
        predictor_response_plots(df, predictor, response_name)

    # generate mean of response table with their plots
    d = []
    for i in range(0, len(predictors_list)):
        mean_resp_weighted = mean_of_response(
            df, predictors_list[i], response_name, figure=False
        )[0]
        mean_resp_unweighted = mean_of_response(
            df, predictors_list[i], response_name, figure=False
        )[1]
        mean_of_response(df, predictors_list[i], response_name, figure=True)
        plot_link = (
            predictors_list[i] + "-" + response_name + "_mean_of_response_plot.html"
        )
        d.append(
            {
                "predictor": predictors_list[i],
                "mean_squared_diff_weighted": mean_resp_weighted,
                "mean_squared_diff": mean_resp_unweighted,
                "plot": f'<a href="{plot_link}">Link</a>',
            }
        )
    mean_of_resp_table = pd.DataFrame(d)
    mean_of_resp_table.to_html("mean_of_response_table.html", escape=False)

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
        brute_force(df, var_list[i][0], var_list[i][1], response_name, figure=True)
        plot_link = var_list[i][0] + "-" + var_list[i][1] + "_brute_force_plot.html"
        d.append(
            {
                "predictor1": var1,
                "predictor2": var2,
                "correlation": corr,
                "brute_force_value": brute_force_value,
                "plot": f'<a href="{plot_link}">Link</a>',
            }
        )
    df = pd.DataFrame(d)
    df["correlation_absolute_value"] = abs(df["correlation"])
    df = df.sort_values(by="correlation_absolute_value", ascending=False)
    df.to_html("pair_wise_variable_table.html", escape=False)


if __name__ == "__main__":
    sys.exit(main())
