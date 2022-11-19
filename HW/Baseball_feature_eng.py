import sys

import pandas as pd
import sqlalchemy
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def logistic_reg(dataset, predictor, response, pvalue=True):
    y = dataset[response]
    x = dataset[predictor]
    cons = sm.add_constant(x)
    logistic_regression_model = sm.Logit(y, cons)
    logistic_regression_model_fitted = logistic_regression_model.fit()
    # Get the stats
    t_value = round(logistic_regression_model_fitted.tvalues[1], 6)
    p_value = "{:.6e}".format(logistic_regression_model_fitted.pvalues[1])
    if pvalue:
        return p_value
    else:
        return t_value


def train_test_split_func(dataset, predictor_list, response_name):
    features = dataset[predictor_list]
    labels = dataset[response_name]
    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, test_size=0.2, random_state=0
    )
    return train_features, test_features, train_labels, test_labels


def predict_model(dataset, predictor_list, response_name):
    train_features, test_features, train_labels, test_labels = train_test_split_func(
        dataset, predictor_list, response_name
    )
    rf_model_pipeline = Pipeline(
        steps=[
            ("scaling", StandardScaler()),
            ("random_forest", RandomForestClassifier(n_estimators=20, random_state=0)),
        ]
    )
    rf_model_pipeline.fit(train_features, train_labels)
    rf_model_pipeline_predictions = rf_model_pipeline.predict(test_features)
    print("-----RF Model Prediction------")
    print(
        "confustion matrix is : ",
        confusion_matrix(test_labels, rf_model_pipeline_predictions),
    )
    print(
        "calssification report is :",
        classification_report(test_labels, rf_model_pipeline_predictions),
    )
    print(
        "Accuracy score is:", accuracy_score(test_labels, rf_model_pipeline_predictions)
    )

    KNN_model_pipeline = Pipeline(
        steps=[
            ("scaling", StandardScaler()),
            ("KNN", KNeighborsClassifier(n_neighbors=8)),
        ]
    )
    KNN_model_pipeline.fit(train_features, train_labels)
    KNN_model_pipeline_predictions = rf_model_pipeline.predict(test_features)
    print("-----KNN Model Prediction------")
    print(
        "confustion matrix is : ",
        confusion_matrix(test_labels, KNN_model_pipeline_predictions),
    )
    print(
        "calssification report is :",
        classification_report(test_labels, KNN_model_pipeline_predictions),
    )
    print(
        "Accuracy score is:",
        accuracy_score(test_labels, KNN_model_pipeline_predictions),
    )


def main():
    db_user = ""
    db_pass = ""  # pragma: allowlist secret
    db_host = "localhost"
    db_database = "baseball"
    connect_string = (
        f"mariadb+mariadbconnector://{db_user}:{db_pass}@{db_host}/{db_database}"
    )
    # pragma: allowlist secret

    sql_engine = sqlalchemy.create_engine(connect_string)

    query = """select
    bs.winner_home_or_away
    ,bc.team_id
    ,bc.game_id
    ,bc.homeTeam
    ,bc.awayTeam
    ,bc.Hit
    ,bc.atBat
    ,bc.Hit_By_Pitch
     ,bc.Home_run
     ,bc.Strikeout
     ,bc.Single
     ,bc.Double
     ,bc.Triple
     ,bc.Sac_Fly
     ,sum(bc.Home_run)/sum(nullif(bc.Hit,0)) as Home_run_per_hit
     from batter_counts bc, boxscore bs where bs.game_id=bc.game_id group by game_id, team_id;"""

    df = pd.read_sql_query(query, sql_engine)
    # creating a new value to filter home team
    df["H_A"] = [1 if x == 1 else 0 for x in df["homeTeam"]]
    df["winner"] = [1 if x == "H" else 0 for x in df["winner_home_or_away"]]
    # filter home team
    df_filtered = df[df.H_A == 1]
    dataset = df_filtered.dropna()
    del dataset["team_id"]
    del dataset["game_id"]
    del dataset["H_A"]
    del dataset["homeTeam"]
    del dataset["awayTeam"]
    del dataset["winner_home_or_away"]
    response_name = "winner"
    predictors_list = [col for col in dataset.columns if col != response_name]
    predictor_dataset = dataset[predictors_list]

    # Split dataset on predictors in list to categoricals and continuous
    numeric_columns = predictor_dataset.select_dtypes("number").columns
    categorical_columns = predictor_dataset.select_dtypes("object").columns

    from feature_eng import generate_tables

    # individual plots with correlation table
    num_num_cor_table = generate_tables(
        dataset,
        categorical_columns,
        numeric_columns,
        response_name,
        cat1=False,
        cat2=False,
    )
    num_num_cor_table.to_html("num_num_correlation_table.html", escape=False)

    # regression table
    d = []
    for i in range(0, len(predictors_list)):
        pvalue = logistic_reg(dataset, predictors_list[i], response_name, pvalue=True)
        tvalue = logistic_reg(dataset, predictors_list[i], response_name, pvalue=False)

        d.append(
            {"predictor1": predictors_list[i], "p-value": pvalue, "tvalue": tvalue}
        )
    regression_table = pd.DataFrame(d)

    regression_table.to_html("regression_table.html", escape=False)

    # RF variable importance
    from feature_eng import random_forest_var_imp

    print(random_forest_var_imp(dataset, predictors_list, response_name))
    # brute force
    from feature_eng import generate_brute_force

    num_num_brute_force_table = generate_brute_force(
        dataset,
        categorical_columns,
        numeric_columns,
        response_name,
        cat1=False,
        cat2=False,
    )
    num_num_brute_force_table.to_html("num_num_brute_force_table.html", escape=False)
    # diff in mean of response
    from feature_eng import make_clickable, mean_of_response

    d = []
    for i in range(0, len(predictors_list)):
        mean_resp = mean_of_response(
            dataset, predictors_list[i], response_name, figure=False
        )
        mean_of_response(dataset, predictors_list[i], response_name, figure=True)
        plot_link = predictors_list[i] + "_mean_response_plot.html"
        d.append(
            {
                "predictor1": predictors_list[i],
                "mean_of_response": mean_resp,
                "plot": plot_link,
            }
        )
    mean_of_resp_table = pd.DataFrame(d)
    mean_of_resp_table.style.format({"plot": make_clickable})
    mean_of_resp_table.to_html("mean_of_table.html", escape=False)

    # building RF and KNN model, testing it
    predict_model(dataset, predictors_list, response_name)
    # both models had the same accuracy


if __name__ == "__main__":
    sys.exit(main())
