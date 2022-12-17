import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train_test_split_func(dataset, predictor_list, response_name):

    train_dataset = dataset[dataset["local_date"] < "2011-06-02"]
    test_dataset = dataset[dataset["local_date"] >= "2011-06-02"]
    train_features = train_dataset.drop([response_name, "local_date"], axis=1)
    test_features = test_dataset.drop([response_name, "local_date"], axis=1)
    train_labels = train_dataset[response_name]
    test_labels = test_dataset[response_name]
    return train_features, test_features, train_labels, test_labels


def random_forest_var_imp(dataset, predictor_list, response):
    train_features, test_features, train_labels, test_labels = train_test_split_func(
        dataset, predictor_list, response
    )
    feature_names = dataset[predictor_list].columns
    forest = RandomForestClassifier(random_state=0)
    forest.fit(train_features, train_labels)
    importances = forest.feature_importances_
    forest_importances = pd.Series(importances, index=feature_names)
    return forest_importances.sort_values(ascending=False)


def predict_model(dataset, predictor_list, response_name, RF=True):
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
