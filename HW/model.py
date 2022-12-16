import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train_test_split_func(dataset, predictor_list, response_name):
    features = dataset[predictor_list]
    labels = dataset[response_name]
    train = dataset[predictor_list]
    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, test_size=0.2, random_state=0, stratify=labels
    )
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
