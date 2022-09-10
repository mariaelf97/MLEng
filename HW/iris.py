#!/usr/bin/env python3

import collections
import os
import sys
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

collections.Callable = collections.abc.Callable


# importing dataset as a pandas DF
# no header in the dataset so we define the headers manually
def import_iris_data():
    dirname = os.path.dirname(__file__)
    iris_data_path = os.path.join(dirname, "./iris.data")
    iris_data = pd.read_csv(
        iris_data_path,
        header=None,
        names=["sepal_length", "sepal_width", "petal_length", "petal_width", "class"],
    )
    return iris_data


def summary_stat():
    # changing pandas df to numpy array
    iris_data = import_iris_data()
    iris_data_numpy = iris_data.to_numpy()
    # calculating summary statistics max,min,median and average using numpy
    print(
        "maximum value for each attribute : ", np.max(iris_data_numpy[:, 0:3], axis=0)
    )
    print(
        "minimum value for each attribute : ", np.min(iris_data_numpy[:, 0:3], axis=0)
    )
    print(
        "Average value for each attribute : ", np.mean(iris_data_numpy[:, 0:3], axis=0)
    )
    print(
        "Median value for each attribute : ", np.median(iris_data_numpy[:, 0:3], axis=0)
    )


# visualizing attributes
def iris_plots():
    # scatter plot of sepal_width vs petal length, color shows the sepal length
    iris_data = import_iris_data()
    iris_scatter = px.scatter(
        iris_data, x="sepal_width", y="petal_length", color="sepal_length"
    )
    iris_scatter.show()
    # violin plot of sepal_width
    iris_violin = px.violin(iris_data, y="sepal_width")
    iris_violin.show()
    # bar plots for sepal_width and sepal_length
    iris_bar = px.bar(iris_data, x="sepal_width", y="sepal_length")
    iris_bar.show()
    # density heatmap of sepal_length vs sepal_width
    iris_heatmap = px.density_heatmap(iris_data, x="sepal_length", y="sepal_width")
    iris_heatmap.show()
    # sepal_length box plot
    iris_box_plot = px.box(iris_data, y="sepal_length")
    iris_box_plot.show()


def test_train():
    # RF model
    # defining features dataset without labels
    iris_data = import_iris_data()
    features = iris_data.iloc[:, 0:3].values
    # Labels are the values we want to predict
    labels = iris_data.iloc[:, 4].values
    # Split the data into training and testing sets
    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, test_size=0.2, random_state=0
    )
    return train_features, test_features, train_labels, test_labels


def predict_model():
    train_features, test_features, train_labels, test_labels = test_train()
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
    summary_stat()
    iris_plots()
    predict_model()


if __name__ == "__main__":
    sys.exit(main())
