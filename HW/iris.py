import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# importing dataset as a pandas DF
# no header in the dataset so we define the headers manually
iris_data = pd.read_csv(
    "/Users/Maria/Downloads/iris.data",
    header=None,
    names=["sepal_length", "sepal_width", "petal_length", "petal_width", "class"],
)
# changing pandas df to numpy array
iris_data_numpy = iris_data.to_numpy()

# calculating summary statistics max,min,median and average using numpy
print("maximum value for each attribute : ", np.max(iris_data_numpy[:, 0:3], axis=0))
print("minimum value for each attribute : ", np.min(iris_data_numpy[:, 0:3], axis=0))
print("Average value for each attribute : ", np.mean(iris_data_numpy[:, 0:3], axis=0))
print("Median value for each attribute : ", np.median(iris_data_numpy[:, 0:3], axis=0))

# visualizing attributes
# scatter plot of sepal_width vs petal length, color shows the sepal length
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
# RF model
# defining features dataset without labels
features = iris_data.iloc[:, 0:3].values
# Labels are the values we want to predict
labels = iris_data.iloc[:, 4].values
# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(
    features, labels, test_size=0.2, random_state=0
)
# Feature scaling
scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
test_features = scaler.transform(test_features)
# training the algorithm
rf = RandomForestClassifier(n_estimators=20, random_state=0)
rf.fit(train_features, train_labels)
# Use the forest's predict method on the test data
predictions = rf.predict(test_features)
# evaluating the algorithm
print(confusion_matrix(test_labels, predictions))
print(classification_report(test_labels, predictions))
print(accuracy_score(test_labels, predictions))

# KNN classification
knn = KNeighborsClassifier(n_neighbors=8)
knn.fit(train_features, train_labels)
knn_predict = knn.predict(test_features)
knn.score(test_features, test_labels)
# evaluating the algorithm
print(confusion_matrix(test_labels, knn_predict))
print(classification_report(test_labels, knn_predict))
print(accuracy_score(test_labels, knn_predict))
