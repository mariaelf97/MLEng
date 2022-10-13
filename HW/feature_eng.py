import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# user input to receive dataset
# https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
dataset = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
)
# specifing the response variable
response_name = " <=50K"
# Don't include response in other variables, we don't want to plot response vs response.
col_name = [col for col in dataset.columns if col != response_name]


def predictor_response_plots(col_name):

    if (dataset[response_name].dtypes == "object") & (
        dataset[col_name].dtypes == "object"
    ):
        # Heatmap ( categorical variables vs categorical response)
        cat_freq = pd.crosstab(index=dataset[col_name], columns=dataset[response_name])
        sns.heatmap(data=cat_freq, annot=True, fmt=".1f")
        plt.show()
    if (dataset[response_name].dtypes == "object") & (
        dataset[col_name].dtypes == "int64"
    ):
        # first convert the dataset to wide format
        sns.kdeplot(
            data=dataset,
            x=col_name,
            hue=response_name,
            cut=0,
            fill=True,
            common_norm=False,
            alpha=1,
        )
        plt.show()

    if (dataset[response_name].dtypes == "int64") & (
        dataset[col_name].dtypes == "int64"
    ):
        plt.scatter(dataset[col_name], dataset[response_name])
        plt.show()

    if (dataset[response_name].dtypes == "int64") & (
        dataset[col_name].dtypes == "object"
    ):
        sns.violinplot(x=dataset[col_name], y=dataset[response_name])
        plt.show()


def main():
    for i in range(1, len(col_name)):
        predictor_response_plots(col_name[i])


if __name__ == "__main__":
    sys.exit(main())
