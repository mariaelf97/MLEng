import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from define_data_type import define_cat


def predictor_response_plots(dataset, predictor, response):
    predictor_cat = define_cat(dataset, predictor)
    response_cat = define_cat(dataset, response)
    # res = cat - pred = cat
    if response_cat:
        if predictor_cat:
            cat_freq = pd.crosstab(index=dataset[predictor], columns=dataset[response])
            fig = go.Figure(data=go.Heatmap(z=cat_freq, zauto=True))
            fig.update_layout(
                title="Categorical Predictor by Categorical Response",
                xaxis_title="Response=" + response,
                yaxis_title="Predictor=" + predictor,
            )
            with open(
                predictor + "-" + response + "_predictor_response_plot.html", "w"
            ) as output_file:
                output_file.write(fig)

        # res = cat - pred = numeric
        else:
            fig = px.histogram(
                dataset,
                x=dataset[predictor],
                color=dataset[response],
                hover_data=dataset.columns,
            )
            fig.update_layout(
                title="Continuous Predictor by Categorical Response",
                xaxis_title="Response=" + response,
                yaxis_title="Predictor=" + predictor,
            )
            with open(
                predictor + "-" + response + "_predictor_response_plot.html", "w"
            ) as output_file:
                output_file.write(fig)

    # res = numeric - pred = cat
    else:
        if predictor_cat:
            fig = go.Figure(
                data=go.Violin(
                    y=dataset[response],
                    x=dataset[predictor],
                    fillcolor="lightseagreen",
                    opacity=0.6,
                    x0=response,
                )
            )
            fig.update_layout(
                yaxis_zeroline=False,
                title="Categorical Predictor by Numeric Response",
                xaxis_title="Response=" + response,
                yaxis_title="Predictor=" + predictor,
            )
            with open(
                predictor + "-" + response + "_predictor_response_plot.html", "w"
            ) as output_file:
                output_file.write(fig)
        # res = numeric - pred = numeric
        else:
            fig = px.scatter(x=dataset[predictor], y=dataset[response], trendline="ols")
            fig.update_layout(
                title="Continuous Response by Continuous Predictor",
                xaxis_title="Response=" + response,
                yaxis_title="Predictor=" + predictor,
            )
            with open(
                predictor + "-" + response + "_predictor_response_plot.html", "w"
            ) as output_file:
                output_file.write(fig)
