from dash import Dash, html, dcc, callback, Output, Input
import dash_bootstrap_components as dbc
import dash_daq as daq
import plotly.express as px
import pandas as pd
import requests
import shap
import numpy as np
import plotly.graph_objects as go
import json
from io import BytesIO
import base64
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

api_url = "https://home-credit-webapp-api-api.azurewebsites.net"

df = pd.read_feather("./input/application_test.feather")
descirptions = pd.read_csv(
    "./input/HomeCredit_columns_description.csv", encoding="latin-1"
)
df_valid = pd.read_feather("./input/valid_cleaned.feather")


def fig_to_uri(in_fig, close_all=True, **save_args):
    print("fig to uri")
    out_img = BytesIO()
    in_fig.savefig(out_img, format="png", bbox_inches="tight", **save_args)
    if close_all:
        in_fig.clf()
        plt.close("all")
    out_img.seek(0)  # rewind file
    encoded = base64.b64encode(out_img.read()).decode("ascii").replace("\n", "")
    return f"data:image/png;base64,{encoded}"


def get_global_shap_uri():
    res = requests.get(api_url + "/shap_global").json()

    shap_val_local = res["shap_values"]
    base_value = res["base_value"]
    feat_values = res["data"]
    feat_names = res["feature_names"]

    explanation = shap.Explanation(
        np.array(shap_val_local),
        np.array(base_value),
        data=np.array(feat_values),
        feature_names=feat_names,
    )

    shap.plots.beeswarm(explanation, max_display=10, show=False)
    fig = plt.gcf()
    fig = fig_to_uri(fig)
    return fig


app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

server = app.server

app.layout = dbc.Container(
    [
        html.H1(
            children="Home Credit Dashboard",
            style={"textAlign": "center"},
        ),
        html.Br(),
        dcc.Dropdown(df.SK_ID_CURR, id="dropdown-selection"),
        html.Br(),
        dbc.Tabs(
            [
                dbc.Tab(
                    label="Home",
                    tab_id="Home",
                    children=[
                        html.H2(children="", id="client-id"),
                        dcc.Loading(
                            id="loading-1",
                            children=[
                                html.Div(
                                    [
                                        html.H3(
                                            children="Wait for request to finish",
                                            id="api-result",
                                        ),
                                    ]
                                )
                            ],
                            type="default",
                        ),
                        daq.Thermometer(
                            id="thermometer-score",
                            value=0,
                            min=0,
                            max=1,
                            scale={"custom": {"0": "0", "1": "1"}},
                            style={"marginBottom": "5px"},
                        ),
                        html.Br(),
                        html.H3(
                            children="Score",
                            style={"textAlign": "center", "color": "grey"},
                            id="score-label",
                        ),
                    ],
                ),
                dbc.Tab(
                    label="Client",
                    tab_id="Client",
                    children=[
                        dcc.Loading(
                            id="loading-2",
                            children=[
                                html.Div(
                                    [html.Img(id="waterfall", src="")],
                                    id="plot_div",
                                    # style={"width": "60rem", "overflow": "scroll"},
                                )
                            ],
                            type="default",
                        ),
                    ],
                ),
                dbc.Tab(
                    label="Dataset",
                    tab_id="Dataset",
                    children=[
                        ## la le graphique univarie
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dcc.Dropdown(
                                            id="dropdown-feature-1",
                                            options=[
                                                {"label": col, "value": col}
                                                for col in df_valid.columns
                                                if col != "SK_ID_CURR"
                                            ],
                                        ),
                                        dcc.Graph(id="univariate-1"),
                                    ],
                                    width=6,
                                ),
                                dbc.Col(
                                    [
                                        dcc.Dropdown(
                                            id="dropdown-feature-2",
                                            options=[
                                                {"label": col, "value": col}
                                                for col in df_valid.columns
                                                if col != "SK_ID_CURR"
                                            ],
                                        ),
                                        dcc.Graph(id="univariate-2"),
                                    ],
                                    width=6,
                                ),
                            ],
                        ),
                        ## la le graphique bivarie
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        dcc.Dropdown(
                                                            id="dropdown-feature-bi-1-1",
                                                            options=[
                                                                {
                                                                    "label": col,
                                                                    "value": col,
                                                                }
                                                                for col in df_valid.columns
                                                                if col != "SK_ID_CURR"
                                                            ],
                                                        ),
                                                    ],
                                                    width=6,
                                                ),
                                                dbc.Col(
                                                    [
                                                        dcc.Dropdown(
                                                            id="dropdown-feature-bi-1-2",
                                                            options=[
                                                                {
                                                                    "label": col,
                                                                    "value": col,
                                                                }
                                                                for col in df_valid.columns
                                                                if col != "SK_ID_CURR"
                                                            ],
                                                        ),
                                                    ],
                                                    width=6,
                                                ),
                                            ],
                                        ),
                                        dcc.Graph(id="bivariate-1"),
                                    ],
                                    width=6,
                                ),
                                dbc.Col(
                                    [
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        dcc.Dropdown(
                                                            id="dropdown-feature-bi-2-1",
                                                            options=[
                                                                {
                                                                    "label": col,
                                                                    "value": col,
                                                                }
                                                                for col in df_valid.columns
                                                                if col != "SK_ID_CURR"
                                                            ],
                                                        ),
                                                    ],
                                                    width=6,
                                                ),
                                                dbc.Col(
                                                    [
                                                        dcc.Dropdown(
                                                            id="dropdown-feature-bi-2-2",
                                                            options=[
                                                                {
                                                                    "label": col,
                                                                    "value": col,
                                                                }
                                                                for col in df_valid.columns
                                                                if col != "SK_ID_CURR"
                                                            ],
                                                        ),
                                                    ],
                                                    width=6,
                                                ),
                                            ],
                                        ),
                                        dcc.Graph(id="bivariate-2"),
                                    ],
                                    width=6,
                                ),
                            ],
                        ),
                    ],
                ),
                dbc.Tab(
                    label="Global",
                    tab_id="Global",
                    children=[
                        dcc.Loading(
                            id="loading-3",
                            children=[
                                html.Div(
                                    [
                                        html.Img(
                                            id="beeswarm", src=get_global_shap_uri()
                                        )
                                    ],
                                    id="plot_div-2",
                                    # style={"width": "60rem", "overflow": "scroll"},
                                )
                            ],
                            type="default",
                        ),
                    ],
                ),
            ],
            id="tabs",
            active_tab="Home",
        ),
    ]
)


@callback(
    [
        Output("client-id", "children"),
        Output("api-result", "children"),
        Output("thermometer-score", "value"),
        Output("thermometer-score", "color"),
        Output("thermometer-score", "scale"),
        Output("score-label", "children"),
        Output("score-label", "style"),
        Output("waterfall", "src"),
    ],
    Input("dropdown-selection", "value"),
)
def update_api(value):
    print("update local")
    if value is None:
        return (
            "Please select a client ID",
            "",
            0,
            "grey",
            {"custom": {"0": "0", "1": "1"}},
            "Score",
            {"textAlign": "center", "color": "grey"},
            "",
        )
    res = requests.post(api_url + "/predict", json={"client_id": value}).json()
    accepted = res[0]
    score = 1 - res[1]
    threshold = 1 - res[2]

    summary = requests.post(api_url + "/shap_local", json={"client_id": value}).json()

    shap_val_local = summary["shap_values"]
    base_value = summary["base_value"]
    feat_values = summary["data"]
    feat_names = summary["feature_names"]

    explanation = shap.Explanation(
        np.reshape(np.array(shap_val_local, dtype="float"), (1, -1)),
        base_value,
        data=np.reshape(np.array(feat_values, dtype="float"), (1, -1)),
        feature_names=feat_names,
    )

    fig = shap.waterfall_plot(explanation[0], max_display=10, show=False)
    fig = fig_to_uri(fig)

    return (
        f"Client ID : {value}",
        "Credit Granted" if accepted == 0 else "Credit Denied",
        score,
        "green"
        if score > threshold
        else "orange"
        if score > (threshold / 2)
        else "red",
        {
            "custom": {"0": "0", threshold: f"{threshold} - Granted", "1": "1"},
            "interval": 0.01,
            "labelInterval": 100,
        },
        f"{score:.2f}",
        {
            "textAlign": "center",
            "color": "green"
            if score > threshold
            else "orange"
            if score > (threshold / 2)
            else "red",
        },
        fig,
    )


def get_fig_univariate(value):
    fig = px.strip(
        df_valid,
        x="TARGET",
        y=value,
        color="TARGET",
    )
    return fig


def get_fig_bivariate(value1, value2):
    fig = px.scatter(
        df_valid,
        x=value1,
        y=value2,
        color="TARGET",
        marginal_x="histogram",
        marginal_y="histogram",
    )
    return fig


@callback(
    Output("univariate-1", "figure"),
    Input("dropdown-feature-1", "value"),
)
def update_univariate(value):
    print("update univariate")
    if value is None:
        return {}
    return get_fig_univariate(value)


@callback(
    Output("univariate-2", "figure"),
    Input("dropdown-feature-2", "value"),
)
def update_univariate(value):
    print("update univariate")
    if value is None:
        return {}
    return get_fig_univariate(value)


@callback(
    Output("bivariate-1", "figure"),
    Input("dropdown-feature-bi-1-1", "value"),
    Input("dropdown-feature-bi-1-2", "value"),
)
def update_bivariate(value1, value2):
    print("update bivariate")
    if value1 is None or value2 is None:
        return {}

    return get_fig_bivariate(value1, value2)


@callback(
    Output("bivariate-2", "figure"),
    Input("dropdown-feature-bi-2-1", "value"),
    Input("dropdown-feature-bi-2-2", "value"),
)
def update_bivariate(value1, value2):
    print("update bivariate")
    if value1 is None or value2 is None:
        return {}

    return get_fig_bivariate(value1, value2)


if __name__ == "__main__":
    api_url = "http://127.0.0.1:8000"
    app.run(debug=True)

# il manque deux graph ou l'on slect une feature, et on voit la distribution de la feature ainsi que l'emplacement du client
# idem mais avec un graph d'analyse bivariee avec un degrade de couleur en fonction du score
