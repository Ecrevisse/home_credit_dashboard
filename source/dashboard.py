from dash import Dash, html, dcc, callback, Output, Input, dash_table
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

if __name__ == "__main__":
    api_url = "http://127.0.0.1:8000"
else:
    api_url = "https://home-credit-webapp-api-api.azurewebsites.net"

df = pd.read_feather("./input/application_test.feather")
descriptions = pd.read_csv(
    "./input/HomeCredit_columns_description.csv", encoding="latin-1"
)

# for row in descriptions.iterrows():
#     print(row[1].Row, row[1].Description)

df_valid = pd.read_feather("./input/valid_cleaned.feather")

model_features = requests.get(api_url + "/model_features").json()


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
                        html.Br(),
                        html.H2(children="Home Credit Dashboard"),
                        html.Br(),
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
                        html.Br(),
                        html.H2(children="Interpretability Local"),
                        html.Br(),
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
                        html.Br(),
                        html.H2(children="Exploration of the dataset"),
                        html.Br(),
                        ## la le graphique univarie
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dcc.Dropdown(
                                            id="dropdown-feature-1",
                                            options=[
                                                {"label": col, "value": col}
                                                for col in model_features
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
                                                for col in model_features
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
                                                                for col in model_features
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
                                                                for col in model_features
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
                                                                for col in model_features
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
                                                                for col in model_features
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
                        html.Br(),
                        html.H2(children="Interpretability Global"),
                        html.Br(),
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
                # this tab contain a description of all the features
                dbc.Tab(
                    label="Help",
                    tab_id="Help",
                    children=[
                        html.Br(),
                        html.H2(children="Help"),
                        html.Br(),
                        html.H3(children="Description of the features"),
                        html.Br(),
                        html.Div(
                            [
                                dash_table.DataTable(
                                    id="table",
                                    columns=[
                                        {"name": "Row", "id": "Row"},
                                        {"name": "Description", "id": "Description"},
                                    ],
                                    data=[
                                        {
                                            "Row": row[1].Row,
                                            "Description": row[1].Description,
                                        }
                                        for row in descriptions.iterrows()
                                    ],
                                    style_cell={
                                        "textAlign": "left",
                                        "whiteSpace": "normal",
                                        "height": "auto",
                                    },
                                    style_data_conditional=[
                                        {
                                            "if": {"row_index": "odd"},
                                            "backgroundColor": "rgb(248, 248, 248)",
                                        }
                                    ],
                                    style_header={
                                        "backgroundColor": "rgb(230, 230, 230)",
                                        "fontWeight": "bold",
                                    },
                                )
                            ]
                        ),
                    ],
                ),
            ],
            id="tabs",
            active_tab="Home",
        ),
        dcc.Store(id="accepted"),
    ]
)


@callback(
    [
        Output("client-id", "children"),
        Output("accepted", "data"),
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
            0,
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
        accepted,
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


def get_fig_univariate(feature, client_id, accepted):
    fig = px.strip(
        df_valid,
        x="TARGET",
        y=feature,
        color="TARGET",
    )
    newnames = {"0.0": "Credit Granted", "1.0": "Credit Denied"}
    fig.for_each_trace(
        lambda t: t.update(
            name=newnames[t.name],
            legendgroup=newnames[t.name],
            hovertemplate=t.hovertemplate.replace(t.name, newnames[t.name]),
        )
    )
    if client_id is not None:
        client_info = requests.post(
            api_url + "/client_info",
            json={"client_id": client_id, "client_infos": [feature]},
        ).json()
        if client_info is None:
            return {}
        if client_info[0][0] is None:
            return {}
        fig.add_trace(
            go.Scatter(
                x=[0 if accepted == 0 else 1],
                y=[client_info[0][0]],
                mode="markers",
                marker=dict(color="orange", size=20),
                name="You",
            )
        )
    return fig


def get_fig_bivariate(value1, value2, client_id):
    fig = px.scatter(
        df_valid,
        x=value1,
        y=value2,
        color="TARGET",
        color_continuous_scale=["green", "red"],
        # marginal_x="histogram",
        # marginal_y="histogram",
    )
    if client_id is not None:
        client_info = requests.post(
            api_url + "/client_info",
            json={"client_id": client_id, "client_infos": [value1, value2]},
        ).json()
        if client_info is None:
            return {}
        if client_info[0][0] is None:
            return {}
        fig.add_trace(
            go.Scatter(
                x=[client_info[0][0]],
                y=[client_info[0][1]],
                mode="markers",
                marker=dict(color="orange", size=20),
                name="You",
            )
        )
    return fig


@callback(
    Output("univariate-1", "figure"),
    Input("dropdown-feature-1", "value"),
    Input("dropdown-selection", "value"),
    Input("accepted", "data"),
)
def update_univariate(feature, client_id, accepted):
    if feature is None:
        return {}
    return get_fig_univariate(feature, client_id, accepted)


@callback(
    Output("univariate-2", "figure"),
    Input("dropdown-feature-2", "value"),
    Input("dropdown-selection", "value"),
    Input("accepted", "data"),
)
def update_univariate(feature, client_id, accepted):
    if feature is None:
        return {}
    return get_fig_univariate(feature, client_id, accepted)


@callback(
    Output("bivariate-1", "figure"),
    Input("dropdown-feature-bi-1-1", "value"),
    Input("dropdown-feature-bi-1-2", "value"),
    Input("dropdown-selection", "value"),
)
def update_bivariate(value1, value2, client_id):
    if value1 is None or value2 is None:
        return {}

    return get_fig_bivariate(value1, value2, client_id)


@callback(
    Output("bivariate-2", "figure"),
    Input("dropdown-feature-bi-2-1", "value"),
    Input("dropdown-feature-bi-2-2", "value"),
    Input("dropdown-selection", "value"),
)
def update_bivariate(value1, value2, client_id):
    if value1 is None or value2 is None:
        return {}

    return get_fig_bivariate(value1, value2, client_id)


if __name__ == "__main__":
    app.run(debug=True)

# il manque deux graph ou l'on slect une feature, et on voit la distribution de la feature ainsi que l'emplacement du client
# idem mais avec un graph d'analyse bivariee avec un degrade de couleur en fonction du score
