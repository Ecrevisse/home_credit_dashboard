from dash import Dash, html, dcc, callback, Output, Input
import dash_bootstrap_components as dbc
import dash_daq as daq
import plotly.express as px
import pandas as pd
import requests

api_url = "https://home-credit-webapp-api-api.azurewebsites.net/predict"

df = pd.read_feather("./input/application_test.feather")
descirptions = pd.read_csv(
    "./input/HomeCredit_columns_description.csv", encoding="latin-1"
)

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

server = app.server

app.layout = dbc.Container(
    [
        html.H1(
            children="Home Credit Dashboard",
            style={"textAlign": "center"},
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dcc.Dropdown(df.SK_ID_CURR, id="dropdown-selection"),
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
                    ]
                ),
                dbc.Col(
                    [
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
                    ]
                ),
                dbc.Col([]),
            ]
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
    ],
    Input("dropdown-selection", "value"),
)
def update_api(value):
    if value is None:
        return (
            "Please select a client ID",
            "",
            0,
            "grey",
            {"custom": {"0": "0", "1": "1"}},
            "Score",
            {"textAlign": "center", "color": "grey"},
        )
    res = requests.post(api_url, json={"client_id": value}).json()
    accepted = res[0]
    score = 1 - res[1]
    threshold = 1 - res[2]
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
    )


if __name__ == "__main__":
    api_url = "http://127.0.0.1:8000/predict"
    app.run(debug=True)
