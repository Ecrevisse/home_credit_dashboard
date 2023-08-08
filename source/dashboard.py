from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
import pandas as pd
import requests

api_url = "https://home-credit-webapp-api-api.azurewebsites.net/predict"

df = pd.read_feather("./input/application_test.feather")
descirptions = pd.read_csv(
    "./input/HomeCredit_columns_description.csv", encoding="latin-1"
)

# print(df.head())

app = Dash(__name__)

server = app.server

app.layout = html.Div(
    [
        html.H1(children="Home Credit Dashboard", style={"textAlign": "center"}),
        dcc.Dropdown(df.SK_ID_CURR, id="dropdown-selection"),
        dcc.Loading(
            id="loading-1",
            children=[
                html.Div(
                    [
                        html.H2(children="Wait for request to finish", id="api-result"),
                    ]
                )
            ],
            type="default",
        ),
        # html.H2(children="Wait for request to finish", id="api-result"),
    ]
)


@callback(Output("api-result", "children"), Input("dropdown-selection", "value"))
def update_graph(value):
    if value is None:
        return "Please select a client ID"
    return str(requests.post(api_url, json={"client_id": value}).json())


if __name__ == "__main__":
    app.run(debug=True)
