from contextvars import copy_context
from dash._callback_context import context_value
from dash._utils import AttributeDict

from source.dashboard import update_api


def test_update_api_callback_empty():
    none_ret = update_api(None)

    assert none_ret == (
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


def test_update_api_callback():
    ret = update_api(365820)

    assert ret[0:-1] == (
        "Client ID : 365820",
        0,
        "Credit Granted",
        0.9508958301554126,
        "green",
        {
            "custom": {"0": "0", 0.85: "0.85 - Granted", "1": "1"},
            "interval": 0.01,
            "labelInterval": 100,
        },
        "0.95",
        {"textAlign": "center", "color": "green"},
    )
