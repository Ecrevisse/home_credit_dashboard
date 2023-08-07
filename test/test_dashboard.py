from contextvars import copy_context
from dash._callback_context import context_value
from dash._utils import AttributeDict

# Import the names of callback functions you want to test
from source.dashboard import update_graph


def test_update_callback():
    line = update_graph("France")
    assert line.data[0].y[-1] == 61083916
