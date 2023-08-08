from contextvars import copy_context
from dash._callback_context import context_value
from dash._utils import AttributeDict

from source.dashboard import update_graph


def test_update_callback():
    must_be_1 = update_graph(100548)  # 100548 -> 1
    must_be_0 = update_graph(100791)  # 100791 -> 0
    assert must_be_0 == "0" and must_be_1 == "1"
