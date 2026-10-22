"""Reusable visual models and positional table data, independently of the controller."""

import pandas as pd
from bokeh.core.validation import check_integrity
from bokeh.document import Document

from extensions.analytics.layouts import DashboardLayout, EvaluationLayout
from extensions.analytics.presentation import set_table, table


def test_visual_groups_are_independent_and_inert_until_callbacks_are_attached():
    first = EvaluationLayout(
        algorithm="qbx",
        profiles=["first", "second"],
        selected_profile="first",
        run_options=[("1", "Run 1"), ("2", "Run 2")],
        selected_run="1",
    )
    second = EvaluationLayout(
        algorithm="qbx",
        profiles=["first", "second"],
        selected_profile="first",
        run_options=[("1", "Run 1"), ("2", "Run 2")],
        selected_run="1",
    )
    shell = DashboardLayout(["first", "second"])
    shell.tab_container.children = [first.layout]
    documents = [Document(), Document()]
    documents[0].add_root(shell.layout)
    documents[1].add_root(second.layout)

    shell.dataset_select.value = "second"
    first.profile_select.value = "second"
    first.run_select.value = "2"
    first.sweep_source.data = dict(eps=[0.5], mean=[0.8], low=[0.7], high=[0.9], runs=[3])
    assert shell.dataset_note.text == "<p></p>"
    assert all(value.text == "" for value in first.run_values.values())
    assert second.profile_select.value == "first"
    assert second.run_select.value == "1"
    assert second.sweep_source.data["eps"] == []
    assert not first.layout.references() & second.layout.references()
    assert all(not model.js_event_callbacks for model in shell.layout.references())
    for document in documents:
        assert not check_integrity(document.models).error
        document.to_json()


def test_table_updates_preserve_duplicate_column_labels_and_empty_selections():
    frame = pd.DataFrame([[1, "one"], [2, "two"]], columns=["value", "value"])
    widget = table(frame)
    assert [(column.field, column.title) for column in widget.columns] == [
        ("c0", "value"),
        ("c1", "value"),
    ]
    assert widget.source.data == {"c0": [1, 2], "c1": ["one", "two"]}
    set_table(widget, frame.iloc[:0])
    assert widget.source.data == {"c0": [], "c1": []}
    set_table(widget, frame.iloc[1:])
    assert widget.source.data == {"c0": [2], "c1": ["two"]}
