"""Bokeh server entry point: python -m bokeh serve scripts/run_server.py --show."""

from bokeh.io import curdoc

from extensions.analytics.dashboard import build_document

build_document(curdoc())
