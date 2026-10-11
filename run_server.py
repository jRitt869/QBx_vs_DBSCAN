"""Bokeh server entry point: bokeh serve run_server.py --show."""

from bokeh.io import curdoc

from track_clusters.viewer import build_document

build_document(curdoc())
