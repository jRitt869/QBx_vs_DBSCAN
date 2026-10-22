"""Preserved internship map, reading an explicitly selected source database."""

from bokeh.io import curdoc

from historical.config.parser import build_parser
from historical.config.validate import ViewerInput, validate_cli
from historical.viewer import build_document

parser = build_parser("run_server")
args = validate_cli(parser, ViewerInput, parser.parse_args())
build_document(curdoc(), args.database, run_id=args.run_id, algorithm=args.algorithm)
