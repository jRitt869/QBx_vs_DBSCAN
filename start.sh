#!/bin/sh
set -eu

export CLUSTER_DATA_DIR="${CLUSTER_DATA_DIR:-./output}"
python ./get_raw_data.py --output-dir "$CLUSTER_DATA_DIR" "$@"
exec bokeh serve ./run_server.py --address 0.0.0.0 --port 5006
