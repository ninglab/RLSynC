#!/bin/bash

# Use this after install or in the docker container
# to compute the results in the README.md file.
#
# Docker usage: docker run -v ${PWD}/data:/rlsync/data rlsync /rlsync/scripts/compute_results.sh
# Self-install usage: bash scripts/compute_results.sh
python3 scripts/apply_agents_parallel.py --testdata data/3test.json -o data/test.csv -c 8 -n 10 -k 3 -m data/rlsync/best.pt
python3 scripts/metrics.py -r data/test.csv --testdata data/3test.json
python3 scripts/keymetrics.py
