#!/bin/bash
# python3 scripts/generate_ground_truth_original.py -d data/debug.json
# python3 scripts/offline_train.py --name debug -d data/3step-base -e 2 -b 2 --validation data/debug.json
# python3 scripts/generate_agent_actions.py --name aug1 -d data/debug.json -m iterations/debug/pretrain_epoch_00001.pt
# python3 scripts/offline_train.py --name debug-aug1 -d data/3step-base data/aug1 -e 2 -b 2 --validation data/debug.json
# python3 scripts/generate_topn_actions.py --name topn1 -d data/debug.json -m iterations/debug-aug1/pretrain_epoch_00001.pt -k 2
# python3 scripts/offline_train.py --name debug-topn1 -d data/3step-base data/aug1 data/topn1 -e 2 -b 2 --validation data/debug.json
# python3 scripts/apply_agents.py --testdata data/debug.json -o debug.csv -n 10 -k 2 -m iterations/debug-topn1/pretrain_epoch_00001.pt
# python3 scripts/metrics.py -r debug.csv --testdata data/debug.json