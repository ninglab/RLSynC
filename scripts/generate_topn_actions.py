import torch
from rdkit import Chem
import pandas as pd
import random
from typing import Tuple, List
from rlsync.agents.rlsync.agent import DQNAgent
from rlsync.agents.rlsync.model import DQNModel
from rlsync.agents.rlsync.trainer import Trainer
from rlsync.environments.two_synthon_completion.mfp_limited import MorganFingerprintEnv
from rlsync.utils.data import load_synthons_reaction_center_json
import os
import sys
import numpy as np
from argparse import ArgumentParser

ap = ArgumentParser()
ap.add_argument("--name", "-n", type=str, required=True, help="name for data generation run")
ap.add_argument("--dataset", "-d", type=str, default="data/3train.json", help="dataset for training")
ap.add_argument("--model", "-m", type=str, required=True, help="model parameter *.pt file")
ap.add_argument("-k", type=int, default=3, help="top k actions explored at each step")
ap.add_argument("-N", type=int, default=5, help="top N reactions selected at final step")
ap.add_argument("--gpu", action="store_true", help="set flag to use GPU")
args = ap.parse_args()
np.random.seed(1996)
torch.manual_seed(1996)

gamma = 0.95
device = "cpu"
if args.gpu:
    device = "cuda"

os.makedirs(f"data/{args.name:s}", exist_ok=True)

eval_trainer = Trainer({
        "can_change_existing": False,
        "can_multi_bond_ring_atom": False,
        "seed": 1996,
        "step_limit": 3,
        "alternating": False,
        "use_gpu": args.gpu
    }, name=args.model, # lr=1e-5,
        epsilon_start=0, epsilon_end=0, device=device,
        training_data_json="data/debug.json", validation_data_json="data/debug.json")

model = torch.load(args.model, map_location=torch.device(device))
model.eval()
eval_trainer.agents["synthon_1"].model = model
eval_trainer.agents["synthon_2"].model = model

training_all = [x for x in load_synthons_reaction_center_json(args.dataset)]
training = [[i] + training_all[i] for i in range(len(training_all))]
# task = int(os.environ["SLURM_ARRAY_TASK_ID"])
# end_idx = 26*int(os.environ["SLURM_ARRAY_TASK_ID"])
# begin_idx = end_idx - 26
# end_idx = min(end_idx, len(training))
# idx = begin_idx
for reaction in training: #training[begin_idx:end_idx]:
    idx = reaction[0]
    envopts = {"reaction": reaction[1:]}
    results = eval_trainer.perform_multi_agent_q_beam_search(envopts, beam_size=args.k)
    topn = sorted(results, key=lambda x: -x[-1])[:args.N]
    for i in range(len(topn)):
        reward, actions, inputs, preds, rpb, q_value = topn[i]
        posneg = "nrpb"
        if reward["synthon_1"] == 1:
            posneg = "fsprpb"
        filename = f"data/{args.name:s}/{posneg:s}_{idx:05d}_{i:03d}.pt"
        torch.save(rpb, filename)
    
