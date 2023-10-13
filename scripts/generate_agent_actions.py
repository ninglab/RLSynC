import torch
from rdkit import Chem
import pandas as pd
import random
from typing import Tuple, List
from rlsync.agents.rlsync.agent import DQNAgent
from rlsync.agents.rlsync.model import DQNModel
from rlsync.environments.two_synthon_completion.mfp_limited import MorganFingerprintEnv
from rlsync.utils.data import load_synthons_reaction_center_json
import os
import sys
import json
import numpy as np
from argparse import ArgumentParser

ap = ArgumentParser()
ap.add_argument("--name", "-n", type=str, required=True, help="name for data generation run")
ap.add_argument("--dataset", "-d", type=str, default="data/3train.json", help="training reactions for generation of agent-derived training data")
ap.add_argument("--model", "-m", type=str, required=True, help="*.pt model parameter file")
ap.add_argument("--gpu", action="store_true", help="set this flag to use GPU")
args = ap.parse_args()


np.random.seed(1996)
torch.manual_seed(1996)

name = args.name

env = MorganFingerprintEnv({
    "can_change_existing": False,
    "can_multi_bond_ring_atom": False,
    "seed": 1996,
    "step_limit": 3,
    "alternating": False,
    "use_gpu": args.gpu
})

device = "cpu"
if args.gpu:
    device="cuda"

model = torch.load(args.model, map_location=torch.device(device))
agent = DQNAgent(model, device=device)

training_all = list(load_synthons_reaction_center_json(args.dataset))
training = [[i] + training_all[i] for i in range(len(training_all))]

os.makedirs(f"data/{args.name:s}", exist_ok=True)

def gen1traj_agent(row, n=1):
    random.seed(1995+row[0])
    np.random.seed(1995+row[0])
    torch.manual_seed(1995+row[0])
    rpb = []
    obs = env.reset(1995+row[0], {"reaction": row[1:]})
    done = False
    while not done:
        possible = env.get_possible_actions()
        action = {
            "synthon_1": agent.select_action(obs["synthon_1"], possible["synthon_1"]),
            "synthon_2": agent.select_action(obs["synthon_2"], possible["synthon_2"])
        }
        new_obs, rew, new_done, _info = env.step(action)
        done = new_done
        rpb_entry = (obs, rew, new_obs, done)
        obs = new_obs
        rpb.append(rpb_entry)
    if rpb[-1][1]["synthon_1"] == 1:
        torch.save(rpb, f"data/{args.name:s}/fsprpb_{row[0]:05d}.pt")
    else:
        torch.save(rpb, f"data/{args.name:s}/nrpb_{row[0]:05d}.pt")
    sys.stdout.write("%05d agent\tdone\n" % row[0])
    sys.stdout.flush()

#end_idx = 640*int(os.environ["SLURM_ARRAY_TASK_ID"])
#begin_idx = end_idx - 640
#end_idx = min(end_idx, len(training))
for reaction in training: #training[begin_idx:end_idx]:
    gen1traj_agent(reaction)
