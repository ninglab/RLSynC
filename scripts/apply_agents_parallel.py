import torch
from rlsync.agents.rlsync.trainer import Trainer
from rlsync.agents.rlsync.model import DQNModel
from rlsync.utils.data import load_synthons_reaction_center_json
import numpy as np
import os
import pandas as pd
import sys
from argparse import ArgumentParser
from pqdm.processes import pqdm
import itertools

ap = ArgumentParser()
ap.add_argument("--model", "-m", type=str, required=True, help="path to model parameters (*.pt)")
ap.add_argument("--testdata", "-t", type=str, default="data/3test.json", help="path to data in rlsync JSON format (see JSON Format section in README.md)")
ap.add_argument("--output", "-o", type=str, default="rlsync_output.csv", help="desired output CSV file name")
ap.add_argument("-n", "-N", type=int, default=10, help="number of reactions to generate for each product")
ap.add_argument("-k", type=int, default=3, help="number of actions to explore for each step")
ap.add_argument("--gpu", action="store_true", help="set flag to use GPU")
ap.add_argument("--cpus", "-c", type=int, default=8, help="number of cpus to use")
args = ap.parse_args()

device = "cpu"
if args.gpu:
    device = "cuda"

np.random.seed(1996)
torch.manual_seed(1996)
testdata = [x for x in load_synthons_reaction_center_json(args.testdata) if len(x[0].split(".")) == 2]

def generate_predictions(self, n=10, beam_size=3, indices=None, cpus=1):
    for model in self.models:
        self.models[model].eval()
        self.agents[model].model.eval()
    val_len = len(self.validation_data)
    results = []
    for ep in range(val_len):
        print(f"Starting reaction #{ep:04d} of {val_len:04d}")
        sys.stdout.flush()
        envopts = {"reaction": self.validation_data[ep]}
        paths = self.perform_multi_agent_q_beam_search(envopts, beam_size=beam_size, both_q=True, calculate_reward=False)
        uniquepaths = {p[3]: p for p in paths}
        print(len(paths))
        sys.stdout.flush()
        # making sure to sort by -Q value so that largest Q values are first
        topn_paths = sorted(uniquepaths.keys(), key=lambda x: -uniquepaths[x][-1])[:min(n, len(uniquepaths.keys()))]
        for k in range(min(n, len(topn_paths))):
            predictions = topn_paths[k]
            p = uniquepaths[predictions]
            gt_reactants = p[2][3]
            gt_product = p[2][2]
            q = float(p[-1])
            res = {"predict_smiles": predictions, "reactant_smiles": gt_reactants, "product": gt_product, "q": q}
            if indices is not None:
                res["idx"] = indices[ep]
            results.append(res)
    return results

def make_predictions(epidx):
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
    eval_trainer.evaluate_beam = generate_predictions.__get__(eval_trainer)
    eval_trainer.validation_data = testdata
    eval_trainer.validation_data = [eval_trainer.validation_data[epidx]]
    # print(len(eval_trainer.validation_data))
    # task = int(os.environ["SLURM_ARRAY_TASK_ID"])
    # eval_trainer.validation_data = eval_trainer.validation_data[(task-1)*32:(task)*32]
    return eval_trainer.evaluate_beam(n=args.n, cpus=args.cpus, beam_size=args.k, indices=[epidx]) # indices=range((task-1)*32,(task)*32))
allresults = pqdm(range(len(testdata)), make_predictions, n_jobs=args.cpus)
print(allresults[:10])
results = itertools.chain.from_iterable(allresults)
df = pd.DataFrame(results)
df.to_csv(args.output, index=False)