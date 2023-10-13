import torch
from rdkit import Chem
import pandas as pd
import random
from typing import Tuple, List
from rlsync.environments.two_synthon_completion.mfp_limited import MorganFingerprintEnv
from rlsync.utils.data import load_synthons_reaction_center_json, load_synthons_reaction_center
import os
import sys
from argparse import ArgumentParser

random.seed(1995)

ap = ArgumentParser()
ap.add_argument("--data", "-d", type=str, required=True, help="training dataset for the extraction of ground truth and random episodes (RLSync JSON Format)")
ap.add_argument("--output", "-o", type=str, default="data/3step-base", help="path of output directory")
args = ap.parse_args()

training_all = list(load_synthons_reaction_center_json(args.data))
training = [[i] + training_all[i] for i in range(len(training_all))]

os.makedirs(args.output, exist_ok=True)

env = MorganFingerprintEnv({
    "can_change_existing": False,
    "can_multi_bond_ring_atom": False,
    "seed": 1996,
    "step_limit": 3,
    "alternating": False,
    "use_gpu": False
})

def backwalk(reactant, stepmax=3) -> List:
    reference = Chem.MolFromSmiles(reactant, sanitize=False)
    added = [a for a in reference.GetAtoms() if a.GetAtomMapNum() == 0]
    if len(added) > stepmax:
        return None
    paths = []
    if len(added) == 0:
        return [[Chem.MolToSmiles(reference, isomericSmiles=False)]]
    for a in added:
        editable = Chem.RWMol(reference)
        editable.RemoveAtom(a.GetIdx())
        smiles = Chem.MolToSmiles(editable, isomericSmiles=False)
        if "." not in smiles:
            subsequent = backwalk(smiles)
            for seq in subsequent:
                paths.append([Chem.MolToSmiles(reference, isomericSmiles=False)] + seq)
    return paths

def generate_trajectories(reactants, stepmax=3) -> List[Tuple[List, List]]:
    r1, r2 = reactants.split(".")
    b1 = backwalk(r1, stepmax=stepmax)
    b2 = backwalk(r2, stepmax=stepmax)
    if b1 is None or b2 is None:
        return [None]
    paths1 = [list(reversed(b)) for b in b1]
    paths2 = [list(reversed(b)) for b in b2]
    return [(p1, p2) for p1 in paths1 for p2 in paths2]

def genNtraj_pos(reaction, n=1, stepmax=3):
    idx = reaction.pop(0)
    if os.path.exists("%s/rpb_%05d.pt" % (args.output, idx)):
        sys.stdout.write("%05d positive\talready completed\n" % idx)
        sys.stdout.flush()
        return
    random.seed(1995+idx)
    reactants = reaction[3]
    trajectpop = generate_trajectories(reactants, stepmax=stepmax)
    trajectsamp = random.sample(trajectpop, k=min(n, len(trajectpop)))
    for i in range(len(trajectsamp)):
        trajectories = trajectsamp[i]
        rpb = []
        if trajectories is None:
            sys.stdout.write("Trajectory is none. "+reactants+"\n")
            sys.stdout.flush()
            return
        trajectories = list(trajectories)
        maxlen = max(len(trajectories[0]), len(trajectories[1]))
        # print(maxlen)
        if maxlen <= stepmax + 1:
            product = reaction[2]
            reaction_centers = reaction[1]
            synthons = reaction[0]
            t1full = trajectories[0] + [trajectories[0][-1]]*(stepmax + 1 - len(trajectories[0]))
            t2full = trajectories[1] + [trajectories[1][-1]]*(stepmax + 1 - len(trajectories[1]))
            obs = env.reset(1995+idx, {"reaction": [
                synthons,
                reaction_centers,
                product,
                reactants
            ]})
            done = False
            # Pop initial synthons (index 0) during step 0
            t1full.pop(0)
            t2full.pop(0)
            while not done:
                action = {
                    "synthon_1": Chem.MolFromSmiles(t1full.pop(0)),
                    "synthon_2": Chem.MolFromSmiles(t2full.pop(0))
                }
                new_obs, rew, new_done, _info = env.step(action)
                done = new_done
                rpb_entry = (obs, rew, new_obs, done)
                obs = new_obs
                rpb.append(rpb_entry)
        torch.save(rpb, "%s/rpb_%05d_%d.pt" % (args.output,idx,i))
    sys.stdout.write("%05d positive\tdone\n" % idx)
    sys.stdout.flush()

def gen1traj_neg(row, n=1):
    for k in range(n):
        random.seed(1995+row[0]+k)
        rpb = []
        obs = env.reset(1995+row[0]+k, {"reaction": row[1:]})
        done = False
        while not done:
            possible = env.get_possible_actions()
            action = {
                "synthon_1": random.choice(possible["synthon_1"]),
                "synthon_2": random.choice(possible["synthon_2"])
            }
            new_obs, rew, new_done, _info = env.step(action)
            done = new_done
            rpb_entry = (obs, rew, new_obs, done)
            obs = new_obs
            rpb.append(rpb_entry)
        if rpb[-1][1]["synthon_1"] == 1:
            torch.save(rpb, "%s/fsprpb_%05d_%d.pt" % (args.output, row[0], k))
        else:
            torch.save(rpb, "%s/nrpb_%05d_%d.pt" % (args.output, row[0], k))
    sys.stdout.write("%05d negative\tdone\n" % row[0])
    sys.stdout.flush()

# end_idx = 500*int(os.environ["SLURM_ARRAY_TASK_ID"])
# begin_idx = end_idx - 500
# end_idx = min(end_idx, len(training))
for reaction in training: #training[begin_idx:end_idx]:
    genNtraj_pos([x for x in reaction], n=2)
    gen1traj_neg([x for x in reaction], n=4)
