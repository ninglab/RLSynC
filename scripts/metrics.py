#!/usr/bin/env python3
import pandas as pd
from rlsync.utils.forward_synthesis import ForwardSynthesis
import argparse
import json
import numpy as np
import sys
import datetime
import re
import itertools
import torch

import rdkit
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors

def similarity2(a, b, sim_type="binary"):
    a = a.replace("\n","")
    b = b.replace("\n","")
    a = re.sub(r'\.$', '', a)
    b = re.sub(r'\.$', '', b)
    if a.count(".") == b.count(".") and a.count(".") == 1:
        a1, a2 = a.split(".")
        b1, b2 = b.split(".")
        s11 = similarity(a1,b1, sim_type="binary")
        s22 = similarity(a2,b2, sim_type="binary")
        s12 = similarity(a1,b2, sim_type="binary")
        s21 = similarity(a2,b1, sim_type="binary")
        if None in [s11, s22, s12, s21]:
            return None
        simpairs = [
            (s11+s22)/2,
            (s12+s21)/2
        ]
        return max(simpairs)
    return similarity(a,b, sim_type="binary")

def similarity(a, b, sim_type="count"):
    if a is None or b is None: 
        return None
    amol = Chem.MolFromSmiles(a)
    bmol = Chem.MolFromSmiles(b)
    try:
        if amol is None:
            amol = Chem.MolFromSmiles(a, sanitize=False)
            if amol is None:
                return None
            Chem.SanitizeMol(amol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
            amol.UpdatePropertyCache()
            Chem.FastFindRings(amol)
        if bmol is None:
            bmol = Chem.MolFromSmiles(b, sanitize=False)
            if bmol is None:
                return None
            Chem.SanitizeMol(bmol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
            amol.UpdatePropertyCache()
            Chem.FastFindRings(bmol)
    except Exception as e:
        # Valence error
        return None
    if sim_type == "binary":
        fp1 = AllChem.GetMorganFingerprintAsBitVect(amol, 2, nBits=2048, useChirality=False)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(bmol, 2, nBits=2048, useChirality=False)
    else:
        fp1 = AllChem.GetMorganFingerprint(amol, 2, useChirality=False)
        fp2 = AllChem.GetMorganFingerprint(bmol, 2, useChirality=False)
    sim = DataStructs.TanimotoSimilarity(fp1, fp2)
    return sim

def openjson(x):
    try:
        with open(x, "r") as f:
            return json.load(f)
    except Exception as e:
        return None

ap = argparse.ArgumentParser()
ap.add_argument("--results-csv", "-r", type=str, required=True, help="CSV containing results from RLSynC")
ap.add_argument("--oracle", "-o", type=openjson, required=False, default=None, help="Specify an existing full metrics JSON file to avoid recomputing forward synthesis")
ap.add_argument("--valonly", "-v", action="store_true", help="Run validation metrics only")
ap.add_argument("--testdata", "-t", type=str, default="data/3test.json", help="original testing dataset (RLSynC JSON format)")
ap.add_argument("--moltransformer", "-m", type=str, default="data/mt/MIT_mixed_augm_model_average_20.pt", help="molecular transformer model file (*.pt)")
ap.add_argument("--gpu", action="store_true", help="set this flag to use GPU")
args = ap.parse_args()

results = pd.read_csv(args.results_csv)

fs = None
if args.oracle is not None:
    # Using for canonicalization, no need to use GPU
    fs = ForwardSynthesis(args.moltransformer,
        n_best=5,
        beam_size=5,
        use_gpu=False
    )
else:
    fs = ForwardSynthesis(args.moltransformer,
        n_best=5,
        beam_size=5,
        use_gpu=args.gpu
    )

reactions = {}
rewards = {}
exacts = {}

rowitems = results.iterrows()

maxn = 0

for rowitem in rowitems:
    row = rowitem[1]
    idx = str(int(row["idx"]))
    print(row)
    if idx not in reactions:
        reactions[idx] = []
        rewards[idx] = []
        exacts[idx] = []
    sys.stdout.write(datetime.datetime.now().isoformat()+"\tRow Item %05d\n" % int(idx))
    sys.stdout.flush()
    if row["predict_smiles"] == "None" or "." not in str(row["predict_smiles"]):
        rewards[idx].append(0)
    else:
        reactions[idx].append(row["predict_smiles"])
        if args.oracle is not None or args.valonly:
            continue
        synthons = row["predict_smiles"].split(".")
        r1, r2 = row["reactant_smiles"].split(".")
        r1c = fs.canonicalize(r1)
        r2c = fs.canonicalize(r2)
        remainder = []
        try:
            s1c = fs.canonicalize(synthons[0])
            s2c = fs.canonicalize(synthons[1])
            remainder = []
            if len(synthons) > 2:
                remainder = [fs.canonicalize(s) for s in synthons[2:]]
            else:
                if (s1c == r2c) and (s2c == r1c):
                    print("Exact match for index %d" % int(idx))
                    rewards[idx].append(1)
                    exacts[idx].append(1)
                    continue
                elif (s2c == r2c) and (s1c == r1c):
                    rewards[idx].append(1)
                    exacts[idx].append(1)
                    print("Exact match for index %d" % int(idx))
                    continue
        except Exception as e:
            rewards[idx].append(0)
            exacts[idx].append(0)
            print("Synthon valence error for index %d" % int(idx))
            continue
        product = fs.canonicalize(row["product"])
        with torch.no_grad():
            rank, _score = fs.check_in_top_n(product, s1c, s2c, *remainder)
        rewards[idx].append(int(rank > 0))
        exacts[idx].append(0)

if args.oracle is not None:
    rewards = args.oracle["rewards"]
    exacts = args.oracle["exacts"]

fulltest = []
with open(args.testdata, "r") as f:
    fulltest = json.load(f)

indexref = [str(k) for k in range(len(rewards))]

maxn = max([len(results) for results in reactions.values()])
diversity = {}
diversen2 = []
diversen = []
diversel = []
dclists = []
dclists2 = []
for topn in range(1,11):
    diverse_counts = []
    diverse_lcounts = []
    diverse_counts_2 = []
    dclists = []
    dclists2 = []
    for idx in indexref:
        preds = reactions.get(idx, [])[:topn]
        dclist = []
        dclist2 = []
        diverse_count = 0
        diverse_count_2 = 0
        dlog_count = 0
        previous = []
        for i in range(len(preds)):
            if rewards[idx][i]:
                if len(previous) == 0:
                    dclist.append(0)
                    dclist2.append(0)
                else:
                    sims2 = [similarity2(preds[i], prev, "binary") for prev in previous]
                    sims2 = [s for s in sims2 if s is not None]
                    if len(sims2) != 0:
                        diverse_count_2 += (1.0 - max(sims2))
                        dclist2.append(1.0-max(sims2))
                previous.append(preds[i])
        dclists2.append(dclist2)
        diverse_counts_2.append(diverse_count_2)
    diversen2.append(diverse_counts_2)

with open(args.results_csv+"___full_metrics.json", "w") as f:
    json.dump({
        "rewards": rewards,
        "exacts": exacts,
        "diverse_counts": diversen2,
        "diversity": [float(np.mean(d)) for d in diversen2],
        "dissim_lists": dclists2,
        "ndcg": np.mean([[np.sum([rewards[str(idx)][k]/np.log2(2+k) for k in range(n+1)]) / np.sum([1/np.log2(2+k) for k in range(n+1)]) for idx in range(len(fulltest))] for n in range(10)], axis=1).tolist(),
        "exact_match_accuracy": np.mean([[int(any(exacts[k][:(j+1)])) for j in range(maxn)] for k in rewards], axis=0).tolist(),
        "anyreward": np.mean([[int(any(rewards[k][:(j+1)])) for j in range(maxn)] for k in rewards], axis=0).tolist(),
        "allreward": np.mean([[int(all(rewards[k][:(j+1)])) for j in range(maxn)] for k in rewards], axis=0).tolist(),
        "avgreward": np.mean([[np.mean(rewards[k][:(j+1)]) for j in range(maxn)] for k in rewards], axis=0).tolist(),
    }, f)
