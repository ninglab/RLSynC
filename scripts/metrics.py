#!/usr/bin/env python3
import pandas as pd
from rlsync.utils.forward_synthesis import ForwardSynthesis
import argparse
import json
import numpy as np
import sys
import datetime
import itertools

import rdkit
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors

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
ap.add_argument("--simonly", "-s", action="store_true", help="Run similarity/diversity metrics only")
ap.add_argument("--valonly", "-v", action="store_true", help="Run validation metrics only")
ap.add_argument("--testdata", "-t", type=str, default="data/3test.json", help="original testing dataset (RLSynC JSON format)")
ap.add_argument("--moltransformer", "-m", type=str, default="data/mt/MIT_mixed_augm_model_average_20.pt", help="molecular transformer model file (*.pt)")
ap.add_argument("--gpu", action="store_true", help="set this flag to use GPU")
args = ap.parse_args()

results = pd.read_csv(args.results_csv)

fs = None
if args.simonly or (args.oracle is not None):
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
    idx = str(row["idx"])
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
        if not args.simonly:
            product = fs.canonicalize(row["product"])
            rank, _score = fs.check_in_top_n(product, s1c, s2c, *remainder)
            rewards[idx].append(int(rank > 0))
        exacts[idx].append(0)

if args.oracle is not None:
    rewards = args.oracle["rewards"]
    exacts = args.oracle["exacts"]

fulltest = []
with open(args.testdata, "r") as f:
    fulltest = json.load(f)

invalid = {}
invalid_counts = {}
for idxint in range(len(fulltest)):
    idx = str(idxint)
    invalid[idx] = {}
    if idx not in reactions:
        continue
    for i in range(10):
        try:
            s1c = fs.canonicalize(synthons[0])
            if s1c is None:
                invalid[idx][str(i)] = invalid[idx][str(i)] + ["Error reading SMILES for synthon 1"]
        except Exception as e:
            invalid[idx][str(i)] = invalid[idx][str(i)] + ["Error reading SMILES for synthon 1: "+str(e)]
        try:
            s2c = fs.canonicalize(synthons[1])
            if s2c is None:
                invalid[idx][str(i)] = invalid[idx][str(i)] + ["Error reading SMILES for synthon 2"]
        except Exception as e:
            invalid[idx][str(i)] = invalid[idx][str(i)] + ["Error reading SMILES for synthon 2: "+str(e)]
        if len(invalid[idx].get(str(i),[])) > 0:
            invalid_counts[str(i)] = invalid_counts.get(str(i), 0) + 1

if args.valonly:
    with open(args.results_csv+"_valonly_metrics.json", "w") as f:
        json.dump({
            "invalid": invalid,
            "invalid_counts": invalid_counts,
        }, f)
    exit(0)

maxn = max([len(results) for results in reactions.values()])
sims = {}
diversity = {}
for topn in range(1,11):
    for idx in reactions:
        if len(reactions[idx][:topn]) >= 2:
            simouts_maxavgpair_bin = []
            simouts_maxavgpair_count = []
            simouts_combined_bin = []
            simouts_combined_count = []
            simouts_maxavgpair_bin_correct = []
            simouts_maxavgpair_count_correct = []
            simouts_combined_bin_correct = []
            simouts_combined_count_correct = []
            for idxpair in itertools.combinations(range(len(reactions[idx][:topn])), 2):
                rxnpair = (reactions[idx][idxpair[0]], reactions[idx][idxpair[1]])
                correctpair = (0,0)
                if not args.simonly:
                    correctpair = (rewards[str(idx)][idxpair[0]], rewards[str(idx)][idxpair[1]])
                preds0 = rxnpair[0].split(".")
                preds1 = rxnpair[1].split(".")
                scombob = similarity(rxnpair[0], rxnpair[1], sim_type="binary")
                scomboc = similarity(rxnpair[0], rxnpair[1], sim_type="binary")
                simouts_combined_bin.append(scombob)
                simouts_combined_count.append(scomboc)
                if all(correctpair):
                    simouts_combined_bin_correct.append(scombob)
                    simouts_combined_count_correct.append(scomboc)
                if len(preds0) == 2 and len(preds1) == 2:
                    s00b = similarity(preds0[0], preds1[0], sim_type="binary")
                    s10b = similarity(preds0[1], preds1[0], sim_type="binary")
                    s01b = similarity(preds0[0], preds1[1], sim_type="binary")
                    s11b = similarity(preds0[1], preds1[1], sim_type="binary")
                    s00c = similarity(preds0[0], preds1[0])
                    s10c = similarity(preds0[1], preds1[0])
                    s01c = similarity(preds0[0], preds1[1])
                    s11c = similarity(preds0[1], preds1[1])
                    if not any(map(lambda x: x is None, [s00b, s10b, s01b, s11b, s00c, s10c, s01c, s11c])):
                        simouts_maxavgpair_bin.append(max([(s00b+s11b)/2, (s10b+s01b)/2]))
                        simouts_maxavgpair_count.append(max([(s00c+s11c)/2, (s10c+s01c)/2]))
                        if all(correctpair):
                            simouts_maxavgpair_bin_correct.append(max([(s00b+s11b)/2, (s10b+s01b)/2]))
                            simouts_maxavgpair_count_correct.append(max([(s00c+s11c)/2, (s10c+s01c)/2]))
                elif len(preds0) + len(preds1) != 4:
                    simouts_maxavgpair_bin_correct.append(scombob)
                    simouts_maxavgpair_count_correct.append(scomboc)
            sims[topn] = sims.get(topn, {})
            sims[topn][idx] = {
                "all": {
                    "binary": {
                        "maxavgpair": simouts_maxavgpair_bin,
                        "combined": simouts_combined_bin
                    },
                    "count": {
                        "maxavgpair": simouts_maxavgpair_count,
                        "combined": simouts_combined_count
                    }
                },
                "correct": {
                    "binary": {
                        "maxavgpair": simouts_maxavgpair_bin_correct,
                        "combined": simouts_combined_bin_correct
                    },
                    "count": {
                        "maxavgpair": simouts_maxavgpair_count_correct,
                        "combined": simouts_combined_count_correct
                    }
                }
            }

diversity = {}
for topn in sims:
    diversity[topn] = {
        "all": {
            "binary": {
                "maxavgpair": 1 - np.mean([
                    np.mean([x for x in sims[topn][idx]["all"]["binary"]["maxavgpair"] if x is not None]) for idx in sims[topn]
                    if len(sims[topn][idx]["all"]["binary"]["maxavgpair"]) > 0
                ]),
                "combined": 1 - np.mean([np.mean([x for x in sims[topn][idx]["all"]["binary"]["combined"] if x is not None]) for idx in sims[topn]]),
            },
            "count": {
                "maxavgpair": 1 - np.mean([
                    np.mean([x for x in sims[topn][idx]["all"]["count"]["maxavgpair"] if x is not None]) for idx in sims[topn]
                    if len(sims[topn][idx]["all"]["binary"]["maxavgpair"]) > 0
                ]),
                "combined": 1 - np.mean([np.mean([x for x in sims[topn][idx]["all"]["count"]["combined"] if x is not None]) for idx in sims[topn]]),
            }
        }
    }

if not args.simonly:
    for idx in range(len(fulltest)):
        idx = str(idx)
        if idx not in reactions:
            reactions[idx] = [None]*maxn
            rewards[idx] = []
            exacts[idx] = []
    rewards = {rxn: rewards[rxn] + [0]*(maxn-len(rewards[rxn])) for rxn in rewards}
    exacts = {rxn: exacts[rxn] + [0]*(maxn-len(exacts[rxn])) for rxn in exacts}
    # Diversity[Correct] = 1 - mean(average similarity of correct results)
    # Do not include reactions with one or fewer correct results
    for topn in diversity:
        diversity[topn]["correct"] = {
            "binary": {
                "maxavgpair": 1 - np.mean([
                    np.mean([
                        x for x in sims[topn][idx]["correct"]["binary"]["maxavgpair"] if x is not None
                    ]) for idx in sims[topn] if len(sims[topn][idx]["correct"]["binary"]["maxavgpair"]) > 0
                ]),
                "combined": 1 - np.mean([
                    np.mean([
                        x for x in sims[topn][idx]["correct"]["binary"]["combined"] if x is not None
                    ]) for idx in sims[topn] if len(sims[topn][idx]["correct"]["binary"]["combined"]) > 0
                ]),
            },
            "count": {
                "maxavgpair": 1 - np.mean([
                        np.mean([x for x in sims[topn][idx]["correct"]["count"]["maxavgpair"] if x is not None
                    ]) for idx in sims[topn] if len(sims[topn][idx]["correct"]["count"]["maxavgpair"]) > 0
                ]),
                "combined": 1 - np.mean([
                    np.mean([
                        x for x in sims[topn][idx]["correct"]["count"]["combined"] if x is not None
                    ]) for idx in sims[topn] if len(sims[topn][idx]["correct"]["count"]["combined"]) > 0
                ]),
            }
        }

if not args.simonly:
    with open(args.results_csv+"___full_metrics.json", "w") as f:
        json.dump({
            "rewards": rewards,
            "exacts": exacts,
            "sims": sims,
            "diversity": diversity,
            "invalid": invalid,
            "invalid_counts": invalid_counts,
            "ndcg": np.mean([[np.sum([rewards[str(idx)][k]/np.log2(2+k) for k in range(n+1)]) / np.sum([1/np.log2(2+k) for k in range(n+1)]) for idx in range(len(fulltest))] for n in range(10)], axis=1).tolist(),
            "exact_match_accuracy": np.mean([[int(any(exacts[k][:(j+1)])) for j in range(maxn)] for k in rewards], axis=0).tolist(),
            "anyreward": np.mean([[int(any(rewards[k][:(j+1)])) for j in range(maxn)] for k in rewards], axis=0).tolist(),
            "allreward": np.mean([[int(all(rewards[k][:(j+1)])) for j in range(maxn)] for k in rewards], axis=0).tolist(),
            "avgreward": np.mean([[np.mean(rewards[k][:(j+1)]) for j in range(maxn)] for k in rewards], axis=0).tolist(),
        }, f)
else:
    with open(args.results_csv+"_simonly_metrics.json", "w") as f:
        json.dump({
            "sims": sims,
            "diversity": diversity,
            "exact_match_accuracy": np.mean([[int(any(exacts[k][:(j+1)])) for j in range(maxn)] for k in rewards], axis=0).tolist(),
        }, f)
