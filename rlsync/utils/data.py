import csv
from typing import Iterator, Tuple
from .forward_synthesis import ForwardSynthesis
import re
from rdkit import Chem
import json

def load_reactants(filename : str, model_path : str) -> Iterator[Tuple[str, str]]:
    fs = ForwardSynthesis(
        model_path,
        n_best=5,
        fsverbose=True,
        use_gpu=False,
        gpu=-1,
        beam_size=5)
    with open(filename, "r") as synthf:
        reader = csv.DictReader(synthf)
        counter = 0
        for row in reader:
            counter += 1
            goal = fs.canonicalize(row["product"], isomeric=False)
            goal = "".join(goal)
            s1, s2 = row["synthons"].split(".")
            r1, r2 = row["reactants"].split(".")
            s1 = fs.canonicalize(s1, isomeric=False)
            s2 = fs.canonicalize(s2, isomeric=False)
            r1 = fs.canonicalize(r1, isomeric=False)
            r2 = fs.canonicalize(r2, isomeric=False)
            yield (r1, r2, goal)

def load_synthons_reaction_center(filename : str, model_path : str, subset="add_one_to_both", includeIdx=False) -> Iterator[Tuple[str, str, str]]:
    fs = ForwardSynthesis(
        model_path,
        n_best=5,
        fsverbose=True,
        use_gpu=False,
        gpu=-1,
        beam_size=5)
    with open(filename, "r") as synthf:
        reader = csv.DictReader(synthf)
        counter = 0
        for row in reader:
            counter += 1
            center = row["center_atoms"]
            #goal = fs.canonicalize(row["product"], isomeric=False)
            #goal = "".join(goal)
            goal = row["product"]
            s1, s2 = row["synthons"].split(".")
            r1, r2 = row["reactants"].split(".")
            cs1 = fs.canonicalize(s1, isomeric=False)
            cs2 = fs.canonicalize(s2, isomeric=False)
            cr1 = fs.canonicalize(r1, isomeric=False)
            cr2 = fs.canonicalize(r2, isomeric=False)
            no_synthon_charge = (re.search(r'\[.*[-+]\]', cs1+"."+cs2) is None)
            no_reactant_charge = (re.search(r'\[.*[-+]\]', cr1+"."+cr2) is None)
            # if no_synthon_charge and no_reactant_charge:
            #     if re.search(r'\[.*[-+]\]', goal) is None:
            s1len, s2len = sorted([Chem.MolFromSmiles(s1, sanitize=False).GetNumAtoms(), Chem.MolFromSmiles(s2, sanitize=False).GetNumAtoms()])
            r1len, r2len = sorted([Chem.MolFromSmiles(r1, sanitize=False).GetNumAtoms(), Chem.MolFromSmiles(r2, sanitize=False).GetNumAtoms()])
            result = (f"{s1:s}.{s2:s}", center, goal, f"{r1:s}.{r2:s}")
            if includeIdx:
                result = (counter-1, f"{s1:s}.{s2:s}", center, goal, f"{r1:s}.{r2:s}")
            if subset == "add_one_to_both":
                if abs(s1len - r1len) == 1 and abs(s2len - r2len) == 1:
                    # {+1 0: 19324, 0 0: 876, +1 +1: 324}
                    yield result
            elif subset == "add_one_to_either_but_not_both":
                if (abs(s1len - r1len) + abs(s2len - r2len)) == 1:
                    # {+1 0: 19324, 0 0: 876, +1 +1: 324}
                    yield result
            elif subset == "add_one_to_either_or_both":
                if abs(s1len - r1len) == 1 or abs(s2len - r2len) == 1:
                    # {+1 0: 19324, 0 0: 876, +1 +1: 324}
                    yield result
            elif subset == "add_at_most_one_to_either_or_both":
                if abs(s1len - r1len) <= 1 and abs(s2len - r2len) <= 1:
                    # {+1 0: 19324, 0 0: 876, +1 +1: 324}
                    yield result
            elif subset == "threestep":
                if abs(s1len - r1len) <= 3 and abs(s2len - r2len) <= 3:
                    # {+1 0: 19324, 0 0: 876, +1 +1: 324}
                    yield result
            elif subset == "noop":
                if abs(s1len - r1len) == 0 and abs(s2len - r2len) == 0:
                    # 876
                    yield result
            elif subset == "1noop":
                if abs(s1len - r1len) == 0 and abs(s2len - r2len) == 0:
                    # 1
                    yield result
                    break
            elif subset == "reward_noop":
                if fs.check_in_top_n(goal, fs.canonicalize(s1, isomeric=False), fs.canonicalize(s2, isomeric=False))[0] > 0:
                    yield result
            elif subset == "all":
                yield result
            else:
                raise ValueError("Invalid subset %s." % subset)

def load_synthons_reaction_center_json(json_filename : str) -> Iterator[Tuple[str, str, str, str]]:
    items = []
    with open(json_filename, "r") as f:
        items = json.load(f)
    for i in items:
        yield i
    
def load_synthons(filename : str, model_path : str, all_synthons:bool=False) -> Iterator[Tuple[str, str]]:
    fs = ForwardSynthesis(
        model_path,
        n_best=5,
        fsverbose=True,
        use_gpu=False,
        gpu=-1,
        beam_size=5)
    with open(filename, "r") as synthf:
        reader = csv.DictReader(synthf)
        counter = 0
        for row in reader:
            counter += 1
            goal = fs.canonicalize(row["product"], isomeric=False)
            goal = "".join(goal)
            s1, s2 = row["synthons"].split(".")
            r1, r2 = row["reactants"].split(".")
            s1 = fs.canonicalize(s1, isomeric=False)
            s2 = fs.canonicalize(s2, isomeric=False)
            r1 = fs.canonicalize(r1, isomeric=False)
            r2 = fs.canonicalize(r2, isomeric=False)
            if re.search(r'\[.*[-+]\]', s1+"."+s2) is None:
                if re.search(r'\[.*[-+]\]', goal) is None:
                    s1len, s2len = sorted([Chem.MolFromSmiles(s1).GetNumAtoms(), Chem.MolFromSmiles(s2).GetNumAtoms()])
                    r1len, r2len = sorted([Chem.MolFromSmiles(r1).GetNumAtoms(), Chem.MolFromSmiles(r2).GetNumAtoms()])
                    result = (s1, s2, goal)
                    if abs(s1len - r1len) <= 1 and abs(s2len - r2len) <= 1:
                        # {+1 0: 19324, 0 0: 783, +1 +1: 324}
                        yield result
                    elif all_synthons:
                        yield result