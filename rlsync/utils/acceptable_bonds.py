import json
from typing import List
from rdkit import Chem

class AcceptableBonds(object):
    def __init__(self, data="data/training_bonds.json"):
        if not hasattr(AcceptableBonds, 'acceptable'):
            with open(data, "r") as f:
                AcceptableBonds.acceptable = json.load(f)
    
    def __contains__(self, item):
        a1, bond, a2 = item
        bondstr = {
            Chem.rdchem.BondType.SINGLE: "single",
            Chem.rdchem.BondType.DOUBLE: "double",
            Chem.rdchem.BondType.TRIPLE: "triple"
        }[bond]
        atoms = sorted([a1, a2])
        query = "%s %s %s" % (atoms[0], bondstr, atoms[1])
        return query in AcceptableBonds.acceptable