import json
import random
from uuid import uuid4
import gymnasium
import numpy as np
import torch
from gymnasium import spaces
from rdkit import Chem
from rdkit.Chem import AllChem
from ...utils import ForwardSynthesis
from ...utils.morgan_fp import *
from ...utils.spaces import Molecule
from .mfp_with_original import MorganFingerprintEnv as MFPBase
from ...utils.acceptable_bonds import AcceptableBonds

class MorganFingerprintEnv(MFPBase):
    def _get_actions_add_atom(self, molecule, n1idx):
        possible = []
        existing_atom_type = molecule.GetAtomWithIdx(n1idx).GetSymbol()
        for atom_type in self.atom_types:
            if atom_type is None:
                continue # skip "None" atom type
            for bond_type in self.bond_types: #
                if bond_type is None or bond_type is Chem.rdchem.BondType.AROMATIC:
                    continue # skip "None" bonds and aromatic bonds
                if (atom_type, bond_type, existing_atom_type) not in AcceptableBonds():
                    continue # skip bonds not found in training ground truth
                mol = self._get_with_added_atom(molecule, n1idx, atom_type, bond_type)
                if mol is not None:
                    possible.append(mol)
        return possible