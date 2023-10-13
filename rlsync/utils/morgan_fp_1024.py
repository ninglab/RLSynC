from rdkit.Chem import AllChem
from rdkit import Chem

import numpy as np

__all__ = ["morgan_fp_1_molecule", "morgan_fp_2_molecules"]

def morgan_fp_2_molecules(molecule_1 : Chem.rdchem.Mol, molecule_2 : Chem.rdchem.Mol) -> np.ndarray:
    molecule_1.UpdatePropertyCache()
    Chem.rdmolops.FastFindRings(molecule_1)
    molecule_2.UpdatePropertyCache()
    Chem.rdmolops.FastFindRings(molecule_2)
    s1input = AllChem.GetMorganFingerprintAsBitVect(
        molecule_1,
        2,
        nBits=1024,
        useChirality=False
    ).ToBitString()
    s2input = AllChem.GetMorganFingerprintAsBitVect(
        molecule_2,
        2,
        nBits=1024,
        useChirality=False
    ).ToBitString()
    return np.array([float(x) for x in s1input + s2input], dtype=np.float32)

def morgan_fp_1_molecule(molecule_1 : Chem.rdchem.Mol) -> np.ndarray:
    molecule_1.UpdatePropertyCache()
    Chem.rdmolops.FastFindRings(molecule_1)
    s1input = AllChem.GetMorganFingerprintAsBitVect(
        molecule_1,
        2,
        nBits=1024,
        useChirality=False
    ).ToBitString()
    return np.array([float(x) for x in s1input], dtype=np.float32)
