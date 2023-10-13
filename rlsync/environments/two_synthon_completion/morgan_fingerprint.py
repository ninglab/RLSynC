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

class MorganFingerprintEnv(gymnasium.Env):
    def __init__(self, env_config={}):
        self.fs = ForwardSynthesis("data/mt/MIT_mixed_augm_model_average_20.pt",
            n_best=5,
            beam_size=5,
            use_gpu=env_config.get("use_gpu", False)
        )
        self.action_log = env_config.get("action_log", False)
        self.can_change_existing = env_config.get("can_change_existing", True)
        self.can_add_new_atom = env_config.get("can_add_new_atom", True)
        self.can_multi_bond_ring_atom = env_config.get("can_multi_bond_ring_atom", True)
        self.uuid = uuid4()
        self.bond_types = env_config.get(
            "bond_types",
            [
                None,
                Chem.rdchem.BondType.SINGLE,
                Chem.rdchem.BondType.DOUBLE,
                Chem.rdchem.BondType.TRIPLE,
                Chem.rdchem.BondType.AROMATIC,
            ],
        )
        self.verbose = env_config.get("verbose", False)
        self.step_limit = env_config.get("step_limit", 3)
        self.atom_types =  env_config.get(
            "atom_types", [
                "B",
                "C",
                "N",
                "O",
                "F",
                # "Mg",
                "Si",
                "P",
                "S",
                "Cl",
                # "Cu",
                # "Zn",
                "Se",
                "Br",
                # "Sn",
                "I"
            ]
        )
        self._num_steps = 0
        self._internal_state = {}
        self._setup_spaces()

    def _setup_spaces(self):
        self._agent_ids = set(["synthon_1", "synthon_2"])
        self.observation_space = spaces.Dict({
            "synthon_1": spaces.Dict({
                "synthons": spaces.MultiBinary(4096),
                "remaining_steps": spaces.Discrete(self.step_limit+1)
            }),
            "synthon_2": spaces.Dict({
                "synthons": spaces.MultiBinary(4096),
                "remaining_steps": spaces.Discrete(self.step_limit+1)
            })
        })
        self.action_space = spaces.Dict({
            "synthon_1": Molecule(),
            "synthon_2": Molecule()
        })
        self.action_record = {"smiles/synthon_1": [], "smiles/synthon_2": []}
    
    def step(self, action_dict):
        if self.is_done():
            raise Exception("Cannot step environment that is already done.")
        for synthon in list(self._agent_ids):
            my_action = action_dict[synthon]
            self.action_record[f"smiles/{synthon:s}"].append(
                self.fs.canonicalize(Chem.MolToSmiles(my_action), isomeric=False)
            )
            self._internal_state[synthon] = my_action
        self._num_steps += 1
        obs = self._generate_observation()
        done = self.is_done()
        rew = self._generate_reward()
        info = {agent: {} for agent in list(self._agent_ids)}
        return obs, rew, done, info

    def _generate_observation(self):
        obs = {
            "synthon_1": {
                "synthons": morgan_fp_2_molecules(self._internal_state["synthon_1"], self._internal_state["synthon_2"]),
                "molecule": self._internal_state["synthon_1"],
                "remaining_steps": self.step_limit - self._num_steps
            },
            "synthon_2": {
                "synthons": morgan_fp_2_molecules(self._internal_state["synthon_2"], self._internal_state["synthon_1"]),
                "molecule": self._internal_state["synthon_2"],
                "remaining_steps": self.step_limit - self._num_steps
            }
        }
        return obs
    
    def is_exact_match(self) -> bool:
        reactants = []
        for synthon in self._agent_ids:
            mol = self._internal_state[synthon]
            reactants.append(
                self.fs.canonicalize(
                    Chem.MolToSmiles(mol, isomericSmiles=False),
                    isomeric=False
                )
            )
        if ".".join(sorted(reactants)) == ".".join(sorted(self.known_reactants)):
            return True
        return False

    def calculate_forward_synthesis_reward(self) -> float:
        """Calculates a reward based on the forward synthesis of the current synthons"""
        reactants = []
        for synthon in self._agent_ids:
            mol = self._internal_state[synthon]
            reactants.append(
                self.fs.canonicalize(
                    Chem.MolToSmiles(mol, isomericSmiles=False),
                    isomeric=False
                )
            )
        if ".".join(sorted(reactants)) == ".".join(sorted(self.known_reactants)):
            return 1
        index, score = self.fs.check_in_top_n(self.known_product, *reactants)
        if self.action_log:
            with open(f"{str(self.uuid):s}.reward.json", "w") as jsonf:
                json.dump({"reactants": reactants, "product": self.known_product, "index": index}, jsonf)
        if index > 0:
            return 1
        else:
            return 0

    def is_done(self) -> bool:
        # Placeholder
        if self._num_steps >= self.step_limit:
            return True
        else:
            return False

    def _generate_reward(self):
        """Generates a final reward"""
        if self.is_done():
            fsrew = self.calculate_forward_synthesis_reward()
            self.action_record["reward"] = fsrew
            self.uuid = uuid4()
            if self.action_log:
                with open(f"{self.uuid}.action_log", "w") as jsonf:
                    json.dump(self.action_record, jsonf)
            return {
                agent: fsrew 
                for agent in list(self._agent_ids)
            }
        else:
            return {agent: 0 for agent in list(self._agent_ids)}

    def reset(self, seed=None, options={"reaction": ("C.C", "1|2", "CC", "C.C")}):
        super().reset(seed=seed)
        self._num_steps = 0
        self._invalid_step_count = 0
        synthons, reaction_center_str, product, reactants = options["reaction"]
        r1, r2 = reactants.split(".")
        self.known_reactants = self.fs.canonicalize(r1), self.fs.canonicalize(r2)
        s1, s2 = synthons.split(".")
        del self.action_record
        self.action_record = {
            "smiles/synthon_1": [s1],
            "smiles/synthon_2": [s2],
            "ground_truth/reactants": reactants,
            "ground_truth/product": product
        }
        rcenters = [int(atomnum) for atomnum in reaction_center_str.split("|")]
        pair = (s1, s2)
        self.known_product = self.fs.canonicalize(product, isomeric=False)
        for i in range(len(self._agent_ids)):
            synthon = sorted(list(self._agent_ids))[i]
            self._internal_state[synthon] = Chem.MolFromSmiles(pair[i])
            atomlist = self._internal_state[synthon].GetAtoms()
            for atom in atomlist:
                atom_map_num = atom.GetAtomMapNum()
                if atom_map_num in rcenters:
                    atom.SetProp("center", "1")
                else:
                    atom.SetProp("center", "0")
        return self._generate_observation()

    def get_possible_actions(self):
        possible_dict = {}
        for synthon in self._agent_ids:
            possible = [Chem.rdchem.Mol(self._internal_state[synthon])] # start with no-op option
            for n1idx in range(possible[0].GetNumAtoms()):
                if int(possible[0].GetAtomWithIdx(n1idx).GetProp("center")) == 1:
                    # First, verify this is an action center atom
                    # Allow adding or removing bond with existing atoms,
                    # Also allows removal of atoms
                    if self.can_change_existing:
                        possible += self._get_actions_change_bond(possible[0], n1idx)
                    # Allows atom to be added
                    if self.can_add_new_atom:
                        possible += self._get_actions_add_atom(possible[0], n1idx)
            possible_dict[synthon] = possible
        return possible_dict

    def _get_with_added_atom(self, molecule, n1idx, atom_type, bond_type):
        mol = Chem.rdchem.RWMol(molecule)
        a1 = mol.GetAtomWithIdx(n1idx)
        totalHs = a1.GetTotalNumHs()
        a1.SetNoImplicit(False)
        a1.SetNumExplicitHs(0)
        a1.GetImplicitValence()
        mol.UpdatePropertyCache()
        atom = Chem.Atom(atom_type)
        atom.SetProp("center", "1")
        mol.AddAtom(atom)
        n2idx = mol.GetNumAtoms()-1
        mol.AddBond(n1idx, n2idx, bond_type)
        try:
            Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
            if not self.can_multi_bond_ring_atom:
                self._sanitize_multi_bond_ring_atom(mol)
            mol.UpdatePropertyCache()
            Chem.rdmolops.FastFindRings(mol)
            return mol
        except ValueError:
            return None
    
    def _sanitize_multi_bond_ring_atom(self, mol):
        for atom in mol.GetAromaticAtoms():
            buddy = []
            for bond in atom.GetBonds():
                if not bond.GetIsAromatic():
                    if atom.GetIdx() == bond.GetEndAtom().GetIdx():
                        buddy.append(bond.GetBeginAtom().GetIdx())
                    else:
                        buddy.append(bond.GetEndAtom().GetIdx())
            if len(buddy) > 1:
                raise ValueError("Cannot bond multiple non-aromatic atoms to a ring atom.")
                return False
        return True

    def _get_with_changed_bond(self, molecule, exist_bond, new_bond, n1idx, n2idx):
        mol = Chem.rdchem.RWMol(molecule)
        if exist_bond is not None:
            mol.RemoveBond(n1idx, n2idx)
        if new_bond is not None:
            mol.AddBond(n1idx, n2idx, new_bond)
        # Iterate atoms in reverse index order (large idx to small idx) and remove orphans
        atom_idxs = sorted([a.GetIdx() for a in mol.GetAtoms()], reverse=True)
        for idx in atom_idxs:
            if len(mol.GetAtoms()[idx].GetBonds()) == 0:
                mol.RemoveAtom(idx)
        try:
            Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
            if Chem.MolFromSmiles(Chem.MolToSmiles(mol)) is None:
                raise ValueError("Cannot kekulize")
            mol = Chem.RemoveHs(mol)
            mol.UpdatePropertyCache()
            Chem.rdmolops.FastFindRings(mol)
            return mol
        except ValueError:
            return None
        
    def _get_actions_change_bond(self, molecule, n1idx):
        """
        Generate list of actions on node 1 (n1idx) that change a bond
        """
        possible = []
        for n2idx in range(molecule.GetNumAtoms()):
            if n2idx != n1idx:
                # skip self
                for bond_type in self.bond_types:
                    if bond_type is Chem.rdchem.BondType.AROMATIC:
                        continue # skip aromatic
                    ebond = molecule.GetBondBetweenAtoms(n1idx, n2idx)
                    if ebond != bond_type:
                        mol = self._get_with_changed_bond(molecule, ebond, bond_type, n1idx, n2idx)
                        if mol is not None:
                            possible.append(mol)
        return possible
    
    def _get_actions_add_atom(self, molecule, n1idx):
        possible = []
        for atom_type in self.atom_types:
            if atom_type is None:
                continue # skip "None" atom type
            # skip self
            for bond_type in self.bond_types: #
                if bond_type is None or bond_type is Chem.rdchem.BondType.AROMATIC:
                    continue # skip "None" bonds and aromatic bonds
                mol = self._get_with_added_atom(molecule, n1idx, atom_type, bond_type)
                if mol is not None:
                    possible.append(mol)
        return possible