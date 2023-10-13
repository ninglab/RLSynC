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
from .morgan_fingerprint import MorganFingerprintEnv as MFPBase

class MorganFingerprintEnv(MFPBase):
    def __init__(self, env_config={}):
        super().__init__(env_config=env_config)
        self.alternating = env_config.get("alternating", False)

    def _setup_spaces(self):
        super()._setup_spaces()
        self.observation_space = spaces.Dict({
            "synthon_1": spaces.Dict({
                "original": spaces.MultiBinary(4096),
                "actions_taken": spaces.Discrete(self.step_limit + 1),
                "noops_taken": spaces.Discrete(self.step_limit + 1),
                "synthons": spaces.MultiBinary(4096),
                "molecule": Molecule(),
                "remaining_steps": spaces.Discrete(self.step_limit+1)
            }),
            "synthon_2": spaces.Dict({
                "original": spaces.MultiBinary(4096),
                "actions_taken": spaces.Discrete(self.step_limit + 1),
                "noops_taken": spaces.Discrete(self.step_limit + 1),
                "synthons": spaces.MultiBinary(4096),
                "molecule": Molecule(),
                "remaining_steps": spaces.Discrete(self.step_limit+1)
            })
        })

    def _generate_observation(self):
        if self._num_steps > 0:
            self._noops = {s: self.count_noops(s) for s in self._noops}
        obs = {
            "synthon_1": {
                "original": morgan_fp_2_molecules(self._originals["synthon_1"], self._originals["synthon_2"]),
                "target": morgan_fp_1_molecule(Chem.MolFromSmiles(self.known_product, sanitize=False)),
                "actions_taken": self._num_steps - self._noops["synthon_1"],
                "noops_taken": self._noops["synthon_1"],
                "synthons": morgan_fp_2_molecules(self._internal_state["synthon_1"], self._internal_state["synthon_2"]),
                "molecule": self.action_record["smiles/synthon_1"][-1],
                "remaining_steps": self.step_limit - self._num_steps
            },
            "synthon_2": {
                "original": morgan_fp_2_molecules(self._originals["synthon_2"], self._originals["synthon_1"]),
                "target": morgan_fp_1_molecule(Chem.MolFromSmiles(self.known_product, sanitize=False)),
                "actions_taken": self._num_steps - self._noops["synthon_2"],
                "noops_taken": self._noops["synthon_2"],
                "synthons": morgan_fp_2_molecules(self._internal_state["synthon_2"], self._internal_state["synthon_1"]),
                "molecule": self.action_record["smiles/synthon_2"][-1],
                "remaining_steps": self.step_limit - self._num_steps
            }
        }
        return obs

    def get_possible_actions(self):
        poss = super().get_possible_actions()
        if self.alternating:
            if self._num_steps % 2 == 0:
                poss["synthon_2"] = [poss["synthon_2"][0]]
            else:
                poss["synthon_1"] = [poss["synthon_1"][0]]
        return poss

    def count_noops(self, synthon):
        count = 0
        for i in range(len(self.action_record[f"smiles/{synthon:s}"])-1):
            if self.action_record[f"smiles/{synthon:s}"][i] == self.action_record[f"smiles/{synthon:s}"][i+1]:
                count += 1
        return count
    
    def reset(self, seed=None, options={"reaction": ("C.C", "1|2", "CC", "C.C")}):
        # Use dummy _originals values so that super() can run:
        self._originals = {s: Chem.MolFromSmiles("C") for s in self._agent_ids}
        self._noops = {"synthon_1": 0, "synthon_2": 0}
        # Don't keep obs from super, regenerate at end using self._generate_observations
        super().reset(seed=seed, options=options)
        self._originals = {s: Chem.Mol(self._internal_state[s]) for s in self._internal_state}
        return self._generate_observation()