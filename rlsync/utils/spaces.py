from gymnasium import spaces
import random
import numpy as np
import string
import rdkit
import rdkit.Chem

# Reference: https://github.com/openai/gym/issues/1945
class String(spaces.Space):
    def __init__(
                self,
                length=None,
                min_length=1,
                max_length=180,
                default=None
            ):
        self.length = length
        self.default = default
        self.min_length = min_length
        self.max_length = max_length
        self._shape = (1,)
        self.dtype = str
        self.letters = string.ascii_letters + " [:].,!-0123456789()=#@!$%^&*/|_,.;}{~"

    def sample(self):
        if self.default is not None:
            return self.default
        return "C"*self.min_length

    def __eq__(self, other):
        if isinstance(other, String):
            if self.min_length == other.min_length and self.max_length == other.max_length:
                return True
        return False

    def contains(self, x):
        is_a_string = isinstance(x, str)
        correct_length = self.min_length <= len(x) <= self.max_length
        correct_letters = all([l in self.letters for l in x])
        return is_a_string and correct_length and correct_letters

class Molecule(spaces.Space):
    def __init__(self):
        self._shape = (1,)

    def sample(self):
        return rdkit.Chem.MolFromSmiles("C")

    def __eq__(self, other):
        if isinstance(other, Molecule):
            return True
        else:
            return False

    def contains(self, x):
        is_a_mol = isinstance(x, rdkit.Chem.rdchem.Mol)
        return is_a_mol
