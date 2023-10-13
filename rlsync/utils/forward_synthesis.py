import onmt
from rdkit import Chem
import codecs
from typing import Tuple, Union
import argparse

import torch
import onmt.model_builder
import onmt.translate.beam
from onmt import inputters
import onmt.opts as opts
import onmt.decoders.ensemble
import onmt.model_builder
import os

class ForwardSynthesis(object):
    def __init__(self, model_path, fsverbose = False, **kwargs):
        self.devnull = open(os.devnull, "w")
        self.verbose = fsverbose
        if "n_best" not in kwargs:
            kwargs["n_best"] = 5
        self.n_best = kwargs["n_best"]
        self.translator = self.build_translator(model_path, **kwargs)
    
    def __del__(self):
        self.devnull.close()

    def canonicalize(self, smiles, isomeric=False):
        return ForwardSynthesis.canonicalize(smiles, isomeric=isomeric)
    
    @classmethod
    def canonicalize(cls, smiles, isomeric=False):
        # When canonicalizing a SMILES string, we typically want to
        # run Chem.RemoveHs(mol), but this will try to kekulize the mol
        # which is not required for canonical SMILES.  Instead, we make a
        # copy of the mol retaining only the information we desire (not explicit Hs)
        # Then, we sanitize the mol without kekulization.  copy_atom and copy_edit_mol
        # Are used to create this clean copy of the mol.
        def copy_atom(atom):
            new_atom = Chem.Atom(atom.GetSymbol())
            new_atom.SetFormalCharge(atom.GetFormalCharge())
            return new_atom

        def copy_edit_mol(mol):
            new_mol = Chem.RWMol(Chem.MolFromSmiles(''))
            for atom in mol.GetAtoms():
                new_atom = copy_atom(atom)
                new_mol.AddAtom(new_atom)
            for bond in mol.GetBonds():
                a1 = bond.GetBeginAtom().GetIdx()
                a2 = bond.GetEndAtom().GetIdx()
                bt = bond.GetBondType()
                new_mol.AddBond(a1, a2, bt)
            return new_mol

        smiles = smiles.replace(" ", "")  
        tmp = Chem.MolFromSmiles(smiles, sanitize=False)
        tmp = copy_edit_mol(tmp)
        Chem.SanitizeMol(tmp, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        return Chem.MolToSmiles(tmp, isomericSmiles=isomeric)

    def canonicalize_old(self, smiles, isomeric=True):
        smiles = smiles.replace(" ", "")
        try:        
            tmp = Chem.MolFromSmiles(smiles)
            tmp = Chem.RemoveHs(tmp)
        except:
            return smiles
        if tmp is None:
            return smiles
        [a.ClearProp('molAtomMapNumber') for a in tmp.GetAtoms()]
        return Chem.MolToSmiles(tmp, isomericSmiles=isomeric)

    def smi_tokenizer(self, smi):
        """
        Tokenize a SMILES molecule or reaction
        Taken from MolecularTransformer repository readme
        https://github.com/pschwllr/MolecularTransformer/tree/aeb339daf0a029b391f8307fb3f467f461605dd2
        """
        import re
        pattern = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        regex = re.compile(pattern)
        tokens = [token for token in regex.findall(smi)]
        assert smi == ''.join(tokens)
        return ' '.join(tokens)
    
    def build_translator(self, model_path : str, **kwargs):
        dummy_parser = argparse.ArgumentParser(description='train.py')
        opts.model_opts(dummy_parser)
        dummy_opt = dummy_parser.parse_known_args([])[0]
        opt_parser = argparse.ArgumentParser(description='translate.py')
        onmt.opts.translate_opts(opt_parser)
        opt = opt_parser.parse_known_args(['-model', model_path, '-src', '/dev/null'])[0]
        for k in kwargs:
            opt.__dict__[k] = kwargs[k]
        fields, model, model_opt = onmt.model_builder.load_test_model(opt, dummy_opt.__dict__, model_path=model_path)

        scorer = onmt.translate.GNMTGlobalScorer(opt.alpha,
                                                opt.beta,
                                                opt.coverage_penalty,
                                                opt.length_penalty)

        gpu = -1
        if opt.use_gpu:
            gpu = 1

        translator = onmt.translate.Translator(model, fields, global_scorer=scorer,
                                out_file=self.devnull, report_score=True,
                                copy_attn=model_opt.copy_attn,
                                log_probs_out_file=self.devnull, beam_size=opt.beam_size,
                                replace_unk=True, n_best=opt.n_best,
                                max_length=opt.max_length, gpu=gpu)
        return translator

    def postprocess_smiles(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return Chem.MolToSmiles(mol, isomericSmiles=True)
        else:
            return ''

    def check_in_top_n(self,
        goal : str,
        *reactants
    ) -> Tuple[int, float]:
        reactants = [r for r in reactants if r != ""]
        return self.check_in_top_n_formatted(
            self.smi_tokenizer(self.canonicalize(goal)),
            self.smi_tokenizer(self.canonicalize(".".join(reactants)))
        )

    def check_in_top_n_formatted(self,
        cg : str,
        input_str : str
    ) -> Tuple[int, float]:
        scores, preds = self.translator.translate(
            src_data_iter=[input_str],
            batch_size=1)
        cg = self.canonicalize(cg)
        topn = []
        for p in list(preds[0]):
            try:
                topn.append(self.canonicalize(p.replace(" ","")))
            except Exception as e:
                pass # Only add valid canonical smiles strings from predicitons
        idx = -1
        try:
            idx = topn.index(cg)
        except ValueError as e:
            if self.verbose:
                print("%s not in " % cg, str(topn))
                print("---------------------------")
        score = 0.0
        if idx >=0: 
            score = float(scores[0][idx].detach())
        return (idx+1), score