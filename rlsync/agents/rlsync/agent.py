import numpy as np
import torch
from ...utils.morgan_fp import morgan_fp_1_molecule

class DQNAgent(object):
    def __init__(self, model, device="cpu", optimizer=None):
        self.model = model
        self.device = device
    
    def select_action(self, obs, possible_actions, epsilon=0):
        if len(possible_actions) == 1:
            return possible_actions[0]
        elif np.random.rand() >= epsilon:
            inputs = torch.tensor(np.stack([self._predict_future_value(obs, pa) for pa in possible_actions])).to(self.device)
            values = self.model(inputs).detach().cpu().numpy()
            return possible_actions[np.argmax(values)]
        return np.random.choice(possible_actions)
    
    def select_action_beam(self, obs, possible_actions, beam_sz=3):
        if len(possible_actions) < beam_sz:
            return possible_actions
        inputs = torch.tensor(np.stack([self._predict_future_value(obs, pa) for pa in possible_actions])).to(self.device)
        values = self.model(inputs).detach().cpu().numpy()
        return possible_actions[np.argsort(-values)]
    
    def select_action_beam(self, obs, possible_actions, beam_size=3):
        if len(possible_actions) < beam_size:
            return possible_actions
        inputs = torch.tensor(np.stack([self._predict_future_value(obs, pa) for pa in possible_actions])).to(self.device)
        values = self.model(inputs).detach().cpu().numpy()
        # reverse sort to get largest values
        topn = np.argsort(values.flatten())[::-1][:beam_size]
        return [possible_actions[int(k)] for k in topn]

    def _predict_future_value(self, obs, action):
        future = {
            "original": np.copy(obs["original"]),
            "target": np.copy(obs["target"]),
            "synthons": np.copy(obs["synthons"]),
            "remaining_steps": np.array([obs["remaining_steps"] - 1 ], dtype=np.float32)
        }
        future["synthons"][0:2048] = morgan_fp_1_molecule(action)
        return np.concatenate((future["original"], future["target"], future["synthons"], future["remaining_steps"]), axis=0)
