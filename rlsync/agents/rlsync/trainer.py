from ...environments.two_synthon_completion.mfp_limited import MorganFingerprintEnv
from .agent import DQNAgent
from .model import DQNModel
from ...utils.buffer import Buffer
from ...utils.data import load_synthons_reaction_center_json
import random
import numpy as np
import itertools
import torch
import os
import json
import uuid
from torch.utils.tensorboard import SummaryWriter
import tracemalloc
import sys
from rdkit import Chem

_DEFAULTS = {
    "lr": 0.0001,
    "epsilon_start": 1.0,
    "epsilon_end": 0.01,
    "epsilon_decay_rate": 0.999,
    "share_params": True,
    "update_interval": 100,
    "replay_buffer_size": 15000,
    "batch_size": 32,
    "gamma": 0.95,
    "device": "cpu",
    "eval_with_epsilon": False
}

class Trainer(object):
    def __init__(self, env_config={}, model_options={}, **kwargs):
        self.name = kwargs.get("name", "untitled")
        self.summary_writer = SummaryWriter(f"runs.scratch/{self.name:s}")
        tracemalloc.start(100)
        self.trace_preinit = tracemalloc.take_snapshot()
        self._init(env_config=env_config, model_options=model_options, **kwargs)
        self.hparams = {k: kwargs.get(k, _DEFAULTS[k]) for k in _DEFAULTS}
        self.hparams["name"] = self.name
        self.summary_writer.add_text("hparams", json.dumps(self.hparams))

    def _init(self, training_data_json=None, validation_data_json=None, checkpoint_prefix=None,
                 env_config={}, model_options={}, lr=0.0001, epsilon_start=1.0, epsilon_end=0.01,
                 epsilon_decay_rate=0.999, epsilon_decay_mode="exponential", share_params=True, eval_with_epsilon=False,
                 update_interval=100, replay_buffer_size=15000, loss_class=torch.nn.HuberLoss,
                 batch_size=32, gamma=0.95, device="cpu", name=None):
        self.epochs = 0
        self.epsilon_decay_mode = epsilon_decay_mode
        self._env = None
        self.checkpoint_prefix = checkpoint_prefix
        self.eval_with_epsilon = eval_with_epsilon
        if self.checkpoint_prefix is None:
            self.checkpoint_prefix = str(uuid.uuid4())
        self.training_data = list(load_synthons_reaction_center_json(training_data_json))
        self.validation_data = list(load_synthons_reaction_center_json(validation_data_json))
        self.env_config = env_config
        self.loss_class = loss_class
        self.loss_fn = loss_class()
        self.device=device
        self.replay_buffer = Buffer(replay_buffer_size)
        self.gamma=gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay_rate
        self.batch_sz = batch_size
        self.update_interval = update_interval
        self.summary_writer.add_text("env_config/training", json.dumps(self.env_config))
        self.loss_avg_denom = 0
        self.loss_avg = {}
        self.num_updates = 0
        self.num_episodes = 0
        self.num_steps = 0
        self._model = DQNModel(options=model_options).to(self.device)
        self.models = {
            "synthon_1": self._model,
            "synthon_2": self._model
        }
        self._target = DQNModel(options=model_options).to(self.device)
        self.targets = {
            "synthon_1": self._target,
            "synthon_2": self._target
        }
        self.optimizer = torch.optim.Adam(self._model.parameters(), lr=lr)
        self.agents = {
            "synthon_1": DQNAgent(self.models["synthon_1"], device=self.device),
            "synthon_2": DQNAgent(self.models["synthon_2"], device=self.device)
        }

    def do_update(self, episode):
        for synthon in self.models:
            # Update each target function with state dict from corresponding DQN.
            self.targets[synthon].load_state_dict(self.models[synthon].state_dict())
        self.log(episode)
        self.checkpoint()
        self.num_updates += 1
        self.cumulative_reward = {}
        for synthon in self.agents:
            self.loss_avg[synthon] = 0.0
            self.cumulative_reward[synthon] = 0.0
        self.loss_avg_denom = 0

    def get_env(self):
        if self._env is None:
            self._env = MorganFingerprintEnv(self.env_config)
        return self._env
    
    def grad_norm(self, m):
        return np.sqrt(sum([p.grad.norm().item() ** 2 for p in m.parameters() if p.grad is not None])) 

    def train(self, seed=None):
        self.epochs += 1
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        env = self.get_env()
        data = np.array(self.training_data)
        np.random.shuffle(data)
        for model in self.models:
            self.models[model].train()
        for episode in range(len(self.training_data)):
            if episode % self.update_interval == 0:
                self.do_update(episode)
            if episode % 10 == 9:
                torch.save({
                    "action_record": env.action_record,
                    "known_product": env.known_product,
                    "internal_state": env._internal_state,
                    "rewards": env._generate_reward(),
                    "done": env.is_done(),
                    "num_steps": env._num_steps,
                    "": ""
                }, f"runs.scratch/{self.name:s}/debug_env.pt")
            obs = env.reset(options={"reaction": data[episode]})
            self.num_episodes += 1
            done = False
            while not done:
                possible = env.get_possible_actions()
                action = {
                    s: self.agents[s].select_action(obs[s], possible[s], self.epsilon) for s in self.agents
                }
                new_obs, rew, new_done, _info = env.step(action)
                self.num_steps += 1
                done = new_done
                for agent in self.agents:
                    self.cumulative_reward[agent] += rew[agent]
                self.replay_buffer.append((obs, rew, new_obs, done))
                self.epsilon_update()
                if len(self.replay_buffer._list) > self.batch_sz:
                    batch = self.replay_buffer.sample(self.batch_sz)
                    for agent in self.agents:
                        q_t = torch.zeros(self.batch_sz, 1, requires_grad=False)
                        q_tp1 = torch.zeros(self.batch_sz, 1, requires_grad=False)
                        for i in range(len(batch)):
                            s, r, a, d = batch[i]
                            state = torch.concat((torch.tensor(s[agent]["original"]), torch.tensor(s[agent]["target"]), torch.tensor(s[agent]["synthons"]), torch.tensor(np.array([s[agent]["remaining_steps"]])))).to(self.device)
                            new_action = torch.concat((torch.tensor(a[agent]["original"]), torch.tensor(a[agent]["target"]), torch.tensor(a[agent]["synthons"]), torch.tensor(np.array([a[agent]["remaining_steps"]])))).to(self.device)
                            q_t[i] = self.models[agent](state)
                            if not d:
                                q_tp1[i] = self.gamma * self.targets[agent](new_action).detach()
                            q_tp1[i] += r[agent]
                        q_t.to(self.device)
                        q_tp1.to(self.device)
                        loss = self.loss_fn(q_t, q_tp1).to(self.device)
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                        self.summary_writer.add_scalar(f"training/grad_norm/{agent:s}", self.grad_norm(self.models[agent]))
                        self.summary_writer.add_scalar(f"training/batch_loss/{agent:s}", loss.detach().float(), self.num_episodes)
                        agent_idx = 0
                        if agent == "synthon_2":
                            agent_idx = 1
                        self.summary_writer.add_scalar(f"training/batch_loss/all", loss.detach().float(), self.num_episodes*2 + agent_idx)
                        self.loss_avg[agent] += loss.detach().float()
                    self.loss_avg_denom += 1
                obs = new_obs
        self.do_update(len(self.training_data))
        os.rename(f"runs.scratch/{self.name:s}/latest.pt", f"runs.scratch/{self.name:s}/epoch_{self.epochs:03d}.pt")

    def checkpoint(self):
        os.makedirs("runs.scratch/"+self.name, exist_ok=True)
        summary = self.summary_writer
        self.summary_writer = None
        env = self._env
        self._env = None
        if os.path.exists(f"runs.scratch/{self.name:s}/latest.pt"):
            os.rename(f"runs.scratch/{self.name:s}/latest.pt", f"runs.scratch/{self.name:s}/previous.pt")
        torch.save(self, f"runs.scratch/{self.name:s}/latest.pt")
        self.summary_writer = summary
        self._env = env

    def log(self, episode):
        if self.num_updates > 0 and episode > 0:
            raw_denom = self.update_interval
            if episode % self.update_interval != 0:
                raw_denom = episode % self.update_interval
            for synthon in self.agents:
                self.summary_writer.add_scalar(f"training/episode_cumulative_reward/{synthon:s}", self.cumulative_reward[synthon], self.num_episodes)
                self.summary_writer.add_scalar(f"training/episode_reward_mean/{synthon:s}", self.cumulative_reward[synthon] / raw_denom, self.num_episodes)
                tlavg = 0.0
                if self.loss_avg_denom > 0:
                    tlavg = self.loss_avg[synthon]*1.0 / self.loss_avg_denom
                self.summary_writer.add_scalar(f"training/loss_avg/{synthon:s}", tlavg, self.num_episodes)
            self.summary_writer.add_scalar(f"training/loss_avg/if_shared", (self.loss_avg["synthon_1"] + self.loss_avg["synthon_2"]) / max(2.0*self.loss_avg_denom, 1), self.num_episodes)
            self.summary_writer.add_scalar(f"batches_since_last_update/per_synthon", self.loss_avg_denom, self.num_episodes)
            self.summary_writer.add_scalar(f"batches_since_last_update/total_combined", self.loss_avg_denom*2, self.num_episodes)
        self.summary_writer.add_scalar("training/target_network_updates", self.num_updates, self.num_episodes)
        self.summary_writer.add_scalar("training/training_steps", self.num_steps, self.num_episodes)
        self.summary_writer.add_scalar("epsilon", self.epsilon, self.num_episodes)
        self.summary_writer.add_scalar("replay_buffer_filled", len(self.replay_buffer._list), self.num_episodes)
        current, peak = tracemalloc.get_traced_memory()
        self.summary_writer.add_scalar("memory/current", current, self.num_episodes)
        self.summary_writer.add_scalar("memory/peak", peak, self.num_episodes)
        self.summary_writer.flush()

    def epsilon_update(self):
        if self.epsilon_decay_mode == "exponential":
            self.epsilon *= self.epsilon_decay
        elif self.epsilon_decay_mode == "linear":
            self.epsilon -= self.epsilon_decay
        self.epsilon = max(self.epsilon, self.epsilon_end)

    def evaluate(self):
        for model in self.models:
            self.models[model].eval()
        episodes = []
        reward = 0.0
        exact = 0.0
        env = self.get_env()
        val_len = len(self.validation_data)
        for ep in range(val_len):
            obs = env.reset(options={"reaction": self.validation_data[ep]})
            record = self._eval_one_episode(env, obs)
            episodes.append(record)
            reward += record["reward"]
            exact += int(env.is_exact_match())
        return reward / val_len, episodes, exact / val_len

    def validate(self):
        reward_mean, episodes, exact = self.evaluate()
        if self.summary_writer is not None:
            self.summary_writer.add_scalar("validation/episode_reward_mean", reward_mean, self.num_episodes)
            self.summary_writer.add_scalar("validation/exact_match_accuracy", exact, self.num_episodes)
        return {"reward_mean_val": reward_mean, "exact_match_accuracy": exact}, episodes

    def record_hparams(self, result):
        self.summary_writer.add_hparams(self.hparams, result)

    def _eval_one_episode(self, env, obs):
        eps = 0
        if self.eval_with_epsilon:
            eps = self.epsilon
        with torch.no_grad():
            done = False
            while not done:
                possible = env.get_possible_actions()
                action = {
                    s: self.agents[s].select_action(obs[s], possible[s], eps) for s in self.agents
                }
                obs, rew, new_done, _info = env.step(action)
                done = new_done
            return env.action_record
        
    def format_current_value(self, obs):
        current = {
            "original": np.copy(obs["original"]),
            "target": np.copy(obs["target"]),
            "synthons": np.copy(obs["synthons"]),
            "remaining_steps": np.array([obs["remaining_steps"]], dtype=np.float32)
        }
        return np.concatenate((current["original"], current["target"], current["synthons"], current["remaining_steps"]), axis=0)

    def perform_multi_agent_q_beam_search(self, envopts, existing_steps=[], beam_size=3, both_q=False, calculate_reward=True):
        env = self.get_env()
        if not calculate_reward:
            env.calculate_forward_synthesis_reward = lambda *args, **kwargs: 0
        obs = env.reset(options=envopts)
        done = False
        with torch.no_grad():
            rpb = []
            for jointaction in existing_steps:
                new_obs, rew, done, _info = env.step(jointaction)
                rpb.append([obs, rew, new_obs, done])
                obs = new_obs
            if done and both_q:
                q1 = self.agents["synthon_1"].model(
                    torch.Tensor(self.format_current_value(obs["synthon_1"])).to(self.device)
                ).detach().float()
                q2 = self.agents["synthon_2"].model(
                    torch.Tensor(self.format_current_value(obs["synthon_2"])).to(self.device)
                ).detach().float()
                return (
                    rew,
                    existing_steps,
                    envopts["reaction"],
                    Chem.MolToSmiles(env._internal_state["synthon_1"], isomericSmiles=False) + "." + 
                        Chem.MolToSmiles(env._internal_state["synthon_2"], isomericSmiles=False),
                    rpb,
                    (q1+q2)
                )
            elif done:
                # Get Q value and final state
                q = self.agents["synthon_1"].model(
                    torch.Tensor(self.format_current_value(obs["synthon_1"])).to(self.device)
                ).detach().float()
                return (
                    rew,
                    existing_steps,
                    envopts["reaction"],
                    Chem.MolToSmiles(env._internal_state["synthon_1"], isomericSmiles=False) + "." + 
                        Chem.MolToSmiles(env._internal_state["synthon_2"], isomericSmiles=False),
                    rpb,
                    q
                )
            possible = env.get_possible_actions()
            actionsets = {
                s: self.agents[s].select_action_beam(obs[s], possible[s], beam_size=beam_size) for s in self.agents
            }
            jointactions = itertools.product(actionsets["synthon_1"], actionsets["synthon_2"])
            results = []
            for jact in jointactions:
                jointact = {"synthon_1": jact[0], "synthon_2": jact[1]}
                result = self.perform_multi_agent_q_beam_search(envopts, existing_steps=[*existing_steps, jointact], beam_size=beam_size)
                if isinstance(result, tuple):
                    results.append(result)
                else:
                    results += result
            return results