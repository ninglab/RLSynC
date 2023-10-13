import torch
import json
from rlsync.agents.rlsync.trainer import Trainer
from rlsync.agents.rlsync.model import DQNModel
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
from argparse import ArgumentParser

ap = ArgumentParser()
ap.add_argument("--name", "-n", type=str, required=True, help="name of training run (will be used for output directory)")
ap.add_argument("--datasets", "-d", nargs="+", required=True, help="list of directories to scan for data, separated by space")
ap.add_argument("--batch-reactions", "-b", type=int, default=10, help="batch size in terms of number of reactions")
ap.add_argument("--gamma", "-g", type=float, default=0.95, help="gamma discount coefficient")
ap.add_argument("--epochs", "-e", type=int, default=25, help="number of epochs to train before stopping")
ap.add_argument("--validation", "-v", type=str, default="data/3val.json", help="validation set to use")
ap.add_argument("--gpu", action="store_true", help="set this flag to use GPU")
args = ap.parse_args()

np.random.seed(1996)
torch.manual_seed(1996)

name = args.name
datadirs = args.datasets
epochs = args.epochs
gamma = args.gamma
device = "cpu"
if args.gpu:
    device = "cuda"

os.makedirs(f"iterations/{name:s}", exist_ok=True)

rpb_files = []
for datadir in datadirs:
    rpb_files += [f"{datadir:s}/{f:s}" for f in os.listdir(datadir)]
model = DQNModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
loss_fn = torch.nn.MSELoss().to(device)

eval_trainer = Trainer({
        "can_change_existing": False,
        "can_multi_bond_ring_atom": False,
        "seed": 1996,
        "step_limit": 3,
        "alternating": False,
        "use_gpu": args.gpu
    }, name=f"iterations/{name:s}", # lr=1e-5,
        epsilon_start=0, epsilon_end=0, device=device,
        training_data_json="data/debug.json", validation_data_json=args.validation)

# # shortcut for evaluations
# full_data = eval_trainer.validation_data
# eval_trainer.validation_data = full_data[:1000]

sw = SummaryWriter(f"iterations/{name:s}")
sw.add_text("data", datadir, 0)
sw.add_text("training_examples", str(len(rpb_files)), 0)

for epoch in range(epochs):
    model.train()
    losses = []
    reaction_idx = 0
    np.random.shuffle(rpb_files)
    while reaction_idx < len(rpb_files):
        batch_inputs = []
        batch_outputs = []
        for reaction in range(args.batch_reactions):
            if reaction_idx >= len(rpb_files):
                break # stop iterations if rpb_files exhausted
            filename = rpb_files[reaction_idx]
            rxn = torch.load(filename)
            for step in rxn:
                for synthon in step[2]:
                    future = step[2][synthon]
                    action_vector = np.concatenate((future["original"], future["target"], future["synthons"], [future["remaining_steps"]]), axis=0)
                    batch_inputs.append(action_vector)
            if "nrpb" in filename:
                batch_outputs += [0]*(3*2)
            else:
                pos_batchout = []
                for i in range(3):
                    pos_batchout += [gamma**i]*2
                batch_outputs += reversed(pos_batchout)
            reaction_idx += 1
        intensor = torch.Tensor(np.array(batch_inputs)).to(device)
        outtensor = torch.Tensor(np.array([batch_outputs]).T).to(device)
        predtensor = model(intensor)
        optimizer.zero_grad()
        loss = loss_fn(predtensor, outtensor)
        loss.backward()
        losses.append(loss.cpu().detach().float())
        optimizer.step()
    epoch_avg_loss = np.mean(losses)
    sw.add_scalar("pretraining/epoch_avg_loss", epoch_avg_loss, epoch)
    
    model.eval()
    eval_trainer.agents["synthon_1"].model = model
    eval_trainer.agents["synthon_2"].model = model
    
    reward_avg, episodes, exact = eval_trainer.evaluate()
    sw.add_scalar("pretraining/validation_reward_mean/1000sample", reward_avg, epoch)
    sw.add_scalar("pretraining/exact_match_rate/1000sample", exact, epoch)

    eval_trainer.summary_writer = None
    eval_trainer._env = None
    torch.save(model, f"iterations/{name:s}/pretrain_epoch_{epoch:05d}.pt")
    torch.save(eval_trainer, f"iterations/{name:s}/pretrain_eval_trainer_{epoch:05d}.pt")