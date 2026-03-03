from utils.env import setup_env
setup_env()

import argparse
import yaml
import torch
import os
from models.sae import SparseAutoencoder
from models.utils import set_seed
from training.trainer import train_sae

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
parser.add_argument("--layer", type=int, required=True)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--width", type=int, default=None)
args = parser.parse_args()

cfg = yaml.safe_load(open(args.config))

if args.seed is not None:
    cfg["seed"] = args.seed

if args.width is not None:
    cfg["sae"]["hidden_dim"] = args.width

set_seed(cfg["seed"])

device = cfg["training"]["device"]

features = torch.load(f"features/layer_{args.layer}.pt")

input_dim = features.shape[-1]

sae = SparseAutoencoder(
    input_dim,
    cfg["sae"]["hidden_dim"],
    cfg["sae"]["sparsity"],
    cfg["sae"]["topk"]
).to(device)

sae = train_sae(sae, features, cfg)

os.makedirs("checkpoints", exist_ok=True)
torch.save(
    sae.state_dict(),
    f"checkpoints/layer_{args.layer}_width_{cfg['sae']['hidden_dim']}_seed_{cfg['seed']}.pt"
)