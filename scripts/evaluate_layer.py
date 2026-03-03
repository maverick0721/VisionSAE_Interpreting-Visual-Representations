from utils.env import setup_env
setup_env()

import argparse
import yaml
import torch
import json
import os
from models.sae import SparseAutoencoder
from analysis.metrics import sparsity
from analysis.geometry import mutual_coherence, effective_rank, condition_number

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
parser.add_argument("--layer", type=int, required=True)
args = parser.parse_args()

cfg = yaml.safe_load(open(args.config))
device = cfg["training"]["device"]

features = torch.load(f"features/layer_{args.layer}.pt").to(device)

input_dim = features.shape[-1]

sae = SparseAutoencoder(
    input_dim,
    cfg["sae"]["hidden_dim"],
    cfg["sae"]["sparsity"],
    cfg["sae"]["topk"]
).to(device)

sae.load_state_dict(torch.load(f"checkpoints/layer_{args.layer}.pt"))
sae.eval()

with torch.no_grad():
    recon, z = sae(features)
    mse = torch.nn.functional.mse_loss(recon, features).item()
    sp = sparsity(z)
    coh = mutual_coherence(sae.decoder.weight)
    er = effective_rank(sae.decoder.weight)
    cond = condition_number(sae.decoder.weight)

os.makedirs("results", exist_ok=True)

result = {
    "layer": args.layer,
    "mse": mse,
    "sparsity": sp,
    "coherence": coh,
    "effective_rank": er,
    "condition_number": cond
}

with open(f"results/layer_{args.layer}.json", "w") as f:
    json.dump(result, f, indent=2)

print(result)