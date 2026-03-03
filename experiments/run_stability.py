import argparse
import torch
import json
import os
from models.sae import SparseAutoencoder
from alignment.stability import compute_stability
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
parser.add_argument("--layer", type=int, required=True)
parser.add_argument("--width", type=int, required=True)
parser.add_argument("--seed1", type=int, required=True)
parser.add_argument("--seed2", type=int, required=True)
args = parser.parse_args()

cfg = yaml.safe_load(open(args.config))

device = cfg["training"]["device"]

input_dim = 768  # ViT base output dim

def load_model(seed):
    sae = SparseAutoencoder(
        input_dim,
        args.width,
        cfg["sae"]["sparsity"],
        cfg["sae"]["topk"]
    ).to(device)

    path = (
        f"checkpoints/layer_{args.layer}"
        f"_width_{args.width}"
        f"_seed_{seed}.pt"
    )

    sae.load_state_dict(torch.load(path, weights_only=True))
    sae.eval()
    return sae.decoder.weight.detach()

W1 = load_model(args.seed1)
W2 = load_model(args.seed2)

score = compute_stability(W1, W2)

os.makedirs("results/stability", exist_ok=True)

output = {
    "layer": args.layer,
    "width": args.width,
    "seed1": args.seed1,
    "seed2": args.seed2,
    "stability_score": score
}

with open(
    f"results/stability/layer_{args.layer}"
    f"_width_{args.width}"
    f"_seed{args.seed1}_{args.seed2}.json",
    "w"
) as f:
    json.dump(output, f, indent=2)

print(output)