import argparse
import torch
import yaml
import json
import os
from models.sae import SparseAutoencoder
from alignment.cross_width import cross_width_alignment

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
parser.add_argument("--layer", type=int, required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--width_small", type=int, required=True)
parser.add_argument("--width_large", type=int, required=True)
args = parser.parse_args()

cfg = yaml.safe_load(open(args.config))
device = cfg["training"]["device"]

input_dim = 768

def load_model(width):
    sae = SparseAutoencoder(
        input_dim,
        width,
        cfg["sae"]["sparsity"],
        cfg["sae"]["topk"]
    ).to(device)

    path = (
        f"checkpoints/layer_{args.layer}"
        f"_width_{width}"
        f"_seed_{args.seed}.pt"
    )

    sae.load_state_dict(torch.load(path, weights_only=True))
    sae.eval()
    return sae.decoder.weight.detach()

W_small = load_model(args.width_small)
W_large = load_model(args.width_large)

score = cross_width_alignment(W_small, W_large)

os.makedirs("results/cross_width", exist_ok=True)

output = {
    "layer": args.layer,
    "seed": args.seed,
    "width_small": args.width_small,
    "width_large": args.width_large,
    "alignment_score": score
}

with open(
    f"results/cross_width/layer_{args.layer}"
    f"_seed_{args.seed}"
    f"_{args.width_small}_{args.width_large}.json",
    "w"
) as f:
    json.dump(output, f, indent=2)

print(output)