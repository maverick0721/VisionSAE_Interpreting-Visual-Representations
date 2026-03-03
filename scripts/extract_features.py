import argparse
import yaml
import torch
import os
from tqdm import tqdm
from data.datamodule import get_dataloader
from models.backbone import VisionBackbone

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
parser.add_argument("--layer", type=int, required=True)
args = parser.parse_args()

cfg = yaml.safe_load(open(args.config))
device = cfg["training"]["device"]

loader = get_dataloader(cfg)
backbone = VisionBackbone(
    cfg["model"]["backbone"],
    cfg["model"]["pretrained"]
).to(device)

backbone.eval()
os.makedirs("features", exist_ok=True)

features = []

with torch.no_grad():
    for images, _ in tqdm(loader):
        images = images.to(device)
        feats = backbone(images, layer_idx=args.layer)
        features.append(feats.cpu())

features = torch.cat(features)
torch.save(features, f"features/layer_{args.layer}.pt")