import torch
from tqdm import tqdm
from training.loss import sae_loss

def train_sae(sae, features, cfg):

    device = cfg["training"]["device"]
    epochs = cfg["training"]["epochs"]

    optimizer = torch.optim.AdamW(
        sae.parameters(),
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"]["weight_decay"])
    )

    scaler = torch.cuda.amp.GradScaler()

    features = features.to(device)

    for epoch in range(epochs):
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            recon, z = sae(features)
            loss = sae_loss(
                features, recon, z,
                float(cfg["training"]["l1_lambda"])
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        sae.normalize_decoder()

        print(f"Epoch {epoch} | Loss {loss.item():.6f}")

    return sae