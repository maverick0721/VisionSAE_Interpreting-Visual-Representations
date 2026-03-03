import torch
import matplotlib.pyplot as plt

def visualize_patch_activation(
    sae,
    backbone,
    image,
    feature_idx,
    layer_idx,
    device
):
    sae.eval()
    backbone.eval()

    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        tokens = backbone(
            image,
            layer_idx=layer_idx,
            return_tokens=True
        )  # [1, 196, 768]

        B, N, D = tokens.shape

        tokens = tokens.view(-1, D)  # [196, 768]

        z = sae.encoder(tokens)      # [196, hidden_dim]

        if sae.sparsity == "topk":
            from models.topk import top_k_activation
            z = top_k_activation(z, sae.topk)

        feature_activation = z[:, feature_idx]  # [196]

        heatmap = feature_activation.view(14, 14).cpu()

    plt.figure(figsize=(6,6))
    plt.imshow(heatmap, cmap="hot")
    plt.colorbar()
    plt.title(f"Feature {feature_idx} Patch Activation")
    plt.axis("off")
    plt.show()