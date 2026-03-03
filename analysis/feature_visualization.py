import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

def top_activating_images(sae, backbone, dataloader, feature_idx, device, top_k=9):

    sae.eval()
    backbone.eval()

    scores = []
    images_list = []

    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            feats = backbone(images)
            _, z = sae(feats)

            activation = z[:, feature_idx]

            scores.append(activation.cpu())
            images_list.append(images.cpu())

    scores = torch.cat(scores)
    images_all = torch.cat(images_list)

    top_indices = torch.topk(scores, top_k).indices

    top_imgs = images_all[top_indices]

    grid = torchvision.utils.make_grid(top_imgs, nrow=3)
    npimg = grid.numpy()

    plt.figure(figsize=(6,6))
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.axis("off")
    plt.title(f"Feature {feature_idx} Top Activations")
    plt.show()