import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def activation_maximization(sae, backbone, feature_idx, device, steps=200, lr=0.05):

    backbone.eval()
    sae.eval()

    img = torch.randn(1, 3, 224, 224, device=device, requires_grad=True)

    optimizer = torch.optim.Adam([img], lr=lr)

    for _ in range(steps):
        optimizer.zero_grad()

        feats = backbone(img)
        _, z = sae(feats)

        loss = -z[0, feature_idx]
        loss.backward()
        optimizer.step()

        img.data.clamp_(0,1)

    result = img.detach().cpu().squeeze()

    plt.imshow(result.permute(1,2,0))
    plt.title(f"Activation Maximization Feature {feature_idx}")
    plt.axis("off")
    plt.show()