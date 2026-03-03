import torch.nn.functional as F

def sae_loss(x, recon, z, l1_lambda):
    recon_loss = F.mse_loss(recon, x)
    sparsity_loss = l1_lambda * z.abs().mean()
    return recon_loss + sparsity_loss