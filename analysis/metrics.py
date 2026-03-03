def sparsity(z):
    return (z == 0).float().mean().item()