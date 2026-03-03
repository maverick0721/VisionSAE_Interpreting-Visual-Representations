import torch

def participation_ratio(W):
    U, S, V = torch.svd(W)
    numerator = (S.sum())**2
    denominator = (S**2).sum()
    return (numerator / denominator).item()