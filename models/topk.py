import torch

def top_k_activation(x, k):
    values, indices = torch.topk(x, k, dim=1)
    mask = torch.zeros_like(x)
    mask.scatter_(1, indices, values)
    return mask