import torch
import torch.nn.functional as F

def collapse_score(W, threshold=0.9):

    W = F.normalize(W, dim=0)
    sim = torch.matmul(W.T, W)

    upper = sim.triu(diagonal=1)

    collapsed = (upper > threshold).float().mean().item()

    return collapsed