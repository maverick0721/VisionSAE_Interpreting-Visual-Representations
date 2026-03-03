import torch
import torch.nn.functional as F

def mutual_coherence(W):
    W = F.normalize(W, dim=0)
    sim = torch.matmul(W.T, W)
    off_diag = sim - torch.eye(sim.size(0), device=sim.device)
    return off_diag.abs().max().item()

def effective_rank(W):
    U, S, V = torch.svd(W)
    p = S / S.sum()
    entropy = -(p * torch.log(p + 1e-8)).sum()
    return torch.exp(entropy).item()

def condition_number(W):
    S = torch.linalg.svdvals(W)
    return (S.max() / S.min()).item()