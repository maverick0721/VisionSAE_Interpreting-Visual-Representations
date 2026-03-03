import torch
import torch.nn.functional as F

def cosine_similarity_matrix(W1, W2):
    W1 = F.normalize(W1, dim=0)
    W2 = F.normalize(W2, dim=0)
    return torch.matmul(W1.T, W2)