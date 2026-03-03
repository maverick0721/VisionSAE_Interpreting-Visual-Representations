import torch
import torch.nn as nn
import torch.nn.functional as F
from .topk import top_k_activation

class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, sparsity="topk", topk=50):
        super().__init__()

        self.encoder = nn.Linear(input_dim, hidden_dim, bias=False)
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=False)

        self.sparsity = sparsity
        self.topk = topk

        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.xavier_uniform_(self.decoder.weight)

    def forward(self, x):
        z = self.encoder(x)

        if self.sparsity == "topk":
            z = top_k_activation(z, self.topk)
        else:
            z = F.relu(z)

        recon = self.decoder(z)
        return recon, z

    def normalize_decoder(self):
        with torch.no_grad():
            self.decoder.weight.data = F.normalize(
                self.decoder.weight.data,
                dim=0
            )