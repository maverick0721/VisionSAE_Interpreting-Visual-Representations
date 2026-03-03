import torch
from alignment.similarity import cosine_similarity_matrix
from alignment.hungarian import hungarian_matching

def cross_width_alignment(W_small, W_large):

    sim = cosine_similarity_matrix(W_small, W_large)

    row_ind, col_ind = hungarian_matching(sim)

    aligned_sim = sim[row_ind, col_ind]

    return aligned_sim.mean().item()