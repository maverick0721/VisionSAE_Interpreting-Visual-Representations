import torch
from alignment.similarity import cosine_similarity_matrix
from alignment.hungarian import hungarian_matching

def compute_stability(W1, W2):

    sim = cosine_similarity_matrix(W1, W2)

    row_ind, col_ind = hungarian_matching(sim)

    aligned_sim = sim[row_ind, col_ind]

    stability_score = aligned_sim.mean().item()

    return stability_score