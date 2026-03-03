import torch
from scipy.optimize import linear_sum_assignment

def hungarian_matching(sim_matrix):
    cost_matrix = -sim_matrix.cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return row_ind, col_ind