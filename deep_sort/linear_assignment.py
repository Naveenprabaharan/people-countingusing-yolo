# FILE: deep_sort/linear_assignment.py

import numpy as np
from scipy.optimize import linear_sum_assignment


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), np.arange(cost_matrix.shape[0]), np.arange(cost_matrix.shape[1])

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches = []
    unmatched_rows = []
    unmatched_cols = []

    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] > thresh:
            unmatched_rows.append(r)
            unmatched_cols.append(c)
        else:
            matches.append([r, c])

    unmatched_rows += [r for r in range(cost_matrix.shape[0]) if r not in row_ind]
    unmatched_cols += [c for c in range(cost_matrix.shape[1]) if c not in col_ind]

    return np.array(matches, dtype=int), np.array(unmatched_rows, dtype=int), np.array(unmatched_cols, dtype=int)
