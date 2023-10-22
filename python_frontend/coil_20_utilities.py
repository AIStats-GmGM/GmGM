import numpy as np

##############################################
# Utilities for testing video reconstruction #
##############################################

def shuffle_axes(_mat, /, axes: list):
    """
    For every axis, shuffle the matrix on that axis.
    """
    mat = _mat.copy()
    for ax in axes:
        idxs = np.arange(mat.shape[ax])
        np.random.shuffle(idxs)
        slices = [slice(None)] * _mat.ndim
        slices[ax] = idxs
        mat = mat[tuple(slices)]
    return mat

def reconstruct_axes(_mat, /, axes: list, Psis, first_idx=None):
    """
    Uses Psis in a greedy way to work out the best ordering
    of the data.
    """
    orders = []
    for ax in axes:
        Psi = Psis[ax].copy()
        np.fill_diagonal(Psi, 0)
        start = Psi.max(axis=0).argmax()
        if first_idx is None:
            rows = [start]
        else:
            rows = [first_idx]
        for i in range(Psi.shape[0]-1):
            row = rows[-1]

            # Set column to 0 if already used so that it
            # never becomes the strongest connection again
            Psi[:, row] = 0

            # Add strongest connection as next frame
            rows.append(np.argmax(np.abs(Psi[row])))
        orders.append(rows)
    return orders

def get_accuracies(orders):
    accs = []
    names = ('row', 'column', 'frame')
    for name, order in zip(names, orders):
        acc = 0
        for idx, val in enumerate(order):
            if np.abs(order[idx-1] - val) == 1:
                acc += 1
            if np.abs(order[(idx+1)%len(order)] - val) == 1:
                acc += 1
        acc /= 2 * len(order)
        accs.append(acc)
    return accs