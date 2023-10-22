import numpy as np

def generate_confusion_matrices(
    pred: "Square matrix",
    truth: "Square matrix",
    eps: "Tolerance" = 0,
    mode: "See docstring" = '<Tolerance'
) -> "(2, 2) confusion matrix [[TP, FP], [FN, TN]]":
    """
    `mode`:
        '<Tolerance': x->1 if |x| < eps, else x->0
        'Nonzero': Same as '<Tolerance'
        'Negative': x->1 if x < 0 else x -> 0
        'Mixed': truth is '<Tolerance', pred is 'Negative'
    """
    mode_pred = mode if mode != 'Mixed' else 'Negative'
    mode_truth = mode if mode != 'Mixed' else 'Nonzero'
    pred = binarize_matrix(pred, eps=eps, mode=mode_pred)
    truth = binarize_matrix(truth, eps=0, mode=mode_truth)
    
    # Identity matrices to remove diagonals
    In = np.eye(pred.shape[0])
    
    np.fill_diagonal(pred, 1)
    np.fill_diagonal(truth, 1)
    
    TP: "True positives"
    TP = (pred * truth - In).sum()
    
    FP: "False positives"
    FP = (pred * (1 - truth)).sum()
    
    TN: "True negatives"
    TN = ((1 - pred) * (1 - truth)).sum()
    
    FN: "False negatives"
    FN = ((1 - pred) *  truth).sum()
    
    return np.array([
        [TP, FP],
        [FN, TN]
    ])

def precision(
    cm: "[[TP, FP], [FN, TN]]"
):
    denom = (cm[0, 0] + cm[0, 1])
    if denom == 0:
        return 1
    return cm[0, 0] / denom

def recall(
    cm: "[[TP, FP], [FN, TN]]"
):
    denom = (cm[0, 0] + cm[1, 0])
    if denom == 0:
        return 1
    return cm[0, 0] / denom

def binarize_matrix(
    M: "Input matrix of any dimensions",
    eps: "Tolerance" = 0,
    mode: "Negative | <Tolerance" = '<Tolerance'
):
    """
    Returns M but with only ones and zeros
    """
    out = np.empty(M.shape)
    if mode == '<Tolerance' or mode == 'Nonzero':
        out[np.abs(M) <= eps] = 0
        out[np.abs(M) > eps] = 1
    elif mode == 'Negative':
        out[M < 0] = 1
        out[M >= 0] = 0
        
        # In negative mode we need to add diagonals back
        out += np.eye(out.shape[0])
    else:
        raise Exception(f'Invalid mode {mode}')
    return out

def binarize_matrices(
    Ms: "Dict of matrices to binarize",
    eps: "Tolerance" = 0,
    mode: "Negative | <Tolerance" = '<Tolerance'
):
    return {
        axis: binarize_matrix(M, eps=eps, mode=mode)
        for axis, M in Ms.items()
    }


##################
# Regularization #
##################

def shrink_sparsities(
    Psis: "List or dict of matrices to shrink",
    sparsities: "List or dict assumed sparsities",
    safe: "If false, edit in-place, else make copy" = True
) -> "List or dict of sparsity-shrunk Psis":
    if isinstance(Psis, dict):
        return {
            axis: shrink_axis(
                Psi,
                sparsities[axis],
                safe
            )
            for axis, Psi in Psis.items()
        }
    else:
        return [
            shrink_axis(Psi, s, safe)
            for Psi, s in zip(Psis, sparsities)
        ]

def shrink_per_row(
    Psis: "List or dict of matrices to shrink",
    ns: "Number of elements per row to keep",
    safe: "If false, edit in-place, else make copy" = True
) -> "List or dict of shrunk Psis":
    if isinstance(Psis, dict):
        return {
            axis: shrink_axis_per_row(
                Psi,
                ns[axis],
                safe
            )
            for axis, Psi in Psis.items()
        }
    else: 
        return [
            shrink_axis_per_row(Psi, n, safe)
            for Psi, n in zip(Psis, ns)
        ]
    
def shrink_per_col(
    Psis: "List or dict of matrices to shrink",
    ns: "Number of elements per row to keep",
    safe: "If false, edit in-place, else make copy" = True
) -> "List or dict of shrunk Psis":
    if isinstance(Psis, dict):
        return {
            axis: shrink_axis_per_col(
                Psi,
                ns[axis],
                safe
            )
            for axis, Psi in Psis.items()
        }
    else: 
        return [
            shrink_axis_per_col(Psi, n, safe)
            for Psi, n in zip(Psis, ns)
        ]

def shrink_axis(
    Psi: "Matrix to shrink",
    sparsity: float,
    safe: "If false, edit in-place, else make copy" = True
) -> "Sparsity-shrunk Psi":
    if safe:
        Psi = Psi.copy()
    Psabs = np.abs(Psi)
    np.fill_diagonal(Psabs, 0)
    quant = np.quantile(Psabs, 1-sparsity)
    Psi[Psabs < quant] = 0
    return Psi

def shrink_axis_per_row(
    Psi: "Matrix to shrink",
    n: "Number of elements per row to keep",
    safe: "If false, edit in-place, else make copy" = True
) -> "Sparsity-shrunk Psi":
    if safe:
        Psi = Psi.copy()
    Psabs = np.abs(Psi)
    np.fill_diagonal(Psabs, 0)
    for i in range(Psabs.shape[0]):
        Psi[i, np.argsort(Psabs[i, :])[:-n]] = 0
    Psi = Psi + Psi.T
    return Psi

def shrink_axis_per_col(
    Psi: "Matrix to shrink",
    n: "Number of elements per row to keep",
    safe: "If false, edit in-place, else make copy" = True
) -> "Sparsity-shrunk Psi":
    if safe:
        Psi = Psi.copy()
    Psabs = np.abs(Psi)
    np.fill_diagonal(Psabs, 0)
    Psabs /= Psabs.sum(axis=0, keepdims=True)
    for i in range(Psabs.shape[0]):
        Psi[i, np.argsort(Psabs[i, :])[:-n]] = 0
    Psi = Psi + Psi.T
    return Psi
