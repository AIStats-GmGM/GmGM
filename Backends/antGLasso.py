import numpy as np
from Backends.utilities import kronecker_factor_diag
from Backends.utilities import nmode_gram, shrink_sparsities


def antGLasso(
    Ys: "(n, d_1, ..., d_K) input tensor",
    sparsities: ("List of numbers of edges to keep for Psis", "Hyperparameter") = None
): 
    ds = Ys.shape[1:]
    Ss = [nmode_gram(Ys, 0), nmode_gram(Ys, 1)]

    es, Vs = zip(*[np.linalg.eigh(S) for S in Ss])
    
    Xs = rescaleYs(Ys, Vs)
    unconstrained = 1 / calculateSigmas(Xs)
    
    es_out = [
        kronecker_factor_diag(
            unconstrained,
            ds,
            ell
        )
        for ell in range(len(ds))
    ]
    
    Psis = [V @ np.diag(e) @ V.T for V, e in zip(Vs, es_out)]
    
    # Regularize
    Psis = shrink_sparsities(Psis, sparsities)
    
    return Psis



def rescaleYs(
    Ys: "(n, d_1, ..., d_K) tensor",
    Vs: "List of (d_ell, d_ell) eigenvectors of Psi_ell",
) -> "(n, d_1, ..., d_K) tensor":
    """
    Rescales our input data to be drawn from a kronecker sum
    distribution with parameters being the eigenvalues
    
    An implementation of Lemma 1
    """
    n, *d = Ys.shape
    K = len(d)
    Xs = Ys
    for k in range(0, K):
        # Shuffle important axis to end, multiply, then move axis back
        Xs = np.moveaxis(
            np.moveaxis(
                Xs,
                k+1,
                -1
            ) @ Vs[k],
            -1,
            k+1
        )
    return Xs

def calculateSigmas(
    Xs: "(n, d_1, ..., d_k) tensor"
) -> "(n, d_1, ..., d_k) tensor OF variances":
    """
    Gets an MLE for variances of our rescaled Ys
    
    An implementation of Lemma 2
    """
    
    (n, *d) = Xs.shape
    return ((Xs**2).sum(axis=0) / n)