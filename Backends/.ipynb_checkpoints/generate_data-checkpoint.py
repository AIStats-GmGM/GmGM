import numpy as np
from scipy.stats import invwishart, bernoulli
from scipy.stats import multivariate_normal
from Backends.utilities import kron_sum, kron_sum_diag_fast, kron_prod

def fast_ks_normal(
    Psis: "List of (d_i, d_i) precision matrices, of length K >= 2",
    size: "Number of samples",
    fail_if_not_posdef: bool = False
) -> "Kronecker sum distributed tensor":
    K = len(Psis)
    ds = [Psi.shape[0] for Psi in Psis]
    vs, Vs = zip(*[np.linalg.eigh(Psi) for Psi in Psis])
    diag_precisions = kron_sum_diag_fast(*vs)
    
    min_diag = diag_precisions.min()
    if min_diag < 0:
        if fail_if_not_posdef:
            raise Exception("KS of Psis not Positive Definite")
        diag_precisions -= (min_diag-1)
    
    z = multivariate_normal(cov=1).rvs(
        size=size*np.prod(ds)
    ).reshape(size, np.prod(ds)) / np.sqrt(diag_precisions)
    
    Xs: "Sample of diagonalized distribution" = z.reshape(size, *ds)
    
    for k in range(K):
        Xs = np.moveaxis(
            np.moveaxis(Xs, k+1, -1) @ Vs[k].T,
            -1,
            k+1
        )
    return Xs

def generate_Psis(
    ds: "List of shapes of precision matrices",
    *,
    sparsities: "List of percent of nonzero edges in ground truth",
    gen_type: "bernoulli or invwishart" = "bernoulli"
) -> "List of precision matrices, (m, *ds) sample tensor":
    
    Psis = []
    
    # Input checking
    if len(sparsities) != len(ds):
        raise Exception("Your number of regularizers need to match dimensions")
        
    for s, d in zip(sparsities, ds):
        Psi = generate_Psi(d, s, gen_type=gen_type)
        Psis.append(Psi)
        
    return Psis

def generate_Psi(
    d: int,
    s: float,
    *,
    gen_type: "bernoulli or invwishart" = "bernoulli"
) -> "List of precision matrices, (m, *ds) sample tensor":
    
    if gen_type == "bernoulli":
        Psi: "(d, d)" = np.triu(bernoulli.rvs(p=s, size=(d, d)))
        Psi = Psi.T + Psi
        np.fill_diagonal(Psi, 1)
    elif gen_type == "invwishart":
        Psi = generate_sparse_invwishart_matrix(
            d,
            s*d**2 / 2,
            off_diagonal_scale=0.9,
            size=1,
            df_scale=1
        ).squeeze()
    else:
        raise Exception(f"Invalid input '{gen_type}'!")
    return Psi
    

def generate_Ys(
    m: "Number of Samples",
    ds: "List of shapes of precision matrices",
    *,
    sparsities: "List of percent of nonzero edges in ground truth",
    gen_type: "bernoulli or invwishart" = "bernoulli"
) -> "List of precision matrices, (m, *ds) sample tensor":
    
    Psis = generate_Psis(ds, sparsities=sparsities, gen_type=gen_type)
        
    Ys = fast_ks_normal(Psis, m)
    if (m > 1):
        Ys -= Ys.mean(axis=0)
        
    return Psis, Ys

def generate_multi_Ys(
    m: list["Number of Samples"],
    structure: list[tuple["Axis Name"]],
    ds: dict["Axis Name", "Axis sizes"],
    *,
    sparsities: dict["Axis Name", "Axis Sparsity"],
    gen_type: "bernoulli or invwishart" = "bernoulli"
) -> tuple[
    dict["Axis Name", "Precision Matrix"],
    list["(m[idx], *ds[idx]) sample tensors"]
]:
    """
    Suppose we have a multiomics dataset of:
    300 people x 50 gut microbes x 12 timestamps
    300 people x 200 metabolites
    
    Then structure would be:
    [
        ("people", "microbes", "time"),
        ("people", "metabolites")
    ]
    
    and ds would be:
    {
        "people": 300,
        "gut microbes": 50,
        "time": 12,
        "metabolites": 200
    }
    """
    
    Psis: dict["Axis Name", "Precision Matrix"] = {}
    
    for name in ds.keys():
        Psis[name] = generate_Psi(
            ds[name],
            sparsities[name],
            gen_type=gen_type
        )
        
    Ys: list = [None] * len(structure)
    for idx, struct in enumerate(structure):
        Ys[idx] = fast_ks_normal(
            [Psis[name] for name in struct],
            m
        )
        
    return Psis, Ys
    
    
    
    

def generate_sparse_invwishart_matrix(
    n: "Number of rows/columns of output",
    expected_nonzero: "Number of nondiagonal nonzero entries expected",
    *,
    off_diagonal_scale: "Value strictly between 0 and 1 to guarantee posdefness" = 0.9,
    size: "Number of samples to return" = 1,
    df_scale: "How much to multiply the df parameter of invwishart, must be >= 1" = 1
) -> "(`size`, n, n) batch of sparse positive definite matrices":
    """
    Generates two sparse positive definite matrices.
    Relies on Schur Product Theorem; we create a positive definite mask matrix and
    then hadamard it with our precision matrices
    """
    
    Psi: "Sparse posdef matrix - the output"
    
    p: "Bernoulli probability to achieve desired expected value of psi nonzeros"
    p = np.sqrt(expected_nonzero / (n**2 - n))
    
    # Note that in the calculation of D, we make use of Numpy broadcasting.
    # It's the same as hadamard-producting with np.eye(n) tiled `size` times
    # in the size dimension and 1-b*b `n` times in the -1 dimension,
    # which is equivalent to making a batch of diagonal matrices from
    # entries of b.
    Mask: "Mask to zero out elements while preserving pos. definiteness"
    b = bernoulli(p=p).rvs(size=(size, n, 1)) * np.sqrt(off_diagonal_scale)
    D = (1-b*b)*np.eye(n)
    Mask = D + b @ b.transpose([0, 2, 1])
    Psi = invwishart.rvs(df_scale * n, np.eye(n), size=size) / (df_scale * n) * Mask
    #Psi = wishart.rvs(df_scale * n, np.eye(n), size=size) / (df_scale * n) * Mask
    
    # This just affects how much normalization is needed
    Psi /= np.trace(Psi, axis1=1, axis2=2).reshape(size, 1, 1) / n
    
    return Psi