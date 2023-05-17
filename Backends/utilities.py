"""
Collected helper functions
"""

import numpy as np

################################
# Utilities for linear algebra #
################################

def tr_p(A: "Matrix", p: "Contraction size"):
    """
    Calculates the blockwise trace, i.e. break the matrix
    up into p by p blocks and form the matrix of traces of
    those blocks
    """
    (r, c) = A.shape
    assert r % p == 0 and c % p == 0, \
        f"Dimensions mismatch: {r, c} not contractible by {p}"
    out = np.empty((r // p, c // p))
    for x in range(r // p):
        for y in range(c // p):
            out[x, y] = np.trace(A[x*p:(x+1)*p, y*p:(y+1)*p])
    return out

def kron_sum(A, B):
    """
    Computes the kronecker sum of two square input matrices
    
    Note: `scipy.sparse.kronsum` is a thing that would
    be useful - but it seems that `scipy.sparse` is not
    yet a mature library to use.
    """
    a, _ = A.shape
    b, _ = B.shape
    return np.kron(A, np.eye(b)) + np.kron(np.eye(a), B)

def kron_prod(A, B):
    """
    Computes the kronecker product.  There is a built in
    np.kron, but it's slow.  Can use a broadcasting trick to
    speed things up.
    
    Trick from greggo's answer in:
    https://stackoverflow.com/questions/7193870/speeding-up-numpy-kronecker-products
    """
    a1, a2 = A.shape
    b1, b2 = B.shape
    return (
        A[:, np.newaxis, :, np.newaxis]
        * B[np.newaxis, :, np.newaxis, :]
    ).reshape((a1*b1, a2*b2))

def nmode_gram(A, n):
    """
    Calculates the mode-n gram matrix of A,
    assuming the first dimension is a batch dimension
    (So n=0 refers to the second index of A,
    n=1 refers to the third, ets)
    """
    An = np.reshape(
        np.moveaxis(A, n+1, 1),
        (A.shape[0], A.shape[n+1], -1), # The -1 infers the value for (d_{\n})
        order='F' # Do math vectorization order rather than numpy vectorization order
    )
    return (An @ An.transpose([0, 2, 1])).mean(axis=0) / An.shape[-1]

def kron_sum_diag_fast(
    *lams: "1D diagonals of matrices to be kronsummed"
) -> "Diagonal of Kronecker sum of lams":
    # Setup
    ds = [len(lam) for lam in lams]
    d_lefts = np.cumprod([1] + ds[:-1]).astype(int)
    d_rights = np.cumprod([1] + ds[::-1])[-2::-1].astype(int)
    total = d_rights[0] * ds[0]
    out = np.zeros(total)
    
    
    for ell, lam in enumerate(lams):
        add_like_kron_sum(
            out,
            lam,
            ell,
            ds[ell],
            d_lefts[ell],
            d_rights[ell]
        )
        
    return out

def kron_sum_diag_faster(
    init: "Matrix to overwrite with output",
    *lams: "1D diagonals of matrices to be kronsummed"
) -> "Diagonal of Kronecker sum of lams":
    """
    This is ever-so-slightly faster than
    `kron_sum_diag_fast`.
    
    The difference is probably not too important
    unless you really want to squeeze every last
    ounce of efficiency out of your code.
    """
    # Setup
    ds = [len(lam) for lam in lams]
    d_lefts = np.cumprod([1] + ds[:-1]).astype(int)
    d_rights = np.cumprod([1] + ds[::-1])[-2::-1].astype(int)
    total = d_rights[0] * ds[0]
    init[:] = 0
    
    
    for ell, lam in enumerate(lams):
        add_like_kron_sum(
            init,
            lam,
            ell,
            ds[ell],
            d_lefts[ell],
            d_rights[ell]
        )
        
    return init

def add_like_kron_sum(
    cur_kron_sum: "Kronsummed matrix",
    to_add: "What to add to matrix",
    ell: "Dimension to add along",
    d, d_left, d_right
) -> None:
    """
    !!!!Modifies cur_kron_sum in place!!!!
    
    Let X[+]Y be the Kronecker sum
    of diagonal matrices.
    Sometimes we want to find X[+](Y+Z)
    for diagonal Z
    
    This is a way to update our pre-computed
    X[+]Y to incorporate the additive Z.
    """
    # We're gonna be really naughty here and use stride_tricks
    # This is going to reshape our vector in a way so that the elements
    # we want to affect are batched by the first two dimensions
    sz = to_add.strides[0]
    toset = np.lib.stride_tricks.as_strided(
        cur_kron_sum,
        shape=(
            d_left, # The skips
            d_right, # The blocks
            d # What we want
        ),
        strides=(
            sz * d * d_right,
            sz * 1,
            sz * d_right,
        )
    )
    toset += to_add
    
def fast_sum_log(
    G: "Vector",
    simplify_size: int = 20
):
    """
    A quicker way to calculate np.sum(np.log(G)),
    equivalently np.log(np.prod(G))
    
    We do this by chunking it into batches of size
    `simplify_size`, computing np.log(np.prod(batch))
    and then summing those together
    [So basically np.sum(np.log(np.prod(G)))]
    
    This is good because `np.log` is quite slow,
    so we don't want to perform it on every element of G.
    
    The least amount of logs would be to apply it only once,
    in np.log(np.prod(G)), but G can be quite large and
    thus we can under/overflow quite easily.
    
    Changing `simplify_size` allows us a trade-off between
    speed (larger `simplify_size`) and stability
    (smaller `simplify_size`)
    """
    
    est_size: int = G.shape[0] // simplify_size
    sum_log = np.sum(
        np.log(
            G[:simplify_size * est_size].reshape(
                simplify_size,
                est_size
            ).prod(axis=0)
        )
    )
    
    # Add the stuff that didn't divide evenly
    sum_log += np.log(G[simplify_size * est_size:].prod())
    
    return sum_log

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

################################################
# Kronecker-Sum Projections and Decompositions #
################################################

def factorwise_average_diag(
    vec: "Diagonal of mat to be factored",
    ds: "Dimensions of factor matrices",
    k: "Dimension"
) -> "Diagonal of resulting matrix":
    """
    Helper function for the KS-Projection
    of a diagonal matrix.
    
    Surprisingly, the calculation of
    d, d_left, and d_right take up a non-trivial
    part of the runtime of this function, so
    consider using
    `factorwise_average_diag_supplied_d`
    if speed is an issue.
    """
    d = ds[k]
    d_left = np.prod(ds[:k]).astype(int)
    d_right = np.prod(ds[k+1:]).astype(int)
    
    rs = vec.strides[0]
    
    out = np.lib.stride_tricks.as_strided(
        vec,
        shape=(
            d_left,
            d_right,
            d,
        ),
        strides=(
            rs * d_right * d,
            rs * 1,
            rs * d_right,
        )
    )

    return out.mean(axis=(0, 1))

def factorwise_average_diag_supplied_d(
    vec: "Diagonal of mat to be factored",
    d: int,
    d_left: int,
    d_right: int
) -> "Diagonal of resulting matrix":
    """
    Helper function for the KS-Projection
    of a diagonal matrix
    """
    
    rs = vec.strides[0]
    
    out =  np.lib.stride_tricks.as_strided(
        vec,
        shape=(
            d_left,
            d_right,
            d,
        ),
        strides=(
            rs * d_right * d,
            rs * 1,
            rs * d_right,
        )
    )

    return out.sum(axis=(0, 1)) / (d_left * d_right)

def kronecker_factor_diag(
    vec: "Diag of mat to be factored",
    ds: "Dimensions of factor matrices",
    k: "Dimension"
) -> "`k`th Kronecker factor of matrix":
    K = len(ds)
    A = factorwise_average_diag(vec, ds, k)
    offset = (K-1)/K * A.sum() / ds[k]
    return A - offset

def kronecker_factor_diag_supplied_d(
    vec: "Diag of mat to be factored",
    d, d_left, d_right,
    K: "Number of dimensions"
) -> "`k`th Kronecker factor of matrix":
    A = factorwise_average_diag_supplied_d(vec, d, d_left, d_right)
    offset = (K-1)/K * A.sum() / d
    return A - offset
    
def ks_project_diag(
    vec: "Diag of mat to be KS-projected",
    ds: "Dimensions of factor matrices",
) -> "Diag of projected matrix":
    """
    Project vec onto the space of KS-decomposable
    matrices such that the Frobenius distance is
    minimized.
    """
    return kron_sum_diag_fast(
        *[kronecker_factor_diag(vec, ds, i) for i in range(len(ds))]
    )

def ks_project_diag_supplied_d(
    vec: "Diag of mat to be KS-projected",
    ds: "Dimensions of factor matrices",
    d_lefts: list[int],
    d_rights: list[int],
    K: "Total number of factor matrices"
) -> "Diag of projected matrix":
    """
    Project vec onto the space of KS-decomposable
    matrices such that the Frobenius distance is
    minimized.
    """
    return kron_sum_diag_fast(
        *[
            kronecker_factor_diag_supplied_d(
                vec,
                ds[i],
                d_lefts[i],
                d_rights[i],
                K
            )
            for i in range(K)
        ]
    )



##############################################
# Utilities for testing video reconstruction #
##############################################

def shuffle_axes(_mat, /, axes: list):
    """
    For every axis, shuffle the matrix on that axis.
    """
    mat = _mat.copy()
    for ax in axes:
        idxs = np.arange(mat.shape[ax+1])
        np.random.shuffle(idxs)
        slices = [slice(None)] * _mat.ndim
        slices[ax+1] = idxs
        mat = mat[tuple(slices)]
    return mat

def reconstruct_axes(_mat, /, axes: list, Psis, first_idx=None):
    """
    Uses Psis in a greedy way to work out the best ordering
    of the data.

    Requires the first axis to be a batch axis.
    """
    orders = []
    for ax in axes:
        Psi = Psis[ax].copy()
        np.fill_diagonal(Psi, 0)
        Psi /= Psi.sum(axis=0, keepdims=True)
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

def quick_kron_project(xs, bs, precision=10):
    """
    Vectorized Laurent Series
    Only works when x << min(bs)
    """
    
    # Preprocess
    old_max_x = xs.max()
    minb = bs.min()
    max_x = minb / 2
    diff = old_max_x - max_x
    bs += diff
    xs -= diff
    recip_bs = 1/ bs
    
    # Laurent Series
    van = np.vander(
      recip_bs,
      precision+1
    ).sum(axis=0)[:-1][::-1]

    return np.einsum(
        "j, ij -> i",
        van,
        np.vander(-xs, precision)[:, ::-1]
    )