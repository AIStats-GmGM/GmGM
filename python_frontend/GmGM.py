from __future__ import annotations

import numpy as np

from typing import Optional

from extract_d_values import extract_d_values

# Imports functions of form project_inv_kron_sum_{2,3,4,5,6}
from project_inv_kron_sum import *

# Imports functions of form sum_log_sum_{2,3,4,5,6}
from sum_log_sum import *

from regularizers import Regularizer


def get_all_axes(
    axis_names: dict[str, tuple[str]]
) -> set[str]:
    """
    Gets a set of every axis in the dataset, i.e.
    [("A", "B"), ("A", "B")]
    => {"A", "B", "C"}
    """
    return {
        axis
        for axes in axis_names.values()
        for axis in axes
    }

def get_axis_presence(
    axis_names: dict[str, tuple[str]]
) -> dict[str, dict[str, Optional[int]]]:
    """
    Returns a dictionary keyed by axis names,
    indicating which datasets the axis is present in

    We assume each axis appears at most once per dataset
    """

    all_axes: set[str] = get_all_axes(axis_names)

    to_return: dict[str, dict[str, Optional[int]]] = {}

    for axis in all_axes:
        to_return[axis] = {}
        for dataset_name, axes in axis_names.items():
            to_add = [
                idx
                for idx, ax
                in enumerate(axes)
                if ax == axis
            ]

            if len(to_add) == 0:
                to_return[axis][dataset_name] = None
            elif len(to_add) == 1:
                to_return[axis][dataset_name] = to_add[0]
            else:
                raise Exception("Don't allow repeated axes")

    

    return to_return


def nmode_gram(A: np.ndarray, n: int, nice_strides: bool = True) -> np.ndarray:
    """
    Computes the gram matrix A @ A^T
    The "outer axis" for this multiplication is
    axis n, the inner axes are all the rest flattened into
    matrix form.

    Intuitively, this is just finding a normal covariance matrix,
    where we treat every axis other than n as a sample axis.
    """
    An = np.reshape(
        np.moveaxis(A, n, 0),
        (A.shape[n], -1), # The -1 infers the value for (d_{\n})
        order='F' # Do math vectorization order rather than numpy vectorization order
    )

    to_return = An @ An.T
    
    return to_return

def get_gram_matrices(
    dataset: dict[str, np.ndarray],
    axis_presence: dict[str, dict[str, int]],
    axis_names: dict[str, tuple[str]],
    batch_axes: set[str] = set({})
) -> dict[str, np.ndarray]:
    """
    Return a dictionary keyed by *axis* name,
    with the entry being the average gram matrix
    for that axis.
    """

    to_return: dict[str, np.ndarray] = {}
    weighting: dict[str, int] = {}

    for axis, presence in axis_presence.items():
        if axis in batch_axes:
            continue
        # We assume each axis only appears once per tensor
        for dataset_name, loc in presence.items():
            if loc is None:
                continue
            tensor: np.ndarray = dataset[dataset_name]
            gram: np.ndarray = nmode_gram(tensor, loc)
            if axis not in to_return:
                to_return[axis] = gram
                weighting[axis] = np.prod(tensor.shape) // tensor.shape[loc]
            else:
                to_return[axis] += gram
                weighting[axis] += np.prod(tensor.shape) // tensor.shape[loc]
        
    # Downweight every gram matrix by the number of samples
    for axis in to_return.keys():
       to_return[axis] /= weighting[axis]

    return to_return

def get_eigenvalues_and_eigenvectors(
    gram_matrices: dict[str, np.ndarray]
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray]
]:
    """
    Returns (eigenvalues, eigenvectors)
    """

    eVs: dict[str, tuple[np.ndarray, np.ndarray]] = {
        axis: np.linalg.eigh(gram_matrix)
        for axis, gram_matrix in gram_matrices.items()
    }
    es: dict[str, np.ndarray] = {
        axis: eV[0]
        for axis, eV in eVs.items()
    }
    Vs: dict[str, np.ndarray] = {
        axis: eV[1]
        for axis, eV in eVs.items()
    }

    return es, Vs

def project_inv_kron_sum(
    evals: dict[str, np.ndarray],
    axis_names: dict[str, tuple[str]],
    tensor_name: str,
    K: int
):
    if K == 2:
        return project_inv_kron_sum_2(
            evals[axis_names[tensor_name][0]],
            evals[axis_names[tensor_name][1]]
        )
    elif K == 3:
        return project_inv_kron_sum_3(
            evals[axis_names[tensor_name][0]],
            evals[axis_names[tensor_name][1]],
            evals[axis_names[tensor_name][2]]
        )
    elif K == 4:
        return project_inv_kron_sum_4(
            evals[axis_names[tensor_name][0]],
            evals[axis_names[tensor_name][1]],
            evals[axis_names[tensor_name][2]],
            evals[axis_names[tensor_name][3]]
        )
    elif K == 5:
        return project_inv_kron_sum_5(
            evals[axis_names[tensor_name][0]],
            evals[axis_names[tensor_name][1]],
            evals[axis_names[tensor_name][2]],
            evals[axis_names[tensor_name][3]],
            evals[axis_names[tensor_name][4]]
        )
    elif K == 6:
        return project_inv_kron_sum_6(
            evals[axis_names[tensor_name][0]],
            evals[axis_names[tensor_name][1]],
            evals[axis_names[tensor_name][2]],
            evals[axis_names[tensor_name][3]],
            evals[axis_names[tensor_name][4]],
            evals[axis_names[tensor_name][5]]
        )
    else:
        raise ValueError(
            "Only 2-6 way tensors are supported"
            + " but you can easily edit the fortran files"
            + " to support more."
        )
    
def sum_log_sum(
        *args: list[np.array]
    ) -> float:
        """
        Computes:
            the sum
            of the log
            of the determinant
            of the kronecker sum
            of the input matrices 
        """
        K = len(args)
        if K == 2:
            log_err: float = sum_log_sum_2(*args)
        elif K == 3:
            log_err: float = sum_log_sum_3(*args)
        elif K == 4:
            log_err: float = sum_log_sum_4(*args)
        elif K == 5:
            log_err: float = sum_log_sum_5(*args)
        elif K == 6:
            log_err: float = sum_log_sum_6(*args)
        else:
            raise ValueError(
                "Only 2-6 way tensors are supported"
                + " but you can easily edit the fortran files"
                + " to support more."
            )
        return log_err

def get_log_err(
    evals: dict[str, np.ndarray],
    axis_names: dict[str, tuple[str]],
) -> float:
    """
    Calculates log-determinant portion of the error
    """
    log_err: float = 0
    for _, axes in axis_names.items():
        log_err -= sum_log_sum(
            *[
                eval
                for axis, eval
                in evals.items()
                if axis in axes
            ]
        )
    return log_err

def get_trace_err(
    es: dict[str, np.ndarray],
    evals: dict[str, np.ndarray],
    all_axes: set[str]
) -> float:
    """
    Returns the trace portion of the error
    """
    trace_err: float = sum(
                es[axis]
                @ evals[axis]
                for axis in all_axes
            )
    return trace_err

def get_reg_err(
    evals: dict[str, np.ndarray],
    evecs: dict[str, np.ndarray],
    regularizing: bool,
    regularizer: Regularizer
) -> float:
    """
    Returns the regularization penalty
    portion of the error
    """
    if regularizing:
        reg_err: float = regularizer.loss(
            evals,
            evecs,
        )
    else:
        reg_err: float = 0
    return reg_err

def GmGM(
    dataset: dict[str, np.ndarray],
    axis_names: dict[str, tuple[str]],
    *,
    batch_axes: set[str] = set({""}),
    max_small_steps: int = 5,
    max_line_search_steps: int = 20,
    lr_init: float = 1.0,
    max_iter: int = 1000,
    tol: float = 1e-3,
    regularizer: Optional[Regularizer] = None,
    return_eigendata: bool = False,
    force_posdef: bool = True,
    gram_matrices: Optional[dict[str, np.ndarray]] = None,
    verbose: bool = False,
    verbose_every: int = 100,
    _always_regularize: bool = False,
    _check_overstep_each_iter: bool = False,
    prior: Optional[dict[str, np.ndarray]] = None,
    prior_type: Optional[dict[str, str]] = None,
) -> tuple[
    dict[str, np.ndarray],
    Optional[dict[str, np.ndarray]]
]:
    """
    Takes in your dataset tensors, and, for each axis,
    calculates the Precision Matrix and Covariance Matrix

    `regularizer` maps (evals, evecs -> float)
    `rhos` contains regularization strengths

    `force_posdef` forces the precision matrix for
        each mode to be positive definite
        (this is not necessarily required by the model)
    """

    all_axes: set[str] = get_all_axes(
        axis_names
    ) - batch_axes

    # Get which datasets axes appear in,
    # and where they appear in the tensor
    presences: dict[str, dict[str, Optional[int]]] = get_axis_presence(
        axis_names
    )

    # Get d_\forall, d_{<\ell}, and d_{>\ell}
    full_sizes: dict[str, np.ndarray] = {}
    left_sizes: dict[str, np.ndarray] = {}
    right_sizes: dict[str, np.ndarray] = {}

    for tensor_name, tensor in dataset.items():
        full_size, left_size, right_size = extract_d_values(
            tensor.shape
        )
        full_sizes[tensor_name] = full_size
        left_sizes[tensor_name] = left_size
        right_sizes[tensor_name] = right_size

    # Get the number of dimensions for each tensor
    Ks: dict[str, int] = {
        dataset_name: len(
            set(axis_names[dataset_name]) - batch_axes
        )
        for dataset_name, tensor in dataset.items()
    }

    # Get S_\ell
    if gram_matrices is None:
        gram_matrices = get_gram_matrices(
            dataset,
            presences,
            axis_names,
            batch_axes=batch_axes
        )

    # Incorporate priors
    if prior is None:
        prior = {}
    if prior_type is None:
        prior_type = {}
    for axis, prior_matrix in prior.items():
        if axis not in prior_type:
            continue
        if prior_type[axis] == "wishart":
            # Eta = -1/2 * prior^-1
            # Drop the 1/2 as we ignore it everywhere
            # Subtract from Gram as in Theorem 3
            gram_matrices[axis] -= -np.linalg.inv(prior_matrix)
        else:
            raise ValueError(
                f"Unrecognized prior type: {prior_type[axis]}"
            )

    # Get the eigenvalues and eigenvectors
    es, evecs = get_eigenvalues_and_eigenvectors(
        gram_matrices
    )

    # Convert eigenvectors to fortran order
    # This will make large multiplications faster, surprisingly!
    evecs = {
        axis: np.asfortranarray(evec)
        for axis, evec in evecs.items()
    }

    # Remove the batch axes
    axis_names = {
        tensor_name: tuple(
            axis_name
            for axis_name in axis_names[tensor_name]
            if axis_name not in batch_axes
        )
        for tensor_name in axis_names
    }

    # Recalculate presences
    presences = get_axis_presence(
        axis_names
    )

    """
    When the trace term is large, we expect the eigenvalues to be small, as naive
    Gram matrix inversion would have the eigenvalues be the inverse of the trace term.
    This puts us close to the boundary of the positive definite cone, which is why
    it performs poorly - we have to tread to lightly and end up "converging" before
    we actually reach the optimum.

    Downscaling the trace term to be around 1 will mean the reciprocal also stays around 1,
    which keeps us away from the boundary.
    """
    es = {
       axis: es[axis] / np.linalg.norm(es[axis])
       for axis in all_axes
    }

    # Initialize looping variables
    num_small_steps: int = 0
    lr_t: float = lr_init
    prev_err: float = np.inf
    regularizing: bool = _always_regularize and not regularizer is None

    # Eigenvalues of the mle, keyed by axis
    evals: dict[str, np.ndarray] = {
        axis: np.ones_like(e)
        for axis, e in es.items()
    }

    # Changes in eigenvalues in current iteration
    # keyed by axis
    diffs: dict[str, np.ndarray] = {
        axis: np.zeros_like(e)
        for axis, e in es.items()
    }

    # Converge to eigenvalue MLE
    for i in range(max_iter):

        # Compute MLE gradient
        for axis, locations in presences.items():
            if axis in batch_axes:
                continue
            diffs[axis] = es[axis].copy()
            for tensor_name, ell in locations.items():
                if ell is None:
                    continue
                # Note to self: it seems project_inv_kron_sum
                # can be removed, because it doesn't affect much...
                # Presumably the log determinant is much smaller in effect
                # than the trace...
                diffs[axis] -= project_inv_kron_sum(
                    evals,
                    axis_names,
                    tensor_name,
                    Ks[tensor_name],
                )[ell]

                if axis in prior_type:
                    if prior_type[axis] == "wishart":
                        # h = -1/2 * precmat^-1
                        # drop the 1/2 as we ignore it everywhere
                        diffs[axis] += -1/evals[axis]

        # Add regularization if necessary
        # Only activates after converging to the MLE
        if regularizing:
            regs: dict[str, np.ndarray] = regularizer.grad(
                evals,
                evecs,
            )
            for axis in all_axes:
                diffs[axis] += regs[axis]

        # Backtracking line search
        line_search_gave_up: bool = False
        lr_t: float = lr_init
        for line_step in range(max_line_search_steps):
            # Decrease step size each time
            # (`line_step` starts at 0, i.e. no decrease)
            step_lr: float = lr_t / 10**line_step
            
            for axis in all_axes:
                evals[axis] -= step_lr * diffs[axis]

            # Since all tuplets of eigenvalues
            # get summed together within each dataset,
            # the minimum final eigenvalue is 
            # the sum of minimum axis eigenvalues
            if not force_posdef:
                minimum_diag: float = min(
                    sum(evals[axis].min() for axis in axes)
                    for axes in axis_names.values()
                )
            # Or we can enforce individual posdefness!
            else:
                minimum_diag = min(
                min(evals[axis].min() for axis in axes)
                for axes in axis_names.values()
                )

            # If the minimum eigenvalue is less than zero
            # have we left the positive definite space we desire
            # to stay in, so we will have to backtrack
            if minimum_diag <= 1e-8:
                for axis in all_axes:
                    evals[axis] += step_lr * diffs[axis]
                continue
            

            # Check if error has gotten worse
            if _check_overstep_each_iter:
                log_err: float = get_log_err(evals, axis_names)
                trace_err: float = get_trace_err(es, evals, all_axes)
                reg_err: float = get_reg_err(evals, evecs, regularizing, regularizer)
                err: float = log_err + trace_err + reg_err
                if err > prev_err:
                    for axis in all_axes:
                        evals[axis] += step_lr * diffs[axis]
                    continue
            
            # If it got here, we have a good step size
            break
        else:
            # Did not find a good step size
            if verbose:
                print(f"@{i}: {prev_err} - Line Search Gave Up!")
            line_search_gave_up = True
            num_small_steps = max_line_search_steps + 1

        # Calculate the error
        if not _check_overstep_each_iter:
            log_err: float = get_log_err(evals, axis_names)
            trace_err: float = get_trace_err(es, evals, all_axes)
            reg_err: float = get_reg_err(evals, evecs, regularizing, regularizer)
            err: float = log_err + trace_err + reg_err

        # Calculate the change in error and
        # whether or not we can consider
        # ourselves to be converged
        err_diff: float = np.abs(prev_err - err)
        prev_err: float = err
        if err_diff/np.abs(err) < tol or line_search_gave_up:
            num_small_steps += 1
            if num_small_steps >= max_small_steps:
                if verbose:
                    print(f"Converged! (@{i}: {err})")
                if not regularizing and regularizer is not None:
                    if verbose:
                        print("Regularizing!")
                    tol /= 10
                    regularizing = True
                    num_small_steps = 0
                else:
                    break
        else:
            num_small_steps = 0

        if verbose:
            if i % verbose_every == 0:
                print(f"@{i}: {err} ({log_err} + {trace_err} + {reg_err}) ∆{err_diff / np.abs(err)}")
    else:
        # This triggers if we don't break out of the loop
        if verbose:
            print("Did not converge!")

    # Calculate the precision matrices
    precisions: dict[str, np.ndarray] = {
        axis: (evecs[axis] * evals[axis]) @ evecs[axis].T
        for axis in all_axes
    }
    if return_eigendata:
        return precisions, evals, evecs, es
    else:
        return precisions