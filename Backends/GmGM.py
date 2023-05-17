"""
The Gaussian multi-Graphical Model
"""
# Can't update to 3.10 for the typing | because `matlabengine`
# does not support Python 3.10 :/
# But this is the next best thing!
from __future__ import annotations

import warnings

import numpy as np
from Backends.utilities import nmode_gram, shrink_sparsities
from Backends.utilities import kronecker_factor_diag_supplied_d
from Backends.utilities import kron_sum_diag_faster, kron_sum_diag_fast
from Backends.utilities import ks_project_diag_supplied_d
from Backends.utilities import fast_sum_log
from Backends.utilities import quick_kron_project


class GmGM:
    def __init__(
        self,
        use_fortran: bool = True
    ):
        self.fortran_sum_log_sum_2: callable = lambda: None
        self.fortran_sum_log_sum_3: callable = lambda: None
        self.fortran_project_inv_kron_sum_2: callable = lambda: None
        self.fortran_project_inv_kron_sum_3: callable = lambda: None

        self.USES_FORTRAN = use_fortran
        if use_fortran:
            try:
                from Backends.sum_log_sum import sum_log_sum_2, sum_log_sum_3
                from Backends.project_inv_kron_sum import project_inv_kron_sum_2, project_inv_kron_sum_3
                self.fortran_sum_log_sum_2 = sum_log_sum_2
                self.fortran_sum_log_sum_3 = sum_log_sum_3
                self.fortran_project_inv_kron_sum_2 = project_inv_kron_sum_2
                self.fortran_project_inv_kron_sum_3 = project_inv_kron_sum_3
            except ModuleNotFoundError as err:
                self.USES_FORTRAN = False
                warn_str: str = (
                    f"{err}\nThis is probably because the Fortran files have not been compiled."
                    + "\nYou may want to run the following:\n"
                    + "\t`f2py -c SUM_LOG_SUM.f90 -m sum_log_sum`\n"
                    + "\t`f2py -c PROJECT_INV_KRON_SUM.f90 -m project_inv_kron_sum`\n"
                    + "as the issue is that some Fortran subroutines are not compiled.\n"
                    + "The code will fall back on using a"
                    + " pure Python implementation of the subroutines."
                )
                warnings.warn(warn_str)


    def sum_log(
        self,
        G,
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
        if not self.USES_FORTRAN or K > 3:
            # Python Implementation
            # Has started crashing my computer
            G_inv = kron_sum_diag_faster(G, *[arg for arg in args])
            log_err: float = fast_sum_log(G_inv, 20)
        elif K == 2:
            # Fortran Implementation
            log_err: float = self.fortran_sum_log_sum_2(*args)
        elif K == 3:
            # Fortran Implementation
            log_err: float = self.fortran_sum_log_sum_3(*args)
        return log_err
    
    def update_diffs(
         self,
         diffs,
         e_tildes,
         lambda_t,
         Gs,
         structure,
         axis_names,
         dss,
         d_lefts,
         d_rights,
         Ks
    ) -> None:
        # TODO: Add the Fortran implementation of this
        if not self.USES_FORTRAN or max(Ks) > 3:
            Gs = [
                1 / kron_sum_diag_faster(G, *[
                    lambda_t[axis]
                    for axis in dataset
                ])
                for G, dataset in zip(Gs, structure)
            ]
            for axis in axis_names:
                diffs[axis] = e_tildes[axis].copy()
                for idx, dataset in enumerate(structure):
                    if axis in dataset:
                        ell = dataset.index(axis)
                        diffs[axis] -= kronecker_factor_diag_supplied_d(
                            Gs[idx],
                            dss[idx][ell],
                            d_lefts[idx][ell],
                            d_rights[idx][ell],
                            Ks[idx]
                        )
        else:
            for axis in axis_names:
                diffs[axis] = e_tildes[axis].copy()
                for idx, dataset in enumerate(structure):
                    if axis in dataset:
                        ell = dataset.index(axis)
                        if Ks[idx] == 2:
                            diffs[axis] -= self.fortran_project_inv_kron_sum_2(
                                lambda_t[dataset[0]],
                                lambda_t[dataset[1]]
                            )[ell]
                        elif Ks[idx] == 3:
                            diffs[axis] -= self.fortran_project_inv_kron_sum_3(
                                lambda_t[dataset[0]],
                                lambda_t[dataset[1]],
                                lambda_t[dataset[2]]
                            )[ell]
        
    
    def get_diffs(self, G, diffs, ds, d_lefts, d_rights, e_tilde, *args):
        K = len(args)
        if not self.USES_FORTRAN or K > 3:
            G_inv = kron_sum_diag_faster(G, *[lam for lam in args])
            G = 1/G_inv
            for ell in range(K):
                log_grad = kronecker_factor_diag_supplied_d(
                    G,
                    ds[ell],
                    d_lefts[ell],
                    d_rights[ell],
                    K
                )
                diffs[ell] = e_tilde[ell] - log_grad
        elif K == 2:
            log_grad = self.fortran_project_inv_kron_sum_2(*args)
            for ell in range(K):
                diffs[ell] = e_tilde[ell] - log_grad[ell]
        elif K == 3:
            log_grad = self.fortran_project_inv_kron_sum_3(*args)
            for ell in range(K):
                diffs[ell] = e_tilde[ell] - log_grad[ell]

    def __call__(
        self,
        multi_Ys: dict[tuple[str], np.ndarray],
        tol: float = 1e-3,
        max_iter: int = 1000,
        lr_init: float = 1.,
        sparsities:
        list[float] | dict["Axis Names", float] | None
        = None,
        verbose: bool = False,
        verbose_every: int = 100,
        return_eigendecomp: bool = False,
    ):
        """
        Suppose we have a multiomics dataset of:
            300 people x 50 gut microbes x 12 timestamps
            300 people x 200 metabolites
        
        Then the people axis is shared between the two datasets,
        and we can use this to obtain a more accurate estimate of the
        axis-wise graphs.
        """
        structure, multi_Ys = zip(*multi_Ys.items())

        # Input checking
        if isinstance(multi_Ys, np.ndarray):
            # Only one dataset, turn into
            # singleton tuple
            multi_Ys = (multi_Ys,)
            
        if len(multi_Ys) != len(structure):
            raise ValueError(
                f"Mismatch of actual datasets {len(multi_Ys)=}"
                + f" and structure info {len(structure)=}"
            )
        
        # Useful constants
        dss: list[list["Dimensions"]] = [
            list(Ys.shape[1:])
            for Ys in multi_Ys
        ]
        axis_names: set["Axis Name"] = set.union(
            *(set(dataset) for dataset in structure)
        )
        
        # Add/collapse batches to multi_Ys if needed
        # I.e. (batch_idx_1, batch_idx_2, people, genes)
        # becomes (batch_idx_1*batch_idx_2, people, genes)
        # and (people, genes) becomes (batch_idx, people, genes)
        multi_Ys = tuple(
            Ys.reshape(
                -1,
                *Ys.shape[
                    -len(structure[idx]):
                ]
            )
            for idx, Ys in enumerate(multi_Ys)
        )
        
        if verbose:
            print("Axes in data:")
            print(axis_names)
        
        # True/False indicator for which axes are
        # in which dataset
        presence: dict["Axis Name", list[int]] = {
            axis: [
                dataset.index(axis)
                if axis in dataset
                else -1
                for dataset in structure
            ]
            for axis in axis_names
        }
        
        # Empirical covariance matrices
        # Recall that they were defined as
        # 1/d[\axis] * nmode_gram[axis]
        # Since we want sum_axis d_\axis Ss[axis],
        # we just need nmode_gram[axis]!
        Ss: dict[
            "Axis Name",
            "Empirical Covariance Matrix"
        ]
        Ss = {
            axis: sum(
                nmode_gram(
                    multi_Ys[idx],
                    presence[axis][idx]
                )
                for idx, dataset in enumerate(structure)
                if presence[axis][idx] > -1
            )
            for axis in axis_names
        }
        
        # Calculate eigenvectors of the MLE
        eVs: dict[
            "Axis Name",
            tuple[
                "Eigenvalues",
                "Eigenvectors"
            ]
        ]
        eVs = {
            axis: np.linalg.eigh(Ss[axis])
            for axis in axis_names
        }
        
        es: dict["Axis Name", "Eigenvalues"]
        es = {
            axis: eV[0]
            for axis, eV in eVs.items()
        }
        
        Vs: dict["Axis Name", "Eigenvectors"]
        Vs = {
            axis: eV[1]
            for axis, eV in eVs.items()
        }
        
        # Iterate to find the eigenvalues of the MLE
        itered: dict["Axis Name", "Eigenvalues"]
        itered = self.iter_eigvals_fast(
            es,
            presence=presence,
            axis_names=axis_names,
            dss=dss,
            structure=structure,
            max_iter=max_iter,
            tol=tol,
            lr_init=lr_init,
            verbose=verbose,
            verbose_every=verbose_every
        )
        
        # Combine the eigenvectors and eigenvalues
        # into the estimated precision matrix
        Psis_iter: dict["Axis Name", "Precision Matrix"]
        Psis_iter = {
            axis: (Vs[axis] * itered[axis]) @ Vs[axis].T
            #axis: Vs[axis] @ np.diag(itered[axis]) @ Vs[axis].T
            for axis in axis_names
        }
        
        # If we don't want to regularize, stop here
        if sparsities is None:
            if return_eigendecomp:
                return Psis_iter, (Vs, itered)
            else:
                return Psis_iter
            
        # If we want to regularize, let's threshold it
        shrunk = shrink_sparsities(Psis_iter, sparsities)
        if return_eigendecomp:
            return shrunk, (Vs, itered)
        else:
            return shrunk
        
    def iter_eigvals_fast(
        self,
        es: dict["Axis Name", "Eigenvalues"],
        presence: dict["Axis Name", list[int]],
        axis_names: set["Axis Names"],
        dss: list[list["Dimensions"]],
        structure: list[tuple["Axis Names"]],
        max_iter: int = 1000,
        tol: float = 1e-3,
        lr_init: float = 1.,
        verbose: bool = True,
        verbose_every: int = 100,
    ):
        """
        Iterative subroutine to solve for the eigenvalues
        """
        
        # Useful constants
        Ks: list["Number of dimensions"] = [
            len(ds)
            for ds in dss
        ]
        d_alls: list[int] = [
            np.prod(ds).astype(int)
            for ds in dss
        ]
        d_lefts: list[list[int]] = [
            np.cumprod([1] + ds).astype(int)[:-1]
            for ds in dss
        ]
        d_rights: list[list[int]] = [
            np.cumprod((ds + [1])[::-1]).astype(int)[::-1][1:]
            for ds in dss
        ]
        
        # Prep eigenvalues for iteration
        e_tildes: dict["Axis Name", "Eigenvalues"] = {
            axis: e
            for axis, e in es.items()
        }
        e_normalizer: float = sum(
            np.linalg.norm(e)
            for axis, e in e_tildes.items()
        )
        e_tildes = {
            axis: e / e_normalizer
            for axis, e in e_tildes.items()
        }
        
        # TODO: Move `max_small_steps` to be func parameter
        # This is how many times we make a small move before
        # we conclude that we have converged!
        num_small_steps = 0
        max_small_steps = 5
        max_line_search_steps = 20 # TODO: also make this a param
        
        # Initial values for iterative variables
        lr_t: float = lr_init
        prev_err: float = np.inf
        lambda_t: dict["Axis Name", "Predicted Eigenvalues"]
        lambda_t = {
            axis: np.ones(e.shape)
            for axis, e in e_tildes.items()
        }
        diffs: dict["Axis Name", "Updates to Eigenvalues"] = {
            axis: np.zeros_like(e)
            for axis, e in e_tildes.items()
        }
        Gs: list["Kronecker Sum Eigenvalues"] = [
            np.zeros(d_all)
            for d_all in d_alls
        ]
        
        for i in range(max_iter):
            self.update_diffs(
                diffs,
                e_tildes,
                lambda_t,
                Gs,
                structure,
                axis_names,
                dss,
                d_lefts,
                d_rights,
                Ks
            )
                        
            # Calculate the error
            log_err: float = 0
            for idx, G in enumerate(Gs):
                log_err -= self.sum_log(
                    G,
                    *[
                        lam
                        for axis, lam
                        in lambda_t.items()
                        if axis in structure[idx]
                    ]
                )
            
            # Backtracking line search
            for line_step in range(max_line_search_steps):
                # Decrease step size each time
                # (`line_step` starts at 0, i.e. no decrease)
                step_lr: float = lr_t / 10**line_step
                
                for axis in axis_names:
                    lambda_t[axis] -= step_lr * diffs[axis]

                # Since all tuplets of eigenvalues
                # get summed together within each dataset,
                # the minimum final eigenvalue is 
                # the sum of minimum axis eigenvalues
                minimum_diag: float = min(
                    sum(lambda_t[axis].min() for axis in dataset)
                    for dataset in structure
                )

                # If and only if the minimum final eigenvalue is less than zero
                # have we left the positive definite space we desire
                # to stay in, so we will have to backtrack
                if minimum_diag <= 1e-8:
                    for axis in axis_names:
                        lambda_t[axis] += step_lr * diffs[axis]
                else:
                    # Found good step size
                    break
            else:
                # Did not find a good step size
                if verbose:
                    print(f"@{i}: {prev_err} - Line Search Gave Up!")
                break
                
            # Calculate total err
            other_err: float = sum(
                e_tildes[axis]
                @ lambda_t[axis]
                for axis in axis_names
            )
            err: float = log_err + other_err

            # Calculate the change in error and
            # whether or not we can consider
            # ourselves to be converged
            err_diff: float = np.abs(prev_err - err)
            prev_err: float = err
            if err_diff/np.abs(err) < tol:
                num_small_steps += 1
                if num_small_steps >= max_small_steps:
                    if verbose:
                        print(f"Converged! (@{i}: {err})")
                    break
            else:
                mum_small_steps = 0

            if verbose:
                if i % verbose_every == 0:
                    print(f"@{i}: {err} ({log_err} + {other_err}) ∆{err_diff / np.abs(err)}")
        else:
            if verbose:
                print("Did not converge!")
            
        return lambda_t