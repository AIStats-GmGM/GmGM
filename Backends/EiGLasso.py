"""
This is a python wrapper for EiGLasso
"""

import matlab.engine
import io
import numpy as np

# To compile the EiGLasso file, I ran:
#
# /Applications/MATLAB_R2022a.app/bin/maci64/mex
#     -output eiglasso_joint eiglasso_joint_mex.cpp -lmwlapack
#
# A similar command should work elsewhere, you might have
# to change the Matlab version

eng = matlab.engine.start_matlab()

eng.addpath(
    './EiGLasso/EiGLasso_JMLR'
)

def EiGLasso(
    Ys: "(m, n, p) matrix of m samples",
    beta_1: "Regularization for Psi" = 0.01,
    beta_2: "Regularization for Theta" = 0.01,
    Psi_init: "nxn matrix" = None,
    Theta_init: "pxp matrix" = None,
    K_EIG: int = 1,
    max_Newton_iter: int = 10_000,
    max_line_iter: int = 20,
    newton_tol: float = 1e-3,
    sigma: float = 0.01,
    tr_ratio: float = 0,
    verbose: bool = False
):
    (m, n, p) = Ys.shape
    T = np.einsum("mnp, mlp -> nl", Ys, Ys) / (m*p)
    S = np.einsum("mnp, mnl -> pl", Ys, Ys) / (m*n)
    return EiGLasso_cov(T, S, beta_1, beta_2)
    
def EiGLasso_cov(
    T,
    S,
    beta_1: "Regularization for Psi" = 0.01,
    beta_2: "Regularization for Theta" = 0.01,
    Psi_init: "nxn matrix" = None,
    Theta_init: "pxp matrix" = None,
    K_EIG: int = 1,
    max_Newton_iter: int = 10_000,
    max_line_iter: int = 20,
    newton_tol: float = 1e-3,
    sigma: float = 0.01,
    tr_ratio: float = 0,
    verbose: bool = False
):
    # Convert to matlab format
    T_ = matlab.double(T)
    S_ = matlab.double(S)
    beta_1 = matlab.double(beta_1)
    beta_2 = matlab.double(beta_2)
    K_EIG = matlab.int64(K_EIG)
    max_Newton_iter = matlab.int64(max_Newton_iter-1)
    max_line_iter = matlab.int64(max_line_iter-1)
    newton_tol = matlab.double(newton_tol)
    sigma = matlab.double(sigma)
    tr_ratio = matlab.double(tr_ratio)
    stdout = {} if verbose else {"stdout": io.StringIO()}
    
    if Psi_init is None:
        Psi_init = np.linalg.inv(T)
    Psi_init = matlab.double(Psi_init)
    if Theta_init is None:
        Theta_init = np.linalg.inv(S)
    Theta_init = matlab.double(Theta_init)
    
    # Call matlab (which itself calls the
    # compiled c++ file `eiglasso_joint_mex.cpp`)
    Theta, Psi, ts, fs = eng.eiglasso_joint(
        S_,
        T_,
        beta_2,
        beta_1,
        nargout=4,
        **stdout
    )
    """
    # Something is going very wrong when
    # I try to add more arguments
        Theta_init,
        Psi_init,
        K_EIG,
        max_Newton_iter,
        max_line_iter,
        newton_tol,
        sigma,
        tr_ratio,
        nargout=4,
        **stdout
    )
    """
    
    # Convert to python format
    Theta = np.asarray(Theta)
    Psi = np.asarray(Psi)
    
    # They're not symmetric though...
    Theta = (Theta + Theta.T) / 2
    Psi = (Psi + Psi.T) / 2
    
    return Psi, Theta