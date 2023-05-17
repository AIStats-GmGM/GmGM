"""
This is a python wrapper for TeraLasso
"""

import matlab.engine
import io
import numpy as np
from Backends.utilities import nmode_gram

eng = matlab.engine.start_matlab()
eng.addpath(
    './teralasso'
)


def TeraLasso(Ys, betas, max_iter=100):
    n, *d = Ys.shape
    
    K = len(Ys.shape[1:])
    Ss = [
        matlab.double(
            nmode_gram(Ys, ell)
        )
        for ell in range(K)
    ]
    
    d_matlab = matlab.double(d)
    betas_matlab = matlab.double(betas)

    Psis_matlab = eng.teralasso(
        Ss,
        d_matlab,
        'L1',
        0,
        1e-8,
        betas_matlab,
        max_iter,
        nargout=1,
        stdout=io.StringIO()
    )
    
    Psis = []
    
    for Psi in Psis_matlab:
        Psis.append(np.asarray(Psi))
    
    return Psis
    
def TeraLasso_cov(Ss, betas, max_iter=100):
    
    d = [S.shape[0] for S in Ss]
    Ss = [matlab.double(S) for S in Ss]
    
    d_matlab = matlab.double(d)
    betas_matlab = matlab.double(betas)

    Psis_matlab = eng.teralasso(
        Ss,
        d_matlab,
        'L1',
        0,
        1e-8,
        betas_matlab,
        max_iter,
        nargout=1,
        stdout=io.StringIO()
    )
    
    Psis = []
    
    for Psi in Psis_matlab:
        Psis.append(np.asarray(Psi))
    
    return Psis