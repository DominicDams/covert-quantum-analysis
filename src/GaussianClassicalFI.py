import itertools

import numpy as np
import scipy as sp
import thewalrus as wr
import thewalrus.quantum
from tqdm.auto import tqdm


def T(n):
    """Defines conversion matrix for thewalrus
    T.T@sigma@T is our converted matrix
    T@sigma@T.T converts back
    """
    v1 = np.array([[1,0]])
    v2 = np.array([[0,1]])
    T1 = sp.linalg.block_diag(*([v1]*n))
    T2 = sp.linalg.block_diag(*([v2]*n))
    T = np.block([[T1],[T2]])
    return T
def vec_Ps_generic_jac_gen(cov,arg,maxval):
    means = np.zeros(4)
    T2 = T(2)
    def vec_Ps_su_jac(x):
        output = np.zeros((*xvals.shape,maxval,maxval))
        for i,val in np.ndenumerate(xvals):
            covval = T2@cov(xvals,arg)@T2
            Ps = wr.quantum.probabilities(means,covval,maxval)
            output[*i,:,:] = Ps
        return output
    return vec_Ps_su_jac
# TODO: make this function more generic. Proposed design -
# Takes in a function, a set of test parameters, and a set of parameters to differentiate by
# and the maxval
# Jam the test parameters and 
def FI_generic(etas,nss,nbs,cov,maxval=10):
    """Determine the Classical Fisher information from a function that generates a covariance matrix
    """
    ds = np.zeros((len(nss),len(nbs),maxval,maxval,len(etas)))
    # iterable = tqdom(itertools.product(map(enumerate,parameters)

    for (i,ns),(j,nb) in tqdm(itertools.product(enumerate(nss),enumerate(nbs)),total=len(nss)*len(nbs),smoothing=.01):
        vec_ps_su_jac = vec_Ps_generic_jac_gen(cov,ns,nb,maxval)
        derivres = sp.differentiate.jacobian(vec_ps_su_jac,etas,initial_step=1e-5)
        ds[i,j,:,:,:] = derivres.df
    T2 = T(2)
    means = np.zeros(4)
    ps = np.zeros((len(nss),len(nbs),maxval,maxval,len(etas)))
    for (i,ns),(j,nb),(k,eta) in tqdm(itertools.product(enumerate(nss),enumerate(nbs),enumerate(etas)),total=len(nss)*len(nbs)*len(etas),smoothing=.01):
        covval = T2@cov(eta,ns,nb)@T2
        Ps = wr.quantum.probabilities(means,covval,maxval)
        ps[i,j,k,:,:] = Ps
    presum = ds**2/ps
    presum[~np.isfinite(presum)] = 0
    fi = np.sum(presum,axis=(-1,-2))
    return fi
