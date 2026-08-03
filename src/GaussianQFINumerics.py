import warnings

import numpy as np
import scipy as sp
from tqdm.autonotebook import tqdm

Omega1 = np.array([
        [0,1],
        [-1,0]
])
def Omega(n):
    """Gives the n mode symplectic form
    Parameters
    ----------
    n: np.double
        The number of modes
    Returns
    -------
    np.double[2*n,2*n]
        The n mode symplectic form
    """
    return sp.linalg.block_diag(*([Omega1]*n))
def will_decomp(sigma):
    """Calculates the Williamson Decomposition of a symmetric matrix
    Parameters
    ----------
    sigma: np.cdouble[2n,2n]
        The symmetric matrix matrix
    Returns
    -------
    [np.cdouble[2n,2n],np.cdouble[n]]
        S and nu s.t. S@diag(*nu)@transpose(S) = sigma
    """
    n = int(len(sigma)/2)
    Omegaval = Omega(n)
    sqrt = sp.linalg.sqrtm(sigma)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="invalid value encountered in divide")
        warnings.filterwarnings("ignore", message="divide by zero encountered in divide")
        warnings.filterwarnings("ignore", message="divide by zero encountered in reciprocal")
        # May be faster to invert the square-rooted matrix, though more ideally we could batch these
        #invsqrt = sp.linalg.fractional_matrix_power(sigma,-.5)
        invsqrt = np.linalg.inv(sqrt)
        Psi = invsqrt@Omegaval@invsqrt
    #[T,Z,_] = sp.linalg.schur(Ψ,output='real',sort='lhp')
        [T,Z] = sp.linalg.schur(Psi,output='real')
        I2 = np.array([[1,0],[0,1]])
        Px = np.array([[0,1],[1,0]])
        Pil = [I2 if T[2*i,2*i+1] >0 else Px for i in range(n)]
        Pi = sp.linalg.block_diag(*Pil)
        Dinv = np.abs(np.linalg.diagonal(T@Omegaval))
        Dinvsqrt = np.sqrt(Dinv)
        #Dt =  np.reciprocal(Dinv)
        St = sqrt@Z@Pi@np.diag(Dinvsqrt)
        # Correct numerical issues, these were initially caused by a typo,so they shouldn't be present anymore
        #err = -St@Ω@np.transpose(St)@Ω
        #print(err)
        #fix = sp.linalg.fractional_matrix_power(err,-.5)
    S = St#fix@St
    Sinv = -Omegaval@np.transpose(S)@Omegaval
    SinvT = np.transpose(Sinv)
    nu = [(Sinv@sigma@SinvT)[2*i][2*i] for i in range(n)]
    # D = np.linalg.diagonal(err@np.diag(Dt))
    #print(St@np.diag(Dt)@np.transpose(St)-sigma)
    #print(np.round(St@Ω@np.transpose(St),5))
    #print(Dt)
    #print(np.round(S@Ω@np.transpose(S),8))
    return [S,nu]
def _SLD_precomp(sigma):
    [S,ν] = will_decomp(sigma)
    n = int(len(sigma)/2)
    Omegaval = Omega(n)
    Sinv = -Omegaval@np.transpose(S)@Omegaval
    SinvT = np.transpose(Sinv)
    def M(j,k,m):
        Mval = np.zeros([2*n,2*n])
        outputs = {
            0: [[0,1],[-1,0]],
            1: [[0,1],[1,0]],
            2: [[1,0],[0,1]],
            3: [[1,0],[0,-1]]
        }
        Mval[(2*j):(2*j+2),(2*k):(2*k+2)] = outputs[m]
        return Mval
    newM = [[[SinvT@M(j,k,m)@Sinv for m in range(n)] for k in range(n)] for j in range(n)]
    return [newM,ν]
    
def SLD(sigma,dσ,precomp = None):
    """Calculates the Symmetric Logarithmic Derivative
    Parameters
    ----------
    sigma: np.cdouble[2m,2m]
        The covariance matrix
    dsigma: np.cdouble[n,2m,2m]
        Derivative of the covariance matrix
    precomp: [np.cdouble[2m,2m][m,m],np.cdouble[m]]
        A precomputation step useful in the case where we want to calculate SLDs for multiple parameters
    Returns
    -------
    np.cdouble[6,6]
        The Symmetric Logarithmic Derivative
    """
    if precomp is None:
        precomp = _SLD_precomp(sigma)
    newM = precomp[0]
    m = int(len(sigma)/2)
    ν=precomp[1]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="invalid value encountered in divide")
        warnings.filterwarnings("ignore", message="divide by zero encountered in divide")
        Lsum = [[[ newM[i][j][k] * 1/(ν[i]*ν[j] - (-1)**k)*np.linalg.trace(newM[i][j][k]@dσ) for i in range(m)] for j in range(m)] for k in range(m)]
    L = np.nansum(Lsum,axis=(0,1,2))
    return L
    #print(ν)
    #print(np.round(Sinv@σ@SinvT,8))
def FIM(σ,dσs):
    """Calculates the Quantum Fisher Information Matrix
    Parameters
    ----------
    sigma: np.cdouble[2m,2m]
        The covariance matrix
    dsigmas: np.cdouble[n,2m,2m]
        An array of derivatives of the covariance matrix
    Returns
    -------
    np.cdouble[n,n]
        The Quantum Fisher Information matrix
    """
    precomp = _SLD_precomp(σ)
    Ls = [SLD(σ,dσ,precomp) for dσ in dσs]
    FIM = np.array([[ np.linalg.trace(L@dσ)/2 for L in Ls] for dσ in dσs])
    return FIM
    #FIη11 = np.linalg.trace(Lη1@dσdη1)/2
    #FIη22 = np.linalg.trace(Lη2@dσdη2)/2
    #FIη12 = np.linalg.trace(Lη1@dσdη2)/2
    #FIη21 = np.linalg.trace(Lη2@dσdη1)/2
    #FIM = np.array([[FIη11,FIη12],[FIη21,FIη22]])
    #print(FIM)
#def FIE(FIM,a):
#    # Currently only written for 2 parameter estimation
#    aeff = a #if (np.abs(a) > .5) else (-a+ np.sign(a))
#    Bm1 = [[aeff,0],[np.sqrt(1-aeff**2),1]]
#    Qtm1 = np.transpose(Bm1)@np.linalg.inv(FIM)@Bm1
#    FIE = 1/Qtm1[0,0]
#    return FIE
#    #print(FIE)
#    #print(np.linalg.inv(FIM))
def vectorized_QFIM(σfunc,dσfuncs,*args):
    """Calculate the Quantum Fisher Information Matrix for a grid of points
    Parameters
    ----------
    sigmafunc: function(*args)
        The function that gives your covariance matrix given *args
    dsigmafuncs: function(*args)
        An array of functions that give derivatives of the covaraince matrix given *args
    args: *args[]
        Arrays for each argument being evaluated
    Returns
    -------
    np.cdouble[arggrid.size,len(dsigmafuncs),len(dsigmafuncs)]
        The Quantum Fisher Information matrix at each grid point
    """
    # This code is very general, and as a result some of this needs more explination than normal
    FIMlen = len(dσfuncs)
    basesize = [len(arg) for arg in args]
    QFIMs = np.zeros(basesize+[FIMlen,FIMlen],np.cdouble)
    # Create an iterable that goes through all points in our grid
    iterable = np.ndindex(tuple(basesize))
    total = np.prod(basesize)
    # We use tqdm here since this can potentially take a while to run
    for it in tqdm(iterable,total=total):
        # For each meshgrid we grab it's value at our current point
        gridelem = [args[i][it[i]] for i in range(len(basesize))]
        # Then evaluate each function at these values
        σval = σfunc(*gridelem)
        dσs = [dσfunc(*gridelem) for dσfunc in dσfuncs]
        #with warnings.catch_warnings(record=True) as w:
        FIMval = FIM(σval,dσs)
        #    if len(w) >0:
        #        print(gridelem)
        # Finally record the matrix at the proper point in the grid
        QFIMs[*it,:,:] = FIMval
    return QFIMs
