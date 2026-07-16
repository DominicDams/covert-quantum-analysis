import numpy as np
import scipy as sp
from tqdm.autonotebook import tqdm
import warnings

Ω = np.array([
        [0,1,0,0,0,0],
        [-1,0,0,0,0,0],
        [0,0,0,1,0,0],
        [0,0,-1,0,0,0],
        [0,0,0,0,0,1],
        [0,0,0,0,-1,0]
])
def will_decomp(σ):
    """Calculates the Williamson Decomposition of a symmetric matrix
    Parameters
    ----------
    sigma: np.cdouble[6,6]
        The symmetric matrix matrix
    Returns
    -------
    [np.cdouble[6,6],np.cdouble[3]]
        S and v s.t. S@diag(v[0],v[0],v[1],v[1],v[2],v[2])@transpose(S) = sigma
    """
    # Currently built for 6x6 covariance matricies, but could be adapted for other matricies
    sqrt = sp.linalg.sqrtm(σ)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="invalid value encountered in divide")
        warnings.filterwarnings("ignore", message="divide by zero encountered in divide")
        warnings.filterwarnings("ignore", message="divide by zero encountered in reciprocal")
        # May be faster to invert the square-rooted matrix, though more ideally we could batch these
        #invsqrt = sp.linalg.fractional_matrix_power(σ,-.5)
        invsqrt = np.linalg.inv(sqrt)
        Ψ = invsqrt@Ω@invsqrt
    #[T,Z,_] = sp.linalg.schur(Ψ,output='real',sort='lhp')
        [T,Z] = sp.linalg.schur(Ψ,output='real')
        I2 = np.array([[1,0],[0,1]])
        Px = np.array([[0,1],[1,0]])
        Πl = [I2 if T[2*i,2*i+1] >0 else Px for i in range(3)]
        Π = sp.linalg.block_diag(Πl[0],Πl[1],Πl[2])
        Dinv = np.abs(np.linalg.diagonal(T@Ω))
        Dinvsqrt = np.sqrt(Dinv)
        #Dt =  np.reciprocal(Dinv)
        St = sqrt@Z@Π@np.diag(Dinvsqrt)
        # Correct numerical issues, these were initially caused by a typo,so they shouldn't be present anymore
        #err = -St@Ω@np.transpose(St)@Ω
        #print(err)
        #fix = sp.linalg.fractional_matrix_power(err,-.5)
    S = St#fix@St
    Sinv = -Ω@np.transpose(S)@Ω
    SinvT = np.transpose(Sinv)
    ν = [(Sinv@σ@SinvT)[2*i][2*i] for i in range(3)]
    # D = np.linalg.diagonal(err@np.diag(Dt))
    #print(St@np.diag(Dt)@np.transpose(St)-σ)
    #print(np.round(St@Ω@np.transpose(St),5))
    #print(Dt)
    #print(np.round(S@Ω@np.transpose(S),8))
    return [S,ν]
def _SLD_precomp(σ):
    [S,ν] = will_decomp(σ)
    Sinv = -Ω@np.transpose(S)@Ω
    SinvT = np.transpose(Sinv)
    def M(j,k,m):
        Mval = np.zeros([6,6])
        outputs = {
            0: [[0,1],[-1,0]],
            1: [[0,1],[1,0]],
            2: [[1,0],[0,1]],
            3: [[1,0],[0,-1]]
        }
        Mval[(2*j):(2*j+2),(2*k):(2*k+2)] = outputs[m]
        return Mval
    newM = [[[SinvT@M(j,k,m)@Sinv for m in range(3)] for k in range(3)] for j in range(3)]
    return [newM,ν]
    
def SLD(σ,dσ,precomp = None):
    """Calculates the Symmetric Logarithmic Derivative
    Parameters
    ----------
    sigma: np.cdouble[6,6]
        The covariance matrix
    dsigma: np.cdouble[n,6,6]
        Derivative of the covariance matrix
    precomp: [np.cdouble[6,6][3,3],np.cdouble[3]]
        A precomputation step useful in the case where we want to calculate SLDs for multiple parameters
    Returns
    -------
    np.cdouble[6,6]
        The Symmetric Logarithmic Derivative
    """
    if precomp is None:
        precomp = _SLD_precomp(σ)
    newM = precomp[0]
    ν=precomp[1]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="invalid value encountered in divide")
        warnings.filterwarnings("ignore", message="divide by zero encountered in divide")
        Lsum = [[[ newM[i][j][k] * 1/(ν[i]*ν[j] - (-1)**k)*np.linalg.trace(newM[i][j][k]@dσ) for i in range(3)] for j in range(3)] for k in range(3)]
    L = np.nansum(Lsum,axis=(0,1,2))
    return L
    #print(ν)
    #print(np.round(Sinv@σ@SinvT,8))
def FIM(σ,dσs):
    """Calculates the Quantum Fisher Information Matrix
    Parameters
    ----------
    sigma: np.cdouble[6,6]
        The covariance matrix
    dsigmas: np.cdouble[n,6,6]
        An array of derivatives of the covariance matrix
    Returns
    -------
    np.cdouble[len(dsigmas),len(dsigmas)]
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
