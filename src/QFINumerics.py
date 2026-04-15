import qutip as qt
import numpy as np
import scipy as sp
from tqdm.autonotebook import trange
import warnings

def QFI(rho,G):
    # Input:
    #  - rho: the state
    #  - G: the generator of a Unitrary transform
    # Output: the QFI of the state with respect to that transform

    # Get the eigenstates, which gives us all the info we need to get the QFI
    (lambdas, Uw) = rho.eigenstates(output_type='oper')
    dims = G.dims
    U = qt.dimensions.from_tensor_rep(np.reshape(qt.dimensions.to_tensor_rep(Uw),np.array(dims).flatten()),dims)
    Gp = U.dag()*G*U
    [lambda1,lambda2] = np.meshgrid(lambdas,lambdas)

    # Vectorized logic to calculate the QFI
    # May be improved in the future, but works very well now
    matsize = [np.prod(dims[0]),np.prod(dims[1])]
    top = (lambda1-lambda2)**2
    prod = np.abs(np.reshape(qt.dimensions.to_tensor_rep(Gp),matsize))**2
    bot = lambda1 + lambda2
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="invalid value encountered in divide")
        warnings.filterwarnings("ignore", message="divide by zero encountered in divide")
        FIs = np.where(bot>0,top*prod/bot,0)

    return 2*np.sum(FIs)

def genChannel(dims=[5,5,5,6], params=[.1,.1],method = 'operator'):
    eta = params[0]
    theta = np.arccos(np.sqrt(eta))
    nth = params[1]
    thdims = dims[0]
    sdims = dims[1]
    idims = dims[2]
    wdims = dims[3]
    # Create our transformation, and our state to compare with
    ath = qt.tensor(qt.destroy(thdims),qt.identity(sdims))
    asi = qt.tensor(qt.identity(thdims),qt.destroy(sdims)) # can't call this as, because it's a keyword
    G = theta *1j*(ath.dag()*asi +asi.dag()*ath)
    # Define the beamsplitter transform
    U = G.expm()
    pth = qt.thermal_dm(thdims,nth,method)
    # Create a superoperator which performs the thermal mixing + tracing out
    S1=0
    for i in range(thdims):
        fockn = qt.fock(thdims,i)
        pthket = pth*fockn
        ql = qt.tensor(qt.identity(sdims),pthket).drop_scalar_dims(inplace=True)
        qr = qt.tensor(qt.identity(sdims),fockn.dag()).drop_scalar_dims(inplace=True)
        S1 = S1+qt.sprepost(ql,qr)
    S2 = qt.sprepost(U,U.dag())*S1
    S3 = qt.tensor_contract(S2,(1,3))
    # Defines a map that turns a 3 mode pure mode to a mixture of two
    # mode pure states with classical witnesses
    mixsup = sum([qt.sprepost(qt.fock_dm(wdims,i),qt.fock_dm(wdims,i)) for i in range(wdims)])
    # Since each of these channels is a single field channel, we can
    # construct our whole channel via tensor product
    chan = qt.super_tensor(S3,qt.to_super(qt.identity(idims)),mixsup)
    return chan


def searchForState(psi0=qt.basis([5,5,6]), max_entropy=.01, epsilon=.1, iters=20000, dims=[5,5,5,6], params=[.1,.1]):
    # Set initial parameters
    eta = params[0]
    nth = params[1]
    sdims = dims[1]
    idims = dims[2]
    wdims = dims[3]
    overall_dims = dims[1:]

    # Create our transformation, and our state to compare with
    #ath = tensor(destroy(thdims),identity(sdims))
    #asi = tensor(identity(thdims),destroy(sdims)) # can't call this as, because it's a keyword
    #G = theta *1j*(ath.dag()*asi +asi.dag()*ath)
    # Define the beamsplitter transform
    #U = G.expm()
    #pth = thermal_dm(thdims,nth)
    # Create a superoperator which performs the thermal mixing + tracing out
    #S1=0
    #for i in range(thdims):
    #    fockn = fock(thdims,i)
    #    pthket = pth*fockn
    #    ql = tensor(identity(sdims),pthket).drop_scalar_dims(inplace=True)
    #    qr = tensor(identity(sdims),fockn.dag()).drop_scalar_dims(inplace=True)
    #    S1 = S1+sprepost(ql,qr)
    chan = genChannel(dims,params)#super_tensor(S3,to_super(identity(idims)),mixsup)
    # Define the thermal state when we mix in vacuum
    pth0 = qt.thermal_dm(sdims,(1-eta)*nth)
    n = qt.tensor(qt.num(idims),qt.identity(sdims),qt.identity(wdims))
    rho0 = qt.vector_to_operator(chan*qt.operator_to_vector(psi0.proj()))

    state = psi0
    max_QFI = QFI(rho0,n)
    # Profiling code
    #pr = cProfile.Profile()
    #pr.enable()
    # rem = max_entropy
    for i in trange(iters):
        newstate = (state + epsilon*qt.rand_ket(overall_dims)).unit()
        rho = qt.vector_to_operator(chan*qt.operator_to_vector(newstate.proj()))
        rel_ent = qt.entropy_relative(pth0,rho.ptrace([0]),tol=1e-50)
        if(rel_ent <= max_entropy):
            FI = QFI(rho,n)
            if (FI >= max_QFI):
                max_QFI = FI
                # rem = max_entropy-rel_ent
                state = newstate
    return state


def calc_for_state(rho,n,As,args=[12,.1,.1]):
    # Input
    #  - rho: the initial state. The first field should be the signal field,
    #         all ancilla fields aren't impacted by our procedure
    #  - n: the number operator for the signal field
    #  - As: the annihilation operator for the signal field
    #  - args: an array of arguments that define our setup
    #    - 0: the number of dimensions to use for the thermal
    #         fields truncated Fock space
    #    - 1: the average photon number in the thermal field
    #    - 2: the transmisivity of our beamsplitter
    # Output
    #  - rel_ent: the quantum relative entropy between our state after mixing
    #             with thermal background, and the field without our input state
    #  - QFI: the quantum Fisher information of our state after mixing with
    #         thermal background
    # Description: Simulates a signal field mixed with a thermal background,
    #              and gives a measure of the effectiveness of a measurement
    #              and the detectability of our measurement to an adversary

    # Here we extract our parameters from our argument
    thdims = args[0]
    nth = args[1]
    eta = args[2]
    theta = np.arccos(np.sqrt(eta))

    # Since the thermal mixing is phase invariant, we don't need to worry about
    # when we pass through the sample. Putting the sample before or after the mixing
    # yields identical results. Additionally since phase transformations are unitary
    # we don't need to worry about applying it at all.
    pth = qt.thermal_dm(thdims,nth)
    pth0 = qt.thermal_dm(rho.dims[0][0],(1-eta)*nth)
    at = qt.destroy(thdims)
    G = theta*(qt.tensor(at.dag(),As)+qt.tensor(at,As.dag()))*1j
    U = G.expm()
    st = qt.tensor(pth, rho)
    stm = U*st*U.dag()

    # Tracing out the thermal mode should just involve doing a partial trace over
    # just the thermal mode. Unfortunately, QuTiP only exposes a function that
    # traces over all **but** the specified modes. Therefore, we just specify
    # all but the first mode
    tokeep = np.arange(1,len(stm.dims[0]))
    rhom = stm.ptrace(tokeep)

    # The adversary only has access to the signal mode when trying to determine
    # if we are making a measurement, so we trace over all other modes
    rel_ent = qt.entropy_relative(pth0,rhom.ptrace([0]),tol=1e-50)

    FI = QFI(rhom,n)
    return [rel_ent,FI]

def FI_observable(rho,G,sigma,thetabounds):
    [_,_,probs0] = qt.measurement.measurement_statistics(rho,sigma)
    thetas = np.reshape(np.linspace(thetabounds[0],thetabounds[1],100),(1,100))
    def prob_non_vec(theta):
        U = (1j*theta[0]*G).expm()
        state = U*rho*U.dag()
        [_,_,probs] = qt.measurement.measurement_statistics(state,sigma)
        return probs
    def probs(thetas):
        return np.apply_along_axis(prob_non_vec, axis=0, arr=thetas)
    def for_jacobian(thetas):
        return np.log(probs(thetas))
    res = sp.differentiate.jacobian(for_jacobian,thetas)
    dlogp = res.df[:,0,:]
    probs = probs(thetas)
    expects = np.sum(dlogp**2*probs,axis=0)
    maxval = np.max(expects)
    thetamax = thetas[0][np.argmax(expects)]
    return [maxval, thetamax]
