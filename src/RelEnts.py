import cblind as cb  #noqa F401
import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np
from matplotlib.transforms import Bbox, TransformedBbox
from mpl_toolkits.axes_grid1.inset_locator import (
    BboxConnector,
    BboxConnectorPatch,
    BboxPatch,
)
from tqdm.auto import tqdm


def connect_bbox(bbox1, bbox2,
                 loc1a, loc2a, loc1b, loc2b,
                 prop_lines, prop_patches=None):
    if prop_patches is None:
        prop_patches = {
            **prop_lines,
            "alpha": prop_lines.get("alpha", 1) * 0.2,
            "clip_on": False,
            "ec": "black",
        }

    c1 = BboxConnector(
        bbox1, bbox2, loc1=loc1a, loc2=loc2a, clip_on=False, **prop_lines)
    c2 = BboxConnector(
        bbox1, bbox2, loc1=loc1b, loc2=loc2b, clip_on=False, **prop_lines)

    bbox_patch1 = BboxPatch(bbox1, **prop_patches)
    bbox_patch2 = BboxPatch(bbox2, **prop_patches)

    p = BboxConnectorPatch(bbox1, bbox2,
                           loc1a=loc1a, loc2a=loc2a, loc1b=loc1b, loc2b=loc2b,
                           clip_on=False,
                           **prop_patches)

    return c1, c2, bbox_patch1, bbox_patch2, p
def zoom_effect(ax1, ax2, xmin,ymin,xmax,ymax,**kwargs):
    """
    ax1 : the main Axes
    ax2 : the zoomed Axes

    """


    bbox = Bbox.from_extents(xmin, ymin, xmax, ymax)
    mybbox1 = TransformedBbox(bbox, ax1.transData)
    mybbox2 = ax2.bbox

    prop_patches = {**kwargs, "ec": "black", "alpha": 0.4, "fc": 'blanchedalmond'}

    c1, c2, bbox_patch1, bbox_patch2, p = connect_bbox(
        mybbox1, mybbox2,
        loc1a=1, loc2a=1, loc1b=3, loc2b=3,
        prop_lines=kwargs, prop_patches=prop_patches)

    ax1.add_patch(bbox_patch1)
    ax2.add_patch(c1)
    ax2.add_patch(c2)

    return c1, c2, bbox_patch1, bbox_patch2, p

@np.vectorize
def therm_cov(eta,nb,ns):
    eta = mp.mpf(eta)
    nb = mp.mpf(nb)
    ns = mp.mpf(ns)
    t1 = mp.log(1+nb) - mp.log(1+nb+eta*ns)
    t2 = (nb+eta*ns) *(mp.log(1+nb) + mp.log(nb+eta*ns) -mp.log(nb) - mp.log(1+nb+eta*ns))
    return t1 +t2

@np.vectorize(cache=True)
def Qrel(eta1,eta2,nb,ns):
# This freaks out with very small values...
# Could be a situation where I should redo this with mpmath to avoid such extreme values
# but it could also be worth reworking this to use logs instead of arctanh, which may simplify things considerably
# mpmath doesn't seem to fix the issue, which is somewhat surprising.
    eta1 = mp.mpf(eta1)
    eta2 = mp.mpf(eta2)
    nb = mp.mpf(nb)
    ns = mp.mpf(ns)
    def nu0(eta):
        to_sq = (1-eta)*ns +1+nb
        to_sqrt = to_sq**2 + 4*eta*nb*ns
        return mp.sqrt(to_sqrt)
    def nu1(eta):
        diff = nb - (1-eta) *ns
        return diff + nu0(eta)
    def nu2(eta):
        diff = (1-eta) *ns - nb
        return diff + nu0(eta)
    def lamsq(eta):
        to_sqrt = eta*ns*(1+ns)
        num = 1+nb+ns+eta*ns + 2*mp.sqrt(to_sqrt)
        return num/nu0(eta)

    # Define values that will be used
    nu11 = nu1(eta1)
    nu12 = nu1(eta2)
    nu21 = nu2(eta1)
    nu22 = nu2(eta2)


    logterm = mp.log(1+nb-eta2) - mp.log(1+nb-eta1)
    constterm = 2*(nu11 *mp.acoth(nu11) + nu21 * mp.acoth(nu21))
    sumterm = (nu11 + nu21) *(mp.acoth(nu12) + mp.acoth(nu22))/2
    lamterm = lamsq(eta1)/lamsq(eta2) + lamsq(eta2)/lamsq(eta1)
    diffterm = (nu11 - nu21) *(mp.acoth(nu12) - mp.acoth(nu22))
    nonlogterm = sumterm*lamterm + diffterm-constterm
    return (logterm+nonlogterm)/2


def rel_ent_pnr(eta1,eta2,nb,ns):
    def P0I(_eta,nb,ns):
        return (1+ns)*(1+nb)
    def P0II(eta,nb,ns):
        return 1+ eta*ns+nb
    def P0SI(_eta,_nb,ns):
        return 1+ns
    def P0(eta,nb,ns):
        return 1/P0I(eta,nb,ns)
    def PI(eta,nb,ns):
        return 1/P0II(eta,nb,ns)- P0(eta,nb,ns)
    def PS(eta,nb,ns):
        return 1/P0SI(eta,nb,ns)- P0(eta,nb,ns)
    def PC(eta,nb,ns):
        return 1-PS(eta,nb,ns)-PI(eta,nb,ns)-P0(eta,nb,ns)
    PIval = PI(eta1,nb,ns) * (np.log(PI(eta1,nb,ns)) - np.log(PI(eta2,nb,ns)))
    PCval = PC(eta1,nb,ns) * (np.log(PC(eta1,nb,ns)) - np.log(PC(eta2,nb,ns)))
    return PIval +PCval


@np.vectorize
def rel_ent_su11(eta1,eta2,nb,ns):
    eta1 = mp.mpf(eta1)
    eta2 = mp.mpf(eta2)
    nb = mp.mpf(nb)
    ns = mp.mpf(ns)
    def P0I(eta):
        return nb*(1+2*ns) + (1+(1+mp.sqrt(eta))*ns)**2
    def P0II(eta):
        return 1+ nb*(1+ns)+ ns*(1+ns)*(1+mp.sqrt(eta))**2
    def P0SI(eta):
        return 1+ns*(2+nb+ns+2*(1+ns)*mp.sqrt(eta)+ns*eta)
    def P0(eta):
        return 1/P0I(eta)
    def PI(eta):
        return 1/P0II(eta)- P0(eta)
    def PS(eta):
        return 1/P0SI(eta)- P0(eta)
    def PC(eta):
        return 1-PS(eta)-PI(eta)-P0(eta)

    P0val = P0(eta1) * (mp.log(P0(eta1)) - mp.log(P0(eta2)))
    PSval = PS(eta1) * (mp.log(PS(eta1)) - mp.log(PS(eta2)))
    PIval = PI(eta1) * (mp.log(PI(eta1)) - mp.log(PI(eta2)))
    PCval = PC(eta1) * (mp.log(PC(eta1)) - mp.log(PC(eta2)))
    return PIval +PCval+PSval+P0val


def rel_ent_coh_mp_single(eta1,eta2,nb,ns):
    A = 1/(1+nb)
    B = A/2
    C = 1/ns
    D = C/2
    def P0(eta,r):
        return mp.exp(-r**2*eta*B)*A
    def rel_ent(r):
        P01 = P0(eta1,r)
        P02 = P0(eta2,r)
        PC1 = 1-P01
        PC2 = 1-P02
        # PCval is a bit of a lost cause, but P0val can certainly be calculated more efficiently
        # P0val = -r**2*P01 *(eta1-eta2)*B
        # Could potentially work, but it doesn't buy us much computationally
        # The real work seems to be the numeric integration which this doesn't let us eliminate
        # In order to do that we would need to calculate the second term explicitly as well
        # as the first (the first has a form that looks very analytically calculatable)
        P0val = P01*(mp.log(P01) - mp.log(P02))
        PCval = PC1*(mp.log(PC1)-mp.log(PC2))
        res = P0val +PCval
        #res[np.isnan(res)] = 0
        return res
    def to_int(r):
        #return mp.exp(-r**2/(2*ns)) *r/ns *rel_ent(r)
        return mp.exp(-r**2*D) *r*C *rel_ent(r)
    res = mp.quad(to_int,[0,mp.inf])
    return res
def rel_ent_coh_mp(eta1,eta2,nb,ns):
    shape = np.shape(eta1)
    to_ret = np.zeros(shape)
    for index in tqdm(np.ndindex(shape),total=np.prod(shape)):
        eta1v = eta1[index]
        eta2v = eta2[index]
        to_ret[index] = rel_ent_coh_mp_single(eta1v,eta2v,nb,ns)
    return to_ret


def Qrel_coh(eta1,eta2,nb,ns):
    return ns*(eta1-eta2)**2 *(np.log(1+nb) - np.log(nb))


def rel_ent_homo(eta1, eta2, nb, ns):
    return 2*(eta1 - eta2)**2*ns *np.sqrt(np.pi/(1+nb))


@np.vectorize
def pcoh_rel_ent(eta1, eta2, nb, ns):
    eta1 = mp.mpf(eta1)
    eta2 = mp.mpf(eta2)
    nb = mp.mpf(nb)
    ns = mp.mpf(ns)
    return (mp.sqrt(eta1)-mp.sqrt(eta2))**2/eta1 *therm_cov(eta1,nb,ns)


def comp_plot(etamin, etamax, N, nb, ns, f1, f2, title, zlabel):
    etavals = np.linspace(etamin,etamax,N)
    eta1,eta2 = np.meshgrid(etavals,etavals)
    vals1 = f1(eta1, eta2, nb, ns)
    vals2 = f2(eta1, eta2, nb, ns)
    y = np.array([[vals2[i,j]/vals1[i,j] if i!=j else np.nan for i in range(N)]for j in range(N)],dtype=float)
    y[0,0] = np.mean([y[0,1],y[1,0]])
    y[N-1,N-1] = np.mean([y[N-1,N-2],y[N-2,N-1]])
    for i in range(1,N-1):
        y[i,i] = np.mean([y[i,i-1],y[i,i+1],y[i-1,i],y[i+1,i]])
    
    
    to_plot = y
    fig = plt.figure()
    ax = fig.subplots()
    CS = ax.pcolormesh(eta1,eta2,to_plot, cmap = "cb.iris")
    ax.contour(eta1,eta2,to_plot, levels=[1], linestyles = 'dashed')
    
    
    ax.set_xlabel(r'$\eta_1$')
    ax.set_ylabel(r'$\eta_2$')
    cbar = fig.colorbar(CS,ax = ax)
    cbar.ax.set_ylabel(zlabel)
    
    ax.set_title(title)
    
    vals = (f'$\\bar n_b=${nb:.0e}\n$n_s = ${ns:.0e}')
    bbox = {"boxstyle": 'round', "fc": 'blanchedalmond', "ec": 'orange', "alpha": 0.2}
    ax.text(0.95, 0.07, vals, fontsize=9, bbox=bbox,
                transform=ax.transAxes, horizontalalignment='right')
    return fig

def comp_plot_zoom(etamin1, etamax1, etamin2,etamax2, N, nb, ns, f1, f2, title, zlabel):
    etavals1 = np.linspace(etamin1,etamax1,N)
    eta11,eta21 = np.meshgrid(etavals1,etavals1)
    vals11 = f1(eta11, eta21, nb, ns)
    vals21 = f2(eta11, eta21, nb, ns)
    y1 = np.array([[vals21[i,j]/vals11[i,j] if i!=j else np.nan for i in range(N)]for j in range(N)],dtype=float)
    y1[0,0] = np.mean([y1[0,1],y1[1,0]])
    y1[N-1,N-1] = np.mean([y1[N-1,N-2],y1[N-2,N-1]])
    for i in range(1,N-1):
        y1[i,i] = np.mean([y1[i,i-1],y1[i,i+1],y1[i-1,i],y1[i+1,i]])


    etavals2 = np.linspace(etamin2,etamax2,N)
    eta12,eta22 = np.meshgrid(etavals2,etavals2)
    vals12 = f1(eta12, eta22, nb, ns)
    vals22 = f2(eta12, eta22, nb, ns)
    y2 = np.array([[vals22[i,j]/vals12[i,j] if i!=j else np.nan for i in range(N)]for j in range(N)],dtype=float)
    y2[0,0] = np.mean([y2[0,1],y2[1,0]])
    y2[N-1,N-1] = np.mean([y2[N-1,N-2],y2[N-2,N-1]])
    for i in range(1,N-1):
        y2[i,i] = np.mean([y2[i,i-1],y2[i,i+1],y2[i-1,i],y2[i+1,i]])
    
    
    to_plot1 = y1
    fig = plt.figure()
    axs = fig.subplots(2)
    CS1 = axs[0].pcolormesh(eta11,eta21,to_plot1, cmap = "cb.iris")
    axs[0].contour(eta11,eta21,to_plot1, levels=[1], linestyles = 'dashed')
    
    
    axs[0].set_ylabel(r'$\eta_2$')
    cbar = fig.colorbar(CS1,ax = axs[0])
    cbar.ax.set_ylabel(zlabel)

    to_plot2 = y2
    CS2 = axs[1].pcolormesh(eta12,eta22,to_plot2, cmap = "cb.iris")
    axs[1].contour(eta12,eta22,to_plot2, levels=[1], linestyles = 'dashed')
    
    
    axs[1].set_xlabel(r'$\eta_1$')
    axs[1].set_ylabel(r'$\eta_2$')
    cbar = fig.colorbar(CS2,ax=axs[1])
    cbar.ax.set_ylabel(zlabel)
    zoom_effect(axs[0], axs[1], etamin2,etamin2,etamax2,etamax2)
    
    axs[0].set_title(title)
    
    vals = (f'$\\bar n_b=${nb:.0e}\n$n_s = ${ns:.0e}')
    bbox = {"boxstyle": 'round', "fc": 'blanchedalmond', "ec": 'orange', "alpha": 0.2}
    axs[0].text(0.95, 0.07, vals, fontsize=9, bbox=bbox,
                transform=axs[0].transAxes, horizontalalignment='right')
    axs[1].text(0.95, 0.07, vals, fontsize=9, bbox=bbox,
            transform=axs[1].transAxes, horizontalalignment='right')
    return fig

def comp_plot_filter(etamin, etamax, N, nb, ns, f1, f2, title, zlabel, k):
    etavals = np.linspace(etamin,etamax,N)
    eta1,eta2 = np.meshgrid(etavals,etavals)
    vals1 = f1(eta1, eta2, nb, ns)
    vals2 = f2(eta1, eta2, nb, ns)
    y = np.array([[vals2[i,j]/vals1[i,j] if i!=j else np.nan for i in range(N)]for j in range(N)],dtype=float)
    y2 = np.zeros([N-2,N-2])
    for i in range(1,N-1):
        ik = min(k,N-i,i)
        il = i-ik
        iu = i+ik+1
        for j in range(1,N-1):
            jk = min(k,N-j,j)
            jl = j-jk
            ju = j+jk+1
            subrange = y[il:iu,jl:ju]
            y2[i-1,j-1] = np.nanmedian(subrange)
    
    to_plot = y2
    fig = plt.figure()
    ax = fig.subplots()
    CS = ax.pcolormesh(eta1[1:N-1,1:N-1],eta2[1:N-1,1:N-1],to_plot, cmap = "cb.iris")
    ax.contour(eta1[1:N-1,1:N-1],eta2[1:N-1,1:N-1],to_plot, levels=[1], linestyles = 'dashed')
    
    
    ax.set_xlabel(r'$\eta_1$')
    ax.set_ylabel(r'$\eta_2$')
    cbar = fig.colorbar(CS,ax = ax)
    cbar.ax.set_ylabel(zlabel)
    
    ax.set_title(title)
    
    vals = (f'$\\bar n_b=${nb:.0e}\n$n_s = ${ns:.0e}')
    bbox = {"boxstyle": 'round', "fc": 'blanchedalmond', "ec": 'orange', "alpha": 0.2}
    ax.text(0.95, 0.07, vals, fontsize=9, bbox=bbox,
                transform=ax.transAxes, horizontalalignment='right')
    return fig
