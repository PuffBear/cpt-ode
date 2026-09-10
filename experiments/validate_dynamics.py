"""Deterministic numerical validation of the CPT coordination model.

Run: python paper/experiments/validate_dynamics.py
Requires numpy, scipy, matplotlib. All outputs are relative to this file.
No simulated measurements, random seeds, or empirical MARL claims are used.
"""
from pathlib import Path
import csv
import json
import platform
import numpy as np
import scipy
from scipy.integrate import solve_ivp
from scipy.special import expit, logit
from scipy.optimize import brentq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "results"
FIG = ROOT / "figures"
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "savefig.bbox": "tight"})
BLUE, ORANGE = "#0072B2", "#D55E00"
RTOL, ATOL, HORIZON = 1e-10, 1e-12, 20000.


def payoff(lam, alpha=1.):
    return (6.**alpha - lam * 2.**alpha) / 2.


def field(theta, a):
    p = expit(np.asarray(theta).reshape(-1, 2))
    return (p * (1-p) * ((a+1)*p[:, ::-1]-1)).reshape(-1)


def ode(p0, a, horizon, rtol=RTOL, atol=ATOL, dense=False):
    sol = solve_ivp(lambda t, y: field(y, a), (0., horizon),
                    logit(np.asarray(p0)).reshape(-1), method="DOP853",
                    rtol=rtol, atol=atol, dense_output=dense)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol


def primitive(p, k):
    p = np.asarray(p)
    return (k-2)*(np.log(p)-np.log1p(-p))+(k-1)/(1-p)+1/p


def separatrix(p, a):
    k, star = a+1, 1/(a+1)
    if abs(p-star) < 1e-12:
        return star
    target = primitive(p, k)
    lo, hi = ((star, 1-1e-12) if p < star else (1e-12, star))
    return brentq(lambda q: primitive(q, k)-target, lo, hi, xtol=1e-13)


def predicted(p0, a):
    if a <= 1e-14:
        return np.array(["safe"] * len(p0))
    distances = p0[:, 1] - np.array([separatrix(p, a) for p in p0[:, 0]])
    return np.where(abs(distances) < 1e-9, "separatrix",
                    np.where(distances > 0, "risky", "safe"))


def observed(p):
    return np.where(np.max(p, axis=1) < .01, "safe",
                    np.where(np.min(p, axis=1) > .99, "risky", "unresolved"))


def csv_out(name, rows):
    with (OUT / name).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def save(fig, name):
    fig.savefig(FIG / (name + ".pdf"))
    fig.savefig(FIG / (name + ".png"), dpi=220)
    plt.close(fig)


def main():
    grid = np.array(np.meshgrid(np.arange(1, 20)/20,
                               np.arange(1, 20)/20)).reshape(2, -1).T
    lams = [1., 2., 2.5, 2.9, 3., 3.1, 4.]
    basin_rows, summary, saved = [], [], {}
    for lam in lams:
        a = payoff(lam)
        theory = predicted(grid, a)
        mask = theory != "separatrix"
        sol = ode(grid[mask], a, HORIZON, dense=True)
        end = expit(sol.y[:, -1]).reshape(-1, 2)
        obs = observed(end)
        half = observed(expit(sol.sol(HORIZON/2)).reshape(-1, 2))
        for p, pred, result, ep in zip(grid[mask], theory[mask], obs, end):
            basin_rows.append(dict(lam=lam, alpha=1., p1_initial=p[0],
                p2_initial=p[1], predicted=pred, observed=result,
                p1_final=ep[0], p2_final=ep[1]))
        counts = dict(lam=lam, evaluated=int(mask.sum()),
            separatrix_excluded=int((~mask).sum()),
            safe=int(np.sum(obs=="safe")), risky=int(np.sum(obs=="risky")),
            unresolved=int(np.sum(obs=="unresolved")),
            disagreements=int(np.sum((obs != theory[mask]) & (obs != "unresolved"))),
            horizon_label_changes=int(np.sum(obs != half)))
        summary.append(counts)
        saved[lam] = (grid[mask], obs, grid[~mask])
        print("basins", counts, flush=True)
    csv_out("basins.csv", basin_rows)
    csv_out("basin_summary.csv", summary)

    # Curvature sweeps normalize lambda by the derived threshold.
    small = np.array(np.meshgrid(np.arange(1, 10)/10,
                                np.arange(1, 10)/10)).reshape(2, -1).T
    curve_rows = []
    for alpha in [.6, .8, 1.]:
        for ratio in [.8, .95, 1., 1.05, 1.2]:
            lam = ratio * 3.**alpha
            a = payoff(lam, alpha)
            theory = predicted(small, a)
            mask = theory != "separatrix"
            sol = ode(small[mask], a, HORIZON)
            obs = observed(expit(sol.y[:, -1]).reshape(-1, 2))
            curve_rows.append(dict(alpha=alpha, lam=lam, threshold=3.**alpha,
                ratio=ratio, evaluated=int(mask.sum()), safe=int(np.sum(obs=="safe")),
                risky=int(np.sum(obs=="risky")), unresolved=int(np.sum(obs=="unresolved")),
                disagreements=int(np.sum((obs != theory[mask]) & (obs != "unresolved")))))
    csv_out("curvature_summary.csv", curve_rows)

    # Euler is applied to logits, not directly to probabilities.
    starts = np.array([[.2,.2],[.2,.7],[.45,.65],[.7,.2],
                       [.7,.7],[.9,.95],[.95,.95],[.4,.8]])
    hs = [.4,.2,.1,.05,.025,.0125]
    euler_rows, tolerance_rows = [], []
    for lam in lams:
        a = payoff(lam)
        reference = ode(starts, a, 40., rtol=1e-12, atol=1e-14, dense=True)
        check = ode(starts, a, 40., dense=True)
        ts = np.linspace(0,40,3201)
        tolerance_rows.append(dict(lam=lam, max_probability_difference=float(
            np.max(np.abs(expit(reference.sol(ts))-expit(check.sol(ts)))))))
        for h in hs:
            n = round(40/h)
            theta = logit(starts).reshape(-1)
            trajectory = np.empty((len(theta), n+1))
            trajectory[:,0] = theta
            for step in range(n):
                theta = theta + h*field(theta,a)
                trajectory[:,step+1] = theta
            exact = reference.sol(np.linspace(0,40,n+1))
            err = np.max(np.abs(expit(trajectory)-expit(exact)))
            euler_rows.append(dict(lam=lam, h=h, T=40.,
                initial_conditions=len(starts), max_probability_error=float(err)))
    csv_out("euler_errors.csv", euler_rows)
    csv_out("solver_tolerance_check.csv", tolerance_rows)

    # Independent centered finite differences and conservation checks.
    checks = []
    for lam in [1.,2.,2.5,2.9]:
        a = payoff(lam)
        pstar = 1/(a+1)
        theta = np.repeat(logit(pstar),2)
        eps = 1e-5
        fd = np.column_stack([(field(theta+eps*np.eye(2)[j],a)
                               -field(theta-eps*np.eye(2)[j],a))/(2*eps)
                              for j in range(2)])
        s = pstar*(1-pstar)
        analytic = np.array([[0,(a+1)*s*s],[(a+1)*s*s,0]])
        sol = ode(starts, a, 40., dense=True)
        pp = expit(sol.sol(np.linspace(0,40,1001))).reshape(-1,2,1001)
        inv = primitive(pp[:,0],a+1)-primitive(pp[:,1],a+1)
        checks.append(dict(lam=lam, jacobian_max_abs_error=float(np.max(abs(fd-analytic))),
            invariant_max_scaled_drift=float(np.max(abs(inv-inv[:,[0]])/(1+abs(inv[:,[0]]))))))
    csv_out("analytic_checks.csv", checks)

    # Probability-space vector field, endpoints, and analytic separatrices.
    fig, axs = plt.subplots(2,2,figsize=(7.1,5.7),sharex=True,sharey=True)
    axs = axs.ravel()
    u = np.linspace(.015,.985,31)
    x,y = np.meshgrid(u,u)
    for ax, lam in zip(axs,[1.,2.9,3.,3.1]):
        a=payoff(lam)
        vx=x*x*(1-x)**2*((a+1)*y-1)
        vy=y*y*(1-y)**2*((a+1)*x-1)
        ax.streamplot(u,u,vx,vy,color="#B0B0B0",density=.65,
                      linewidth=.45,arrowsize=.65)
        ps,labels,excluded=saved[lam]
        for label,color,marker in [("safe",BLUE,"o"),("risky",ORANGE,"^")]:
            z=ps[labels==label]
            ax.scatter(z[:,0],z[:,1],s=7,c=color,marker=marker,alpha=.65,
                       label=label.capitalize())
        if len(excluded):
            ax.scatter(excluded[:,0],excluded[:,1],s=12,c="black",marker="x")
        if a>0:
            sx=np.linspace(.01,.99,250)
            sy=np.array([separatrix(p,a) for p in sx])
            ax.plot(sx,sy,"k--",lw=1.3,label="Separatrix")
            ax.scatter([1/(a+1)],[1/(a+1)],c="white",edgecolor="black",s=35,zorder=5)
        ax.set_title(rf"$\lambda={lam:g}$")
        ax.set(xlim=(0,1),ylim=(0,1),xlabel=r"Initial $p_1$")
        ax.set_aspect("equal")
    axs[0].set_ylabel(r"Initial $p_2$")
    axs[2].set_ylabel(r"Initial $p_2$")
    axs[0].legend(fontsize=8,loc="lower left",framealpha=.95)
    fig.tight_layout()
    save(fig,"phase_portraits")

    fig,axs=plt.subplots(1,3,figsize=(7.1,2.65))
    ll=np.linspace(.5,2.999,450); a=3-ll; star=1/(a+1)
    axs[0].plot(ll,star,color=ORANGE,lw=2)
    axs[0].axvline(3,color="black",ls="--",lw=1)
    axs[0].set(xlabel=r"Loss aversion $\lambda$",ylabel=r"Interior saddle $p^*$",
               xlim=(.5,3.2),ylim=(0,1.04),title="Saddle location")
    mu=(a+1)*(star*(1-star))**2
    axs[1].plot(ll,mu,color=ORANGE,label=r"$\mu_+$")
    axs[1].plot(ll,-mu,color=BLUE,ls="--",label=r"$\mu_-$")
    axs[1].axvline(3,color="black",ls=":",lw=1)
    axs[1].axhline(0,color="grey",lw=.5)
    axs[1].set(xlabel=r"Loss aversion $\lambda$",ylabel="Jacobian eigenvalues",
               xlim=(.5,3.2),title="Saddle eigenvalues")
    axs[1].legend(frameon=False,loc="center left",fontsize=8)
    for lam,marker,style in zip([1.,2.9,3.1],["o","s","^"],["-","--",":"]):
        rr=[r for r in euler_rows if r['lam']==lam]
        axs[2].loglog([r['h'] for r in rr],[r['max_probability_error'] for r in rr],
                       marker=marker,ls=style,ms=3,label=rf"$\lambda={lam:g}$")
    axs[2].set(xlabel="Euler step size h",ylabel="Maximum probability error",
               title="Euler tracking")
    axs[2].set_xticks([.0125,.05,.2],["0.0125","0.05","0.2"])
    axs[2].xaxis.set_minor_locator(NullLocator())
    axs[2].tick_params(axis="x",labelsize=8)
    axs[2].legend(fontsize=8,frameon=False)
    fig.tight_layout();save(fig,"threshold_and_euler")

    slopes=[]
    for lam in lams:
        rr=[r for r in euler_rows if r['lam']==lam][-3:]
        slopes.append(dict(lam=lam,slope=float(np.polyfit(np.log([r['h'] for r in rr]),
                                  np.log([r['max_probability_error'] for r in rr]),1)[0])))
    csv_out("euler_orders.csv",slopes)
    report=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
        matplotlib=matplotlib.__version__,solver="DOP853",rtol=RTOL,atol=ATOL,
        basin_horizon=HORIZON,basin_threshold=.01,grid="0.05:0.05:0.95 squared",
        basin_summary=summary,curvature_summary=curve_rows,
        analytic_checks=checks,euler_orders=slopes,
        reference_tolerance_max_difference=max(r['max_probability_difference'] for r in tolerance_rows))
    (OUT/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2),flush=True)
    assert all(r['disagreements']==0 and r['unresolved']==0 for r in summary+curve_rows)
    assert max(r['jacobian_max_abs_error'] for r in checks)<1e-8
    assert max(r['invariant_max_scaled_drift'] for r in checks)<1e-6
    assert all(.95<r['slope']<1.05 for r in slopes)


if __name__ == "__main__":
    main()
