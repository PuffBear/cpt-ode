"""Reproduce the reduced-lottery power-weighting saddle-node checks."""
from pathlib import Path
import json
import numpy as np
from scipy.optimize import brentq
from scipy.special import expit, logit
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/results'; OUT.mkdir(exist_ok=True)
SN=(5-1/np.sqrt(3))/2

def R(x): return (2-3*x)*np.sqrt(x)/(3*x*x-4*x+2)**.75

def Rp(x): return (x-2)*(3*x-1)/(np.sqrt(x)*(3*x*x-4*x+2)**1.75)

def field(theta,lam):
    p,q=expit(theta); t=p*q/2; u=t+(1-p)*(1-q)
    A=.25*t**(-.75); B=.25*u**(-.75); C=5-2*lam
    return np.array([p*(1-p)*(C*A*q/2+B*(1.5*q-1)),q*(1-q)*(C*A*p/2+B*(1.5*p-1))])

def eigs(x):
    s=x*(1-x); A=.25*(x*x/2)**(-.75); B=.25*(1-2*x+1.5*x*x)**(-.75)
    return np.array([-s*s*x*A*Rp(x)/2,-s*s*B/x])

rows=[]; max_jacerr=0.; max_resid=0.; trajectories=[]
for lam in [SN+1e-4,2.22,2.25,2.3,2.4,2.49,2.5,2.7,2.99]:
    intervals=[(1e-12,1/3),(1/3,1-1e-12)] if lam<2.5 else [(1/3,1-1e-12)]
    for lo,hi in intervals:
        x=brentq(lambda x:R(x)-(5-2*lam),lo,hi,xtol=1e-14)
        ev=eigs(x); theta=np.repeat(logit(x),2); eps=1e-5
        jac=np.column_stack([(field(theta+eps*np.eye(2)[i],lam)-field(theta-eps*np.eye(2)[i],lam))/(2*eps) for i in range(2)])
        jacerr=float(np.max(np.abs(np.linalg.eigvalsh(jac)-np.sort(ev))))
        max_jacerr=max(max_jacerr,jacerr); max_resid=max(max_resid,float(np.max(np.abs(field(theta,lam)))))
        rows.append([lam,x,*ev,jacerr])
        if lam in [2.25,2.3,2.4] and x<1/3:
            for delta in [[.01,.02],[-.01,-.02],[.015,-.01],[-.015,.01]]:
                y0=theta+delta
                sol=solve_ivp(lambda t,y:field(y,lam),(0,20000),y0,method='DOP853',rtol=1e-10,atol=1e-12,dense_output=True)
                assert sol.success
                error=float(np.max(np.abs(expit(sol.y[:,-1])-x)))
                trajectories.append({'lambda':lam,'logit_perturbation':delta,'final_probability_error':error})
np.savetxt(OUT/'nonlinear_equilibria.csv',rows,delimiter=',',header='lambda,x,mu_parallel,mu_perpendicular,finite_difference_eigenvalue_error',comments='')
# Nondegenerate fold coefficients, independently estimated from the scalar field.
xc=1/3; eps=1e-5
scalar=lambda x,lam: x*(1-x)*field(np.repeat(logit(x),2),lam)[0]
K=(xc*(1-xc))**2*xc/2*.25*(xc*xc/2)**(-.75)
gl=(scalar(xc,SN+eps)-scalar(xc,SN-eps))/(2*eps)
gxx=(scalar(xc+eps,SN)-2*scalar(xc,SN)+scalar(xc-eps,SN))/eps**2
summary={'lambda_saddle_node':SN,'max_stationarity_residual':max_resid,'max_eigenvalue_check_error':max_jacerr,'fold_G_lambda_analytic':-2*K,'fold_G_lambda_numerical':gl,'fold_G_xx_analytic':5*np.sqrt(3)*K,'fold_G_xx_numerical':gxx,'stable_trajectory_checks':trajectories,'max_final_probability_error':max(t['final_probability_error'] for t in trajectories)}
(OUT/'nonlinear_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,axes=plt.subplots(1,2,figsize=(7,2.55),layout='constrained')
x=np.linspace(.0001,1/3,700); y=np.linspace(1/3,.9999,1000)
axes[0].plot((5-R(x))/2,x,color='#0072B2',label='Stable mixed policy')
axes[0].plot((5-R(y))/2,y,'--',color='#D55E00',label='Saddle')
axes[0].plot(SN,1/3,'ko',ms=4); axes[0].axvline(SN,color='.6',lw=.7)
axes[0].set(xlabel=r'Loss aversion $\lambda$',ylabel=r'Stationary probability $x$',xlim=(2.18,3.02),ylim=(0,1),title='(a) Exact interior branches')
axes[0].legend(fontsize=8,loc='upper left')
# Trajectories distinguish the two-dimensional attraction from diagonal stability.
lam=2.3; stable=brentq(lambda x:R(x)-(5-2*lam),1e-12,1/3); saddle=brentq(lambda x:R(x)-(5-2*lam),1/3,1-1e-12)
for start in [[.06,.12],[.12,.06],[.3,.2],[.2,.3],[.45,.42]]:
    sol=solve_ivp(lambda t,y:field(y,lam),(0,500),logit(start),method='DOP853',rtol=1e-10,atol=1e-12,dense_output=True)
    ps=expit(sol.sol(np.linspace(0,500,1000))); axes[1].plot(ps[0],ps[1],lw=1.2); axes[1].plot(*start,'o',ms=2)
axes[1].plot(stable,stable,'o',color='#0072B2',ms=6,label='Stable')
axes[1].plot(saddle,saddle,'x',color='#D55E00',ms=6,label='Saddle')
axes[1].set(xlabel=r'$p_1$',ylabel=r'$p_2$',xlim=(0,.7),ylim=(0,.7),title=r'(b) Full flow at $\lambda=2.3$')
axes[1].legend(fontsize=8)
for ext in ['pdf','png']: fig.savefig(ROOT/f'figures/nonlinear_fold.{ext}',dpi=200)
print(json.dumps({k:v for k,v in summary.items() if k!='stable_trajectory_checks'},indent=2))
# Exact algebra checks for the model-specific fold.
import sympy as sp
xs=sp.symbols('x', positive=True)
rs=(2-3*xs)*sp.sqrt(xs)/(3*xs**2-4*xs+2)**sp.Rational(3,4)
expected=(xs-2)*(3*xs-1)/(sp.sqrt(xs)*(3*xs**2-4*xs+2)**sp.Rational(7,4))
assert sp.simplify(sp.diff(rs,xs)-expected)==0
assert sp.simplify(rs.subs(xs,sp.Rational(1,3))-1/sp.sqrt(3))==0
assert sp.simplify(sp.diff(rs,xs,2).subs(xs,sp.Rational(1,3))+5*sp.sqrt(3))==0
assert max_jacerr < 2e-12 and max_resid < 3e-17
assert summary['max_final_probability_error'] < 2e-11
assert abs(gl+2*K)<1e-10 and abs(gxx-5*np.sqrt(3)*K)<1e-6
print('Exact symbolic identities and all numerical checks passed.')
