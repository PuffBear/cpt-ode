"""Matched baselines, local model perturbations, and oracle-weight Monte Carlo PG.

The sampler uses exact cumulative probabilities in w'; it is not model-free CPT RL.
Perturbed root failures are reported as unresolved, never as nonexistence.
"""
from pathlib import Path
from itertools import product
import csv
import json
import platform
import scipy
import numpy as np
from scipy.special import expit, logit
from scipy.optimize import root, brentq
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'experiments/results/extensions'
OUT.mkdir(parents=True, exist_ok=True)
SN = (5-1/np.sqrt(3))/2

def R(x):
    return (2-3*x)*np.sqrt(x)/(3*x*x-4*x+2)**.75

def field(theta, lam=2.3, G=6., L=2., S=1., delta=0., mode='terminal', tau=2.):
    p = expit(theta)
    q = p[::-1]
    s = p*(1-p)
    ls = np.array([lam-delta, lam+delta])
    if mode in ('eu', 'entropy'):
        f = s*((G/2-L/2+S)*q-S)
        return f-tau*s*theta if mode == 'entropy' else f
    if mode == 'value':
        return s*((G/2-ls*L/2+S)*q-S)
    if mode == 'conditional':
        return s*(((G-ls*L)*.5**.25+S)*q-S)
    t = p[0]*p[1]/2
    u = t+(1-p[0])*(1-p[1])
    return s*((G-S-ls*L)*.25*t**(-.75)*q/2+S*.25*u**(-.75)*(1.5*q-1))

def jac(theta, **kw):
    e = np.eye(2)*1e-5
    return np.column_stack([(field(theta+d, **kw)-field(theta-d, **kw))/2e-5 for d in e])

def save(name, rows):
    with (OUT/name).open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(rows)

def main():
    stable = brentq(lambda x: R(x)-.4, 1e-10, 1/3)
    saddle = brentq(lambda x: R(x)-.4, 1/3, 1-1e-10)
    theta0 = logit([stable, stable])
    baselines = []
    paths = {}
    for mode in ['eu', 'value', 'conditional', 'entropy', 'terminal']:
        start = logit([.2, .3])
        sol = solve_ivp(lambda t,y: field(y, mode=mode), [0,2000], start,
                        method='DOP853', rtol=1e-9, atol=1e-11, dense_output=True)
        assert sol.success
        endpoint = expit(sol.y[:,-1])
        guesses={'eu':1/3,'value':1/1.7,'conditional':1/(1+1.4*.5**.25),'entropy':.5,'terminal':stable}
        rr = root(lambda th: field(th,mode=mode), np.repeat(logit(guesses[mode]),2))
        good = (rr.success and np.max(np.abs(rr.x))<25 and np.max(np.abs(field(rr.x,mode=mode)))<1e-9)
        ev = np.linalg.eigvals(jac(rr.x,mode=mode)) if good else np.array([np.nan])
        baselines.append(dict(mode=mode, p1_final=endpoint[0], p2_final=endpoint[1],
                              finite_root_found=bool(good), root_p1=float(expit(rr.x[0])) if good else None,
                              max_real_eigenvalue=float(np.max(ev.real)) if good else None))
        paths[mode] = expit(sol.sol(np.linspace(0,2000,1000)))
    save('baselines.csv',baselines)
    assert all(r['finite_root_found'] for r in baselines)
    assert all(r['max_real_eigenvalue']>0 for r in baselines[:3])
    assert all(r['max_real_eigenvalue']<0 for r in baselines[3:])
    # A finite grid checks persistence under changed payoffs and broken preference symmetry.
    perturbations=[]
    for G,L,S,delta in product([5.4,6.,6.6],[1.8,2.,2.2],[.9,1.,1.1],[0.,.02,.05]):
        kw=dict(G=G,L=L,S=S,delta=delta)
        rr=root(lambda th:field(th,**kw),theta0)
        residual=float(np.max(np.abs(field(rr.x,**kw))))
        good=bool(rr.success and np.max(np.abs(rr.x))<25 and residual<1e-9)
        ev=np.linalg.eigvals(jac(rr.x,**kw)) if good else np.array([np.nan])
        perturbations.append(dict(**kw,root_found=good,residual=residual,
            p1=float(expit(rr.x[0])) if good else None,p2=float(expit(rr.x[1])) if good else None,
            max_real_eigenvalue=float(np.max(ev.real)) if good else None,
            stable=bool(good and np.max(ev.real)<-1e-7)))
    save('perturbations.csv',perturbations)
    homogeneous=[r for r in perturbations if r['delta']==0]
    for r in homogeneous:
        lower=(r['G']-r['S']-r['S']/np.sqrt(3))/r['L']
        upper=(r['G']-r['S'])/r['L']
        assert r['stable']==bool(lower<2.3<upper)
    # Enumerate outcomes to check the score estimator's conditional expectation exactly.
    for p in [np.array([.2,.3]),np.array([stable,stable]),np.array([.6,.7])]:
        t=p.prod()/2; u=t+np.prod(1-p); C=.4
        expect=np.zeros(2); b=C*.25*t**.25+.25*u**.25
        for a1,a2,coin in product([0,1],repeat=3):
            a=np.array([a1,a2]); prob=np.prod(np.where(a,p,1-p))*.5
            gain6=bool(a1 and a2 and coin); positive=gain6 or not(a1 or a2)
            phi=C*.25*t**(-.75)*gain6+.25*u**(-.75)*positive
            expect+=prob*(phi-b)*(a-p)
        assert np.max(np.abs(expect-field(logit(p))))<1e-12
    sample_rows=[]
    for batch in [64,256,1024]:
        rngs=[np.random.default_rng(10000+seed) for seed in range(40)]
        th=np.tile(theta0+[.1,-.1],(40,1)); h=.05; steps=20000
        for k in range(steps):
            p=expit(th); t=p[:,0]*p[:,1]/2; u=t+np.prod(1-p,axis=1)
            # Multinomial sufficient statistics exactly simulate iid joint actions/outcomes.
            probs=np.stack([np.prod(1-p,axis=1),(1-p[:,0])*p[:,1],p[:,0]*(1-p[:,1]),t,t],axis=1)
            counts=np.array([rng.multinomial(batch,pr) for rng,pr in zip(rngs,probs)])/batch
            acts=np.array([[0,0],[0,1],[1,0],[1,1],[1,1]])
            phi=np.zeros((40,5)); phi[:,0]=.25*u**(-.75)
            phi[:,4]=.4*.25*t**(-.75)+.25*u**(-.75)
            b=.4*.25*t**.25+.25*u**.25
            grad=np.sum((counts*(phi-b[:,None]))[:,:,None]*(acts[None,:,:]-p[:,None,:]),axis=1)
            th+=h*grad
        for seed,endpoint in enumerate(expit(th)):
            sample_rows.append(dict(batch=batch,seed=10000+seed,step=h,steps=steps,
                p1=endpoint[0],p2=endpoint[1],max_error=float(np.max(np.abs(endpoint-stable)))))
    save('sampled_gradients.csv',sample_rows)
    summary=dict(stable_x=stable,baselines=baselines,perturbation_cells=len(perturbations),
        stable_perturbation_cells=sum(r['stable'] for r in perturbations),
        rejected_local_root_searches=sum(not r['root_found'] for r in perturbations),
        homogeneous_stable=sum(r['stable'] for r in homogeneous),
        homogeneous_no_stable_by_exact_criterion=sum(not r['stable'] for r in homogeneous),
        heterogeneous_stable=sum(r['stable'] for r in perturbations if r['delta']>0),
        heterogeneous_unresolved=sum(not r['root_found'] for r in perturbations if r['delta']>0),
        sampled={str(b):dict(median_error=float(np.median([r['max_error'] for r in sample_rows if r['batch']==b])),
            max_error=max(r['max_error'] for r in sample_rows if r['batch']==b),
            within_002=sum(r['max_error']<.02 for r in sample_rows if r['batch']==b)) for b in [64,256,1024]},
        estimator='Unbiased score estimator with exact cumulative probabilities and exact scalar baseline; not model-free.',
        versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,matplotlib=matplotlib.__version__))
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,ax=plt.subplots(1,3,figsize=(7,2.45),layout='constrained')
    x=np.linspace(.0001,1/3,500); y=np.linspace(1/3,.9999,600)
    ax[0].plot((5-R(x))/2,x,color='#0072B2',label='Stable')
    ax[0].plot((5-R(y))/2,y,'--',color='#D55E00',label='Saddle')
    ax[0].plot(SN,1/3,'ko',ms=4); ax[0].axvline(SN,c='.6',lw=.6)
    ax[0].set(xlabel=r'Loss aversion $\lambda$',ylabel='Stationary probability',title='(a) Exact saddle-node',xlim=(2.15,3.02),ylim=(0,1))
    ax[0].legend(fontsize=7)
    colors=['#777777','#CC79A7','#D55E00','#009E73','#0072B2']
    labels=['EU','Value only','Conditional CPT','EU + entropy','Terminal CPT']
    for (mode,path),col,label in zip(paths.items(),colors,labels):
        ax[1].plot(*path,color=col,label=label)
        ax[1].plot(*path[:,-1],'o',color=col,ms=3)
    ax[1].plot(.2,.3,'ks',ms=3); ax[1].plot(stable,stable,'o',color='#0072B2',ms=5)
    ax[1].set(xlabel=r'$p_1$',ylabel=r'$p_2$',title=r'(b) Matched start, $\lambda=2.3$',xlim=(0,.8),ylim=(0,.8))
    ax[1].legend(fontsize=6,loc='upper left')
    data=[[r['max_error'] for r in sample_rows if r['batch']==b] for b in [64,256,1024]]
    ax[2].boxplot(data,tick_labels=['64','256','1024'],showfliers=True)
    ax[2].set(xlabel='Samples per update',ylabel=r'Final $\|p-p^*\|_\infty$',title='(c) Sampled gradients')
    for ext in ['pdf','png']:fig.savefig(ROOT/f'figures/saddle_central.{ext}',dpi=200)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    main()
