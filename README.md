# CPT Gradient Dynamics

Reproducible deterministic experiments for the two-agent stochastic coordination model in *Prospect-Theoretic Gradient Dynamics: Stability and Bifurcations in Multi-Agent Learning*.

The repository contains exact-gradient ODE integrations and numerical cross-checks. It does not contain sampled policy gradients, learned critics, or deep-MARL experiments.

## Reproduce

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r experiments/requirements.txt
MPLCONFIGDIR=/tmp/cpt-matplotlib XDG_CACHE_HOME=/tmp/cpt-cache .venv/bin/python experiments/validate_dynamics.py
MPLCONFIGDIR=/tmp/cpt-matplotlib XDG_CACHE_HOME=/tmp/cpt-cache .venv/bin/python experiments/validate_nonlinear.py
```

The scripts write figures to `figures/` and tabular outputs to `experiments/results/`. The committed result tables record the verification run; rerunning the scripts overwrites those files deterministically up to floating-point and library-version variation.

`validate_dynamics.py` verifies the linear-probability-weighting analysis: basin predictions, the conserved-quantity separatrix, curvature sweeps, finite-difference Jacobians, and Euler tracking.

`validate_nonlinear.py` verifies the power-weighting saddle-node: stationary residuals, analytic eigenvalues, fold coefficients, symbolic identities, and local convergence of nonsymmetric perturbations.

See [experiments/README.md](experiments/README.md) for parameter settings and interpretation of every result file.

## Scope

The exact nonlinear theorem uses the analytically tractable distortion \(w(u)=u^{1/4}\). The completed high-precision experiment extends the numerical evidence to inverse-S Prelec weights; it does not replace an analytic Prelec classification or establish behavior under sampled-gradient learning.

## Prelec sweep

`experiments/prelec_sweep.py` maps the existence and location of stable mixed policies for inverse-S Prelec weights over \((\lambda,\gamma)\). It uses arbitrary-precision log-probability coordinates because the stable branch and fold can approach probabilities far below IEEE double precision as \(\gamma\) approaches one.

The full PBS configuration is in `jobs/prelec_full.pbs`. It requests eight CPU cores on one node and limits numerical libraries to one thread per worker. A small local smoke test is:

```sh
.venv/bin/python experiments/prelec_sweep.py \
  --workers 2 --gamma-count 5 --lambda-count 7 \
  --dps 50 --bisect-steps 80 --scan-points 101 \
  --output-dir /tmp/prelec-smoke/results \
  --figure-dir /tmp/prelec-smoke/figures
```

The recorded SHARC run completed 45,069 phase cells at 80-digit precision. It found stable/saddle pairs in 42,926 cells and at some positive loss aversion for every sampled \(\gamma\in[0.05,0.95]\). See `experiments/results/prelec_full/summary.json` and `figures/prelec_phase_map.*`.
