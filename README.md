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

The nonlinear result uses the analytically tractable distortion \(w(u)=u^{1/4}\). It does not establish a corresponding transition for Prelec or other empirically calibrated inverse-S weighting functions. A systematic Prelec sweep is the intended next experiment.
