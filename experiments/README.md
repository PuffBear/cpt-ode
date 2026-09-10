# Experiment notes

Run both scripts from the repository root after installing `requirements.txt`.

`validate_dynamics.py` uses the exact logit ODE under linear probability weighting. The loss-aversion sweep uses \(\lambda\in\{1,2,2.5,2.9,3,3.1,4\}\), a 19-by-19 probability grid, and horizon \(T=20{,}000\). The curvature sweep uses \(\alpha\in\{0.6,0.8,1\}\) and five ratios \(\lambda/3^\alpha\). It also checks Euler approximation at six step sizes and finite-difference Jacobians.

`validate_nonlinear.py` uses \(w(u)=u^{1/4}\) and \(\alpha=1\). It brackets all relevant stationary branches at nine loss-aversion values, validates analytic logit eigenvalues against finite differences, verifies the saddle-node coefficients, and integrates nonsymmetric perturbations around stable mixed policies.

The `results/` directory contains the CSV and JSON outputs used in the manuscript. They are deterministic numerical checks of the ODE model; they are not estimates of external MARL performance.

`prelec_sweep.py` performs the inverse-S follow-up. It computes the positive-branch fold and both stationary roots in arbitrary-precision log-probability coordinates. The full configuration covers 181 Prelec exponents and 249 loss-aversion values, yielding 45,069 phase cells. Each exponent also receives an independent 1,001-point derivative-sign scan to test the assumed single-maximum structure. The experiment is numerical evidence for the Prelec family, not a replacement for a proof.
