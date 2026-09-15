# Reproducing the empirical extensions

Install `requirements-extensions.txt` in a separate environment and run `python experiments/validate_extensions.py` from the paper directory. Outputs go to `experiments/results/extensions/` and `figures/saddle_central.pdf` and `.png`. These runs are local and do not require a cluster.

## Matched baselines

Five objectives use the same outcome law, independent sigmoid policies, initial probabilities (0.2,0.3), and DOP853 with horizon 2,000, relative tolerance 1e-9, and absolute tolerance 1e-11. CPT conditions use alpha=1 and lambda=2.3; nonlinear weighting has exponent 1/4. Expected return plus own-policy binary entropy uses temperature 2, chosen before execution to make the potential strictly concave. This is a positive control for interior attraction, not a tuned performance competitor. Stationary root eigenvalues are checked independently of trajectory endpoints. Endpoints from one start are not global basin estimates.

## Payoffs and unequal preferences

The full factorial grid contains gains {5.4,6,6.6}, loss magnitudes {1.8,2,2.2}, safe outcomes {.9,1,1.1}, and preference differences delta={0,.02,.05}. Agent loss aversions are (2.3-delta,2.3+delta). All 81 root solves start at the original stable equilibrium. Acceptance requires solver success, logit magnitude below 25, and gradient residual below 1e-9. Jacobians use centered logit differences of 1e-5; stability requires both eigenvalues to have real part below -1e-7. Singular trial iterates can cause warnings and rejected solves. No clipping manufactures roots.

The 27 homogeneous settings agree with the exact payoff-family criterion: nine have a stable interior point and 18 do not. There are 16 verified stable points in 54 unequal-preference settings; 38 searches remain unresolved. Rejected searches do not demonstrate nonexistence. All entries are retained in the CSV.

## Sampled gradients

At lambda=2.3, start at the exact stable logits plus (.1,-.1). Use batch sizes 64,256,1024; 40 seeds numbered 10000--10039 per batch; step .05; and 20,000 updates, corresponding to ODE time 1,000. Seed labels are reused across batch sizes. Multinomial counts over the five action/outcome categories exactly simulate the sufficient statistics of iid batches.

The score estimate uses phi=.4*w'(t)*I(+6)+w'(u)*I(positive), multiplied by action_indicator-p after subtracting its exact expectation as a scalar control variate. Cumulative probabilities t,u are exact and held fixed during the score calculation. Event enumeration checks conditional unbiasedness. This estimator exploits the known equal-probability risky lottery; it is not model-free CPT learning and does not test estimation error in the probability weights.

No gradients or policies are clipped. CSV rows include batch, seed, step, update count, endpoint probabilities, and maximum coordinate error. The .02 distance threshold summarizes the outcomes and was not a stopping rule. Boxplots show all errors, medians, interquartile ranges, 1.5-IQR whiskers, and outliers. They are not confidence intervals. Local finite-horizon concentration does not rule out eventual noise-induced escape or establish a stochastic convergence rate.

Assertions verify the baseline eigenvalue signs, agreement of homogeneous perturbations with the exact threshold, and sampler unbiasedness at three distinct policies. Library versions are recorded in the summary and the extension requirements file.
