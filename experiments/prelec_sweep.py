"""High-precision phase map for inverse-S Prelec probability weighting.

The experiment studies the reduced terminal-lottery objective in the paper at
alpha = 1.  For every gamma in (0, 1), it locates the maximum of R_w on the
positive branch and solves R_w(x) = 5 - 2 lambda on both sides of that maximum.
The stable solution can be too close to zero for double precision, so all root
calculations are performed in log-probability coordinates with mpmath.
"""

import argparse
import csv
import json
import math
import multiprocessing as mp_pool
import os
import platform
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np


LOG10 = math.log(10.0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gamma-min", type=float, default=0.05)
    parser.add_argument("--gamma-max", type=float, default=0.95)
    parser.add_argument("--gamma-count", type=int, default=181)
    parser.add_argument("--lambda-min", type=float, default=0.01)
    parser.add_argument("--lambda-max", type=float, default=2.49)
    parser.add_argument("--lambda-count", type=int, default=249)
    parser.add_argument("--dps", type=int, default=80)
    parser.add_argument("--bisect-steps", type=int, default=140)
    parser.add_argument("--scan-points", type=int, default=1001)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("experiments/results/prelec_full"))
    parser.add_argument("--figure-dir", type=Path, default=Path("figures"))
    return parser.parse_args()


def log_weight_derivative(log_z, loss_log_z, gamma):
    """Return log w'_gamma(z) using log(z) and -log(z)."""
    return (mp.log(gamma) - loss_log_z ** gamma
            + (gamma - 1) * mp.log(loss_log_z) - log_z)


def prelec_log_r(log_x, gamma):
    """Return log R_w(exp(log_x)) on 0 < x < 2/3."""
    x = mp.exp(log_x)
    log_t = 2 * log_x - mp.log(2)
    loss_log_t = -log_t
    log_u = mp.log1p(-2 * x + mp.mpf("1.5") * x * x)
    loss_log_u = -log_u
    log_wp_u = log_weight_derivative(log_u, loss_log_u, gamma)
    log_wp_t = log_weight_derivative(log_t, loss_log_t, gamma)
    return mp.log(2 - 3 * x) - log_x + log_wp_u - log_wp_t


def log_r_derivative(log_x, gamma):
    """Derivative of log R_w(exp(y)) with respect to y."""
    x = mp.exp(log_x)
    log_t = 2 * log_x - mp.log(2)
    loss_log_t = -log_t
    log_u = mp.log1p(-2 * x + mp.mpf("1.5") * x * x)
    loss_log_u = -log_u
    u = mp.exp(log_u)

    def elasticity(loss_log):
        return (gamma * loss_log ** (gamma - 1)
                - (gamma - 1) / loss_log - 1)

    factor_term = -3 * x / (2 - 3 * x) - 1
    u_term = elasticity(loss_log_u) * x * (3 * x - 2) / u
    t_term = 2 * elasticity(loss_log_t)
    return factor_term + u_term - t_term


def bisect(function, left, right, steps):
    f_left = function(left)
    f_right = function(right)
    if f_left == 0:
        return left
    if f_right == 0:
        return right
    if mp.sign(f_left) == mp.sign(f_right):
        raise ValueError("root is not bracketed")
    for _ in range(steps):
        middle = (left + right) / 2
        f_middle = function(middle)
        if f_middle == 0:
            return middle
        if mp.sign(f_middle) == mp.sign(f_left):
            left, f_left = middle, f_middle
        else:
            right, f_right = middle, f_middle
    return (left + right) / 2


def critical_point(gamma, steps):
    """Locate the unique positive-branch maximum in log-x coordinates."""
    gamma = mp.mpf(str(gamma))
    asymptotic_scale = mp.power(2, gamma / (1 - gamma))
    left = -max(mp.mpf(50), 4 * asymptotic_scale)
    right = mp.log(mp.mpf(2) / 3) - mp.mpf("1e-30")
    while log_r_derivative(left, gamma) <= 0:
        left *= 2
    root = bisect(lambda y: log_r_derivative(y, gamma), left, right, steps)
    return gamma, left, right, root, prelec_log_r(root, gamma)


def solve_gamma(task):
    gamma_float, lambdas, dps, steps, scan_points = task
    mp.mp.dps = dps
    gamma, left, right, log_x_fold, log_r_max = critical_point(
        gamma_float, steps)

    # Numerically test the assumed single-extremum structure on a broad grid.
    scan = [left + (right - left) * i / (scan_points - 1)
            for i in range(scan_points)]
    signs = [mp.sign(log_r_derivative(y, gamma)) for y in scan]
    sign_changes = sum(a != b for a, b in zip(signs[:-1], signs[1:])
                       if a != 0 and b != 0)
    if sign_changes != 1:
        raise RuntimeError("unexpected number of positive-branch extrema for "
                           "gamma={}: {}".format(gamma_float, sign_changes))

    log_five = mp.log(5)
    if log_r_max < mp.mpf(700):
        r_max_float = float(mp.exp(log_r_max))
        lambda_fold = (5.0 - r_max_float) / 2.0
    else:
        r_max_float = float("inf")
        lambda_fold = float("-inf")

    critical = {
        "gamma": gamma_float,
        "log10_x_fold": float(log_x_fold / mp.log(10)),
        "log10_r_max": float(log_r_max / mp.log(10)),
        "r_max": r_max_float,
        "lambda_fold": lambda_fold,
        "positive_lambda_fold": bool(log_r_max < log_five),
        "derivative_sign_changes": sign_changes,
    }

    rows = []
    for lam in lambdas:
        c_value = 5.0 - 2.0 * lam
        log_c = mp.log(mp.mpf(str(c_value)))
        exists = log_c < log_r_max
        row = {
            "gamma": gamma_float,
            "lambda": lam,
            "stable_mixed": int(exists),
            "saddle": int(exists),
            "log10_x_stable": "",
            "log10_x_saddle": "",
            "x_stable_double": "",
            "x_saddle_double": "",
        }
        if exists:
            stable_left = left
            while prelec_log_r(stable_left, gamma) >= log_c:
                stable_left *= 2
            stable = bisect(
                lambda y: prelec_log_r(y, gamma) - log_c,
                stable_left, log_x_fold, steps)
            saddle = bisect(
                lambda y: prelec_log_r(y, gamma) - log_c,
                log_x_fold, right, steps)
            row["log10_x_stable"] = float(stable / mp.log(10))
            row["log10_x_saddle"] = float(saddle / mp.log(10))
            row["x_stable_double"] = (float(mp.exp(stable))
                                       if stable > -700 else 0.0)
            row["x_saddle_double"] = (float(mp.exp(saddle))
                                       if saddle > -700 else 0.0)
        rows.append(row)
    return critical, rows


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def locate_gamma_crossings(critical_rows):
    crossings = []
    for left, right in zip(critical_rows[:-1], critical_rows[1:]):
        f_left = left["log10_r_max"] - math.log10(5.0)
        f_right = right["log10_r_max"] - math.log10(5.0)
        if f_left * f_right < 0:
            crossings.append((left["gamma"], right["gamma"]))
    return crossings


def make_figure(critical_rows, phase_rows, path):
    gamma = np.array([row["gamma"] for row in critical_rows])
    log_x = np.array([row["log10_x_fold"] for row in critical_rows])
    lambda_fold = np.array([row["lambda_fold"] for row in critical_rows])
    lambda_fold = np.maximum(lambda_fold, 0.0)

    gamma_values = sorted(set(row["gamma"] for row in phase_rows))
    lambda_values = sorted(set(row["lambda"] for row in phase_rows))
    phase = np.array([row["stable_mixed"] for row in phase_rows], dtype=float)
    phase = phase.reshape(len(gamma_values), len(lambda_values))

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 2.7),
                             constrained_layout=True)
    axes[0].plot(gamma, lambda_fold, color="#0072B2")
    axes[0].fill_between(gamma, lambda_fold, 2.5, color="#0072B2", alpha=0.18)
    axes[0].set(xlabel=r"Prelec exponent $\gamma$",
                ylabel=r"Positive-$\lambda$ onset",
                ylim=(0, 2.5), title="(a) Saddle-node onset")

    axes[1].plot(gamma, -log_x, color="#D55E00")
    axes[1].set_yscale("log")
    axes[1].set(xlabel=r"Prelec exponent $\gamma$",
                ylabel=r"$-\log_{10}x_{\rm fold}$",
                title="(b) Fold location")

    axes[2].imshow(phase.T, origin="lower", aspect="auto", interpolation="nearest",
                   extent=[gamma_values[0], gamma_values[-1],
                           lambda_values[0], lambda_values[-1]],
                   cmap="Blues", vmin=0, vmax=1)
    axes[2].set(xlabel=r"Prelec exponent $\gamma$",
                ylabel=r"Loss aversion $\lambda$",
                title="(c) Stable mixed policy")
    fig.savefig(str(path.with_suffix(".pdf")))
    fig.savefig(str(path.with_suffix(".png")), dpi=240)
    plt.close(fig)


def main():
    args = parse_args()
    if not (0 < args.gamma_min < args.gamma_max < 1):
        raise ValueError("gamma range must lie strictly inside (0, 1)")
    if not (0 < args.lambda_min < args.lambda_max < 2.5):
        raise ValueError("lambda range must lie strictly inside (0, 2.5)")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    gammas = np.linspace(args.gamma_min, args.gamma_max, args.gamma_count)
    lambdas = np.linspace(args.lambda_min, args.lambda_max, args.lambda_count)
    tasks = [(float(g), [float(v) for v in lambdas], args.dps,
              args.bisect_steps, args.scan_points) for g in gammas]

    started = time.time()
    with mp_pool.Pool(processes=args.workers) as pool:
        results = pool.map(solve_gamma, tasks)
    elapsed = time.time() - started

    critical_rows = [item[0] for item in results]
    phase_rows = [row for item in results for row in item[1]]
    write_csv(args.output_dir / "critical_curve.csv", critical_rows)
    write_csv(args.output_dir / "phase_map.csv", phase_rows)
    make_figure(critical_rows, phase_rows, args.figure_dir / "prelec_phase_map")

    crossings = locate_gamma_crossings(critical_rows)
    summary = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "mpmath": mp.__version__,
        "gamma_range": [args.gamma_min, args.gamma_max, args.gamma_count],
        "lambda_range": [args.lambda_min, args.lambda_max, args.lambda_count],
        "decimal_digits": args.dps,
        "bisection_steps": args.bisect_steps,
        "extremum_scan_points": args.scan_points,
        "workers": args.workers,
        "elapsed_seconds": elapsed,
        "positive_lambda_fold_brackets": crossings,
        "all_extremum_scans_have_one_sign_change": all(
            row["derivative_sign_changes"] == 1 for row in critical_rows),
        "phase_cells": len(phase_rows),
        "stable_cells": sum(row["stable_mixed"] for row in phase_rows),
        "pbs_job_id": os.environ.get("PBS_JOBID"),
        "pbs_queue": os.environ.get("PBS_QUEUE"),
    }
    with (args.output_dir / "summary.json").open("w") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
