# Checkable NTK positivity and finite-width gradient descent for linear PINNs: code

Anonymous code release accompanying the ICLR 2027 submission. It contains the
theorem-audit implementation, every configuration used in the paper, the
tracked aggregates and provenance records behind the reported numbers, the
figure scripts, and the locked Python environment. Nothing here is required to
compile the paper; the repository exists so that the numerical claims can be
regenerated and checked.

## Environment

Python 3.11 or newer with [uv](https://docs.astral.sh/uv/). Every command below
runs from the repository root inside the locked environment:

```bash
uv sync
```

`uv.lock` pins the exact versions (`requirements.txt` lists the same direct
dependencies for readers who prefer `pip`). All computation is float64 on CPU
and deterministic.

## Run the tests

```bash
uv run python -m unittest discover -s experiments/theorem_audit -p 'test_*.py' -t .
```

The suite uses PyTorch autograd as an implementation-independent oracle for the
hand-derived features, Jacobians, and gradient-descent step, and covers the
certificate procedures, the plotting validators, and the per-iterate and
nonlinear-variance experiments.

## Smoke test (minutes), then the full local audit

```bash
uv run python experiments/theorem_audit/run.py all       --config experiments/theorem_audit/configs/smoke.json
uv run python experiments/theorem_audit/run.py aggregate --config experiments/theorem_audit/configs/smoke.json
```

The `smoke` configuration exercises the same code path as every larger run:
for each of the eleven PDE designs it builds the registry-selected
positive-definiteness certificate, refuses to train a design whose certificate
fails, computes a replicated quasi-Monte Carlo reference for the limiting
kernel, trains the two-layer tanh network by full-batch gradient descent, and
writes self-describing JSON/NPZ artifacts under
`experiments/theorem_audit/results/smoke/`. The `local` configuration is the
same audit at larger reference sample counts; its certificate manifests are
tracked in `experiments/theorem_audit/results/local/references/`.

## Regenerate the figures in the paper

Main-text figure (kernel stabilization with width; conditioning versus
finite-horizon loss), from the tracked control-study aggregate:

```bash
uv run python experiments/theorem_audit/plot_supportive_evidence.py
uv run python -m unittest experiments.theorem_audit.test_plot_supportive_evidence
```

Appendix figure (active-subspace loss decomposition for the three-row audit),
recomputed from scratch and then plotted:

```bash
uv run python -m experiments.theorem_audit.joint_nullspace_controls
uv run python -m experiments.theorem_audit.plot_joint_nullspace_evidence
```

Each plotting script validates its input table (row counts and SHA256) before
drawing and writes `figures/<name>.pdf`, `figures/<name>.png`, and
`figures/<name>_summary.json` with every plotted value.

## Per-iterate monotonicity and nonlinear kernel variance

```bash
uv run python -m experiments.theorem_audit.per_iterate_monotonicity  --config experiments/theorem_audit/configs/per_iterate_monotonicity_smoke.json
uv run python -m experiments.theorem_audit.per_iterate_monotonicity  --config experiments/theorem_audit/configs/per_iterate_monotonicity.json
uv run python -m experiments.theorem_audit.nonlinear_kernel_variance --config experiments/theorem_audit/configs/nonlinear_kernel_variance_smoke.json
uv run python -m experiments.theorem_audit.nonlinear_kernel_variance --config experiments/theorem_audit/configs/nonlinear_kernel_variance.json
```

The first records the loss at every gradient step on a two-row design and
compares the worst per-step ratio with the frozen-kernel factor; the second
is a supplementary check that the paper does not report: it measures how the
variance of a tangent Gram entry scales with width for a linear row and for a
product of two linear rows. Their summaries are tracked under
`experiments/theorem_audit/results/per_iterate_monotonicity/` and
`experiments/theorem_audit/results/nonlinear_kernel_variance/`.

## Worked example of the width condition

```bash
uv run python -m experiments.theorem_audit.example_gd_certificate --config experiments/theorem_audit/configs/example_gd_certificate_smoke.json
uv run python -m experiments.theorem_audit.example_gd_certificate --config experiments/theorem_audit/configs/example_gd_certificate.json
```

This recomputes every constant of the worked single-row example of the width
condition: the gap bound, `R_K`, `B_*^2`, `L_*^2`, the residual bound, the
three width terms, the integer width 1356, and the step bound 10/21. Rational
constants are exact and transcendental ones are enclosed by outward-rounded
interval arithmetic (`mpmath`, pinned in `uv.lock`). The summary is tracked
under `experiments/theorem_audit/results/example_gd_certificate/`.

## Width required by the sufficient condition for the eleven designs

The finite-width gradient-descent theorem is a sufficient condition. Its width
condition is the maximum of three terms: a concentration term of order
`1/lambda_0`, a kernel-drift term of order `1/lambda_0^3`, and a localization
term of order `1/lambda_0^2`. The table lists them for each of the eleven PDE
designs at confidence 0.95, with `lambda_0` equal to the smallest eigenvalue
`lambda_min` of the limiting kernel, which is the most favorable admissible
choice. `N` is the number of rows of the design.

| Design | N | lambda_min | Concentration term | Drift term | Localization term | Required width |
|---|---|---|---|---|---|---|
| Poisson 1D | 7 | 2.46e-04 | 2.16e+06 | 1.25e+18 | 7.58e+12 | 1.25e+18 |
| Variable elliptic 2D | 13 | 5.20e-03 | 1.39e+06 | 9.34e+16 | 1.28e+12 | 9.34e+16 |
| Heat 1D | 17 | 1.80e-06 | 2.78e+08 | 1.78e+24 | 1.74e+17 | 1.78e+24 |
| Transport 1D | 15 | 6.25e-07 | 1.04e+09 | 5.93e+25 | 2.24e+18 | 5.93e+25 |
| Wave 1D | 22 | 1.57e-07 | 7.69e+09 | 1.39e+28 | 4.12e+19 | 1.39e+28 |
| Biharmonic 2D | 21 | 3.36e-06 | 1.17e+11 | 4.45e+31 | 2.63e+21 | 4.45e+31 |
| Stokes 2D | 32 | 1.38e-04 | 6.88e+06 | 1.63e+19 | 8.02e+13 | 1.63e+19 |
| Weak Poisson 1D | 6 | 4.47e-05 | 3.99e+06 | 2.10e+18 | 3.58e+13 | 2.10e+18 |
| Nonlocal diffusion 1D | 5 | 9.91e-06 | 1.59e+07 | 1.15e+20 | 5.94e+14 | 1.15e+20 |
| Integro-differential 1D | 6 | 2.99e-05 | 1.52e+07 | 4.80e+19 | 2.54e+14 | 4.80e+19 |
| Caputo diffusion 1D | 17 | 1.63e-06 | 2.69e+08 | 1.64e+24 | 1.97e+17 | 1.64e+24 |

The largest trained width is 16,384, so every run of the paper is far below
the width the condition requires, and the experiments do not test its
constants. The limiting kernel is evaluated by Gauss-Legendre quadrature in
float64. The constants of the theorem are maxima over sampled points of the
sets on which the theorem takes suprema, so each listed width is a lower bound
of the width the condition requires.

The table is the content of the tracked summary under
`experiments/theorem_audit/results/width_requirement/`. Recompute it with the
smoke configuration first and the full one second:

```bash
uv run python -m experiments.theorem_audit.width_requirement --config experiments/theorem_audit/configs/width_requirement_smoke.json
uv run python -m experiments.theorem_audit.width_requirement --config experiments/theorem_audit/configs/width_requirement.json
```

## Weak-direction share of the two-row controls

```bash
uv run python -m experiments.theorem_audit.weak_direction_share --config experiments/theorem_audit/configs/weak_direction_share_smoke.json
uv run python -m experiments.theorem_audit.weak_direction_share --config experiments/theorem_audit/configs/weak_direction_share.json
```

For each control family at `epsilon = 0.01` and width 16,384 this evaluates the
initial kernel and residual of the contrast control at the five campaign seeds,
the share of the residual in the weak eigendirection, and the frozen-kernel
prediction of the final-to-initial loss ratio, next to the observed median of
the tracked control aggregate. It explains the different final losses of
families with similar condition numbers. The summary is tracked under
`experiments/theorem_audit/results/weak_direction_share/`.

## Depth variation

```bash
uv run python -m experiments.theorem_audit.depth_variation --config experiments/theorem_audit/configs/depth_variation_smoke.json
uv run python -m experiments.theorem_audit.depth_variation --config experiments/theorem_audit/configs/depth_variation.json
```

This trains networks with `H = 1, 2, 3` hidden layers of uniform width (16 to
1,024) on the Poisson 1D and heat 1D designs with the initialization, loss
scaling, step-size rule and 3,000 steps of the main campaign, five seeds per
cell. Per run it records the smallest eigenvalue of the initial Gram matrix in
total and per layer block, the largest relative kernel drift over the saved
checkpoints, the smallest ratio of the Gram gap to its initial value, the
discrepancy from the frozen-kernel recursion, and the loss ratio. No theorem of
the paper covers `H >= 2`; these are observations at the configured settings.
The summary is tracked under `experiments/theorem_audit/results/depth_variation/`.

## The broad campaign

The 770-run PDE study, the 1,080-run control study, and the 7,700-kernel
initialization sweep were run as Slurm job arrays with
`experiments/theorem_audit/campaign.py`; the sharded runner is
`run.py train --task-index K`, and the array scripts are in
`experiments/theorem_audit/slurm/`. See
`experiments/theorem_audit/EXTENSIVE_EXPERIMENTS.md` for the submission and
aggregation commands and `experiments/theorem_audit/CAMPAIGN_STATUS.md` for
coverage. The per-run histories of that campaign are not included; the
tracked aggregate `experiments/theorem_audit/data/compatible_rank_convergence_conditioning.json`
holds every count and median shown in the main-text figure together with the
SHA256 of the 1,080-row source table, and the three-row audit is fully
regenerable with the commands above.

## Layout

| Path | Contents |
|---|---|
| `experiments/theorem_audit/*.py` | runner, engine, features, constraints, problems, aggregation, plotting, tests |
| `experiments/theorem_audit/configs/` | every configuration used in the paper (single source of numerical settings) |
| `experiments/theorem_audit/data/` | tracked aggregates behind the figures |
| `experiments/theorem_audit/results/` | tracked certificate manifests and experiment summaries |
| `experiments/theorem_audit/slurm/` | job-array scripts for the broad campaign |
| `figures/` | figures in the paper with their `_summary.json` provenance |

The three-row audit script also writes a LaTeX table
(`experiments/theorem_audit/joint_nullspace_controls.tex`) for the paper's
appendix; it is a generated output and is ignored here.
