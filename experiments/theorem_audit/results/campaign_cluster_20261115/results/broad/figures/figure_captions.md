# Theorem-audit figure captions

**Positivity overview.** Estimated limiting-NTK relative eigengap (left) and the
replicated, scrambled-Sobol lower bound after subtracting both the
between-scramble uncertainty term and the coarse/fine operator-norm difference
(right). Color distinguishes the pointwise DNTK, weak-functional,
signed-measure nonlocal, and any numerical-precheck routes. A circle means the
conservative QMC gate passed; an X means it was inconclusive. For analytically
covered rows QMC estimates the gap size but is not the logical positivity test.
An inconclusive numerical gate is not evidence of singularity, and a numerical
pass is evidence rather than a proof.

**Training dynamics.** Constraint loss for full-batch nonlinear gradient
descent under the paper's exact shallow biased architecture, normalized by its
initial value, at the largest completed width for each PDE. Curves show the
median over seeds; the translucent band is the nonlinear-GD interquartile
range. Dashed and dash-dotted curves propagate the same initial residual with
the frozen empirical kernel and independently estimated limiting kernel. The
dotted curve is the theorem's conservative geometric envelope
`L_0 (1 - eta lambda_LB)^k`, shown only when the uncertainty-adjusted QMC
lower quantity `lambda_LB` is positive. Values below `1e-16` are clipped only for
logarithmic display.

**Finite-width stability.** Medians with seedwise interquartile bars as width
increases. Panel (a) uses `initial_reference_error_operator`; panel (b)
is the empirical fraction satisfying the recorded three-quarter initialization
gap event (Wilson 95% interval; seeds without a generalized Loewner diagnostic
are omitted from that panel, and scalar eigenvalue ratios are stored separately);
panel (c) uses
`max_kernel_drift_operator` and takes the maximum over recorded GD
checkpoints; panel (d) shows the theorem-scaled maximum per-neuron displacement.
Line style distinguishes the pointwise, weak-functional, signed-measure, and
numerical-precheck routes.

Aggregation coverage: 770 completed, 0 positivity-gated or
explicitly skipped, and 0 missing expected training artifacts. See
`runs.csv` and `aggregate.json` for every expected job and exclusion reason.
