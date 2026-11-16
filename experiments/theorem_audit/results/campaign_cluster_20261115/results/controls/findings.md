# Compatible-target rank and conditioning results

Completed **1080/1080 compatible controls**: 600 manufactured-target and 480 compatible contrast-target runs. Every reported target is functionally compatible. No training was rerun for this reporting overlay.

Among 120 exact-duplicate manufactured-target runs, 118 reached loss below 1e-12. The duplicate functionals have structural rank one, and their equal targets permit convergence despite a singular NTK.

The table pools compatible contrast runs across 6 functional families at width 16384. It reports medians after 3,000 full-batch GD steps; family-specific plots should be used for scientific comparisons. Pooling is descriptive and is not a common spectral constant or a statistical confidence interval.

| Row mixing | Runs | Median initial minimum eigenvalue | Median final/initial loss |
|---|---:|---:|---:|
| Independent | 30 | 0.10511 | 1.53263e-31 |
| 1 | 30 | 0.035472 | 3.54223e-30 |
| 0.1 | 30 | 0.00068762 | 0.0100731 |
| 0.01 | 30 | 7.12189e-06 | 0.40002 |

Smaller positive eigenvalues permit much slower convergence on these compatible targets. Frozen-kernel rates are comparisons, and neither an initial positive gap nor finite numerical training verifies the theorem's existential width and step-size constants.

Compatible contrast targets use independent rows (the unmodified pair or positive mixing parameter). They need not match the original manufactured PDE solution. These two-row controls measure constraint optimization, not held-out PDE solution accuracy.

[Singular compatible targets](figures/singular_target_compatibility.png) · [Conditioning and GD](figures/conditioning_training_dynamics.png) · [Gap and decay rate](figures/conditioning_gap_and_rate.png) · [Width diagnostics](figures/rank_control_width_diagnostics.png)

[Detailed figure captions](figures/figure_captions.md). The selection and original artifact hashes are recorded in `aggregate.json` and `selection_provenance.json`; `runs.csv` contains only retained compatible runs.
