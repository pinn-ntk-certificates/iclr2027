# Completed campaign

The campaign reported in the paper completed on **15 November 2026** on a
Slurm cluster (float64, one thread per run, CPython 3.14.10,
torch 2.13.0, numpy 2.5.2). It was prepared and submitted with
`campaign.py prepare` and `campaign.py submit` from commit `6ac2821` and lives
in `results/campaign_cluster_20261115/`. All configured jobs completed; the
reported control analysis includes compatible targets only.

| Included results | Completed | Coverage |
|---|---:|---|
| PDE training | 770 / 770 | 11 PDEs, 7 widths, 10 seeds, 3000 GD steps |
| Compatible-target controls | 1080 / 1080 | 600 manufactured targets + 480 compatible contrast targets |
| Initialization measurements | 7700 / 7700 | 100 independent seeds per PDE and width |
| Limiting-kernel references | 11 / 11 | 8 scrambled QMC replicates per PDE |

All 770 PDE runs reduced loss; 767 reduced it by at least a factor of ten.
Across widths 16 to 16384, pooled median relative kernel drift decreased from
0.611 to 0.000862. Every PDE had lower median drift at the larger width.
These numerical diagnostics do not verify the theorem's sufficient width bound.

## What is tracked and what is released

Tracked in the repository, inside `results/campaign_cluster_20261115/`:

- `broad.json`, `controls.json`: the configurations the campaign ran from;
  `snapshot.json`: hashes of the configurations, sources and environment at
  preparation time; `submission.json`: the Slurm job identifiers;
  `environment.freeze.txt`.
- `results/broad/`: `findings.json` (one row per training run), `findings.md`,
  `findings.tex`, `aggregate.json`, `initialization_aggregate.json`, the
  tables and figures of the campaign report.
- `results/controls/`: the control aggregate and its figures;
  `results/controls_compatible/`: the compatible-only report and the
  1,080-row `runs.csv` whose SHA256 the figure aggregate records;
  `results/evidence_assessment/`: the operator-class, width, per-problem,
  conditioning and rank-tolerance summaries.
- `logs/`: the job scripts and the aggregate and post-processing logs.

Published as assets of the release `campaign-records-20261115` of the repository, outside the
git history: the
per-run records, one JSON record and one NPZ history for each of the 770
training runs (`results/broad/runs/<pde>/`) and 1,080 control runs
(`results/controls/runs/`), the 11 references (`results/broad/references/`)
and the 11 initialization-sweep files (`results/broad/initialization_sweep/`).

| Asset | Contents | Size | SHA256 |
|---|---|---:|---|
| `campaign_cluster_20261115_training_records.tar.gz` | 1,573 files: training runs, references, initialization sweep | 181.5 MB | `57c9622c33591956548be12d9adf8a5a909c5027240ba5650031d874babb1a6b` |
| `campaign_cluster_20261115_control_records.tar.gz` | 2,160 files: control runs | 316.9 MB | `901b9c3b3e87bc9b53fbec52b8aa8d04dea9427c55475c09ac14fefd39d7c2ce` |

`RECORDS.sha256` in the campaign directory lists both asset hashes and the
hash of every file inside them. Extract both archives inside
`results/campaign_cluster_20261115/` (member paths are relative to it) and
verify with `sha256sum -c RECORDS.sha256` from that directory. The records
then make every aggregation command in
[EXTENSIVE_EXPERIMENTS.md](EXTENSIVE_EXPERIMENTS.md) rerunnable, and
`campaign.py worker --kind aggregate` reproduces the tracked aggregates from
them. The archives are deterministic; `scripts/package_campaign_records.py`
of the development repository builds them from the record directory.

Derived artifacts:

- [Figure aggregate](data/compatible_rank_convergence_conditioning.json), built
  from the campaign by `build_figure_aggregate.py` (`--check` verifies it).
- [Generated provenance record](../../figures/compatible_rank_convergence_conditioning_summary.json)
  of the main-text figure.
- [Reproduction instructions](EXTENSIVE_EXPERIMENTS.md).

The compatible-control report excludes all 120 structurally incompatible cases
from tables, figures and conclusions. The remaining contrast cases use independent
functionals, so their targets are attainable in the trial space; they are reported
separately from manufactured-PDE targets. Selection uses compatibility, never
observed loss or convergence.

## Relation to the campaign of 7 September 2026

The submitted manuscript quoted an earlier run of the same configuration
(7 September 2026) whose per-run records were not kept. The 15 November campaign
reproduces every value quoted from it: all tallies and counts, the Table 8
medians to the printed digits except one loss ratio at the float64 floor
(6.36e-30 became 6.03e-30), the Table 9 medians to three digits, and the
initialization slopes -0.512 to -0.469. The 15 November campaign is now the
single source of these numbers.

A second run of the training stage on Windows (CPU build of the same library
versions) agrees with the cluster run to 1.6e-9 relative at widths 256 and above;
at widths 4 to 64 the final losses sit at the float64 floor and 49 of 330 runs
differ there beyond 1e-6, which changes the tenfold count from 767 to 766 (one
width-4 Poisson run) and the checkpoint-monotonicity flag in 4 runs. Initial
kernel quantities agree to 1e-12 on both platforms.
