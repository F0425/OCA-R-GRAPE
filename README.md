# OCA-R-GRAPE

Code accompanying **Graph-Based Joint Imputation of Multi-Source Maritime
Voyage-Segment Data by Integrating Cross-Voyage Relations and Target-Specific
Context**.

## Scope

The publication model combines the original GRAPE bipartite backbone, fixed
Vessel + Destination connections, and a target-specific Observed Context Adapter
(OCA). The encoder, decoder and adapter are trained jointly. Vessel and
Destination edges are merged using the maximum weight on overlapping pairs.
Relation neighbours follow retained input-record order, not time or numerical
similarity. Vessel uses k=2 and weight=0.6; Destination uses k=5 and weight=0.4.

The experiment runner includes:

- MCAR comparison: Mean, KNN, MICE, supervised GAIN, GRAPE, IGRM-common and
  OCA-R-GRAPE, with R-GRAPE and OCA-GRAPE as component ablations.
- Independent module-ablation and relation-ablation runs.
- Partition-isolated strict inductive evaluation.
- Joint loss of the 12 environmental features at affected-voyage probabilities
  0.25, 0.50 and 0.75. These are not full-matrix cell-missing rates.
- Eight source-group evaluations.
- Fuel prediction with LR, RF, XGBoost and LightGBM, plus a complete-data reference.

Exploratory models, frozen/two-stage OCA, MAR, generic source-block masking,
alternative feature sets, alternative split schemes, numerical-similarity
relations, explanations and plotting tools are outside this release.

## Files

- `oca_rgrape.py`: model/preprocessing/evaluation library. There is no
  separate model-script command-line entry point.
- `run_experiments.py`: the paper experiment entry point.
- `igrm_common_protocol.py`: IGRM-common subprocess adapter.
- `analyze_paper_results.py`: table summaries and downstream Spearman statistics.
- `GRAPE-master/models/` and `GRAPE-master/utils/`: required upstream code.

## Data

The original company data are confidential and are not distributed. Supply a
locally available, authorised voyage-level CSV using `--excel_path`. See
`DATA_FORMAT.md` for its interface. This repository implements the experimental
procedure; the original paper's numerical results cannot be reproduced without
access to its underlying data. No synthetic dataset is included in this release.

The paper uses 2,128 complete records and 27 numerical features. Row order is part
of the relation protocol and must be retained. A random split first reserves 30%
for testing, then reserves 20% of the remaining 70% for validation (approximately
56/14/30, not 70/20/10). Seeds are 0--4. Train-fitted min-max scaling is performed
before masking. MCAR masks are nested across rates within a seed. The artificial
test targets never enter message passing or checkpoint selection.

Under transductive evaluation, unmasked observations from all partitions are
available to graph encoding; losses use training targets only. Strict inductive
evaluation uses separate train, validation and test graphs and constructs the
test graph after checkpoint selection. It is not a vessel-disjoint evaluation.

## Environment

The paper reports Python 3.12.3 and PyTorch 2.11.0. Install PyTorch for your
platform, then install `requirements.txt`. Install `torch-scatter` separately
using a wheel compatible with the exact PyTorch, Python and CUDA build. A wheel
compiled for another PyTorch version must not be reused.

The package versions in `requirements.txt` are dependency names, not a lockfile
or a claim that every combination was tested. The runtime manifest records
several installed versions for each run.

```powershell
python -m pip install -r requirements.txt
python -c "import torch, torch_geometric, torch_scatter; print(torch.__version__, torch_geometric.__version__)"
```

IGRM is fetched automatically from `G-AILab/IGRM` at commit
`5cfc955daa5d0f4bbdcbad1552cfd7493dfd5fd0`, into `IGRM_pinned/` beside the runner.
For offline use, provide that checkout using `--igrm_repo_dir` and
`--no_auto_download_igrm`. `--skip_igrm` is for an incomplete installation and does
not reproduce the paper's full benchmark comparison.

## Run

Use one-line commands from this directory. Replace the input path with your own.

```powershell
python run_experiments.py --excel_path "C:\path\to\authorised_data.csv" --device cuda --seeds "0,1,2,3,4" --graph_max_epochs 20000 --out_dir "./paper_results" --run_all --resume --verbose
```

Individual suites: `--run_main_mcar`, `--run_module_ablation`,
`--run_relation_ablation`, `--run_inductive`, `--run_structured`,
`--run_downstream`, and `--run_complete_reference`. `--run_downstream` runs the
main MCAR comparison with downstream prediction enabled.

```powershell
python analyze_paper_results.py --results_dir "./paper_results" --out_dir "./paper_tables"
```

Resume skips completed method/condition caches; there is no epoch-level resume.
Use a new output directory after changing code, data or training settings.
Failures are recorded in `failed_runs.csv`. The release retains leakage checks
but no longer requires the historical development scripts for parity audits.

## Comparator and preprocessing notes

GRAPE and its extensions use unit feature weights and the sum of separately
averaged artificial-cell and observed-cell MSE. The GAIN comparator preserves its
experimental supervised reconstruction and fixed relative feature weights:
distance=2.5, cargo volume=1.5, TEU=1.4, heavy containers=1.3, other features=1.0,
normalised to mean one. GAIN is capped at 1,000 epochs. These are adaptations,
not claims of an unchanged official GAIN implementation.

IGRM-common uses fixed external masks, common train-fitted scaling, validation
MAE checkpoint selection, friend reconstruction every 100 epochs and a sparse
global friend-edge budget with k=10. It is not the unmodified official protocol.

The historical preprocessing appends an all-missing `end_is_eu` field when absent.
Its one-hot missing-category column is constant in the paper dataset and carries
no voyage-specific information. This constant is retained to preserve the exact
downstream input shape and stochastic tree-feature sampling. The two informative
categorical inputs are vessel identity and destination. If a new dataset contains
varying `end_is_eu`, its downstream experiment is not the same paper protocol.

Metrics clip imputed numerical values to [0,1], leaving references unchanged.
Source-group overall summaries first average rates within each seed, then report
the mean and sample SD across seeds. Spearman analysis pairs the imputation MAE
with the RMSE increase from each seed-matched complete-data reference, averages
seeds within method/rate, and computes pooled and within-rate correlations.

## Third-party code

See `THIRD_PARTY_NOTICES.md`. The retained upstream GRAPE files are distributed
with their original MIT licence. This release does not assign a new licence to
the authors' project code; a project licence and final paper citation should be
selected by the authors before publication.
