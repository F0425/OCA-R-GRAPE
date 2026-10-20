# Publication cleanup verification

The original project files and experimental outputs outside this folder were
not changed. This release removes historical entry points and inactive options
rather than redesigning the paper model.

Removed: two-stage/frozen OCA training, non-paper feature sets, non-random split
schemes, MAR and generic source-block mechanisms, environmental/load/similarity
relation construction, graph feature-weight tuning, unused helper functions,
SMAPE and non-paper imputation metric variants. Environmental-block experiments
remain in the experiment runner and include all 12 environmental variables.

Historical parity scripts are no longer runtime dependencies. Model settings
have one factory and experiments have one runner. Destination names now replace
the ambiguous internal route names. Relation edges use fixed weights and retained
record order as described in the manuscript.

Checks performed locally:

- Python syntax checks for all release modules.
- Unchanged ASTs for the OCA class and protected encoder/decoder/message-graph
  and loss helpers.
- Exact equality of the analytical input columns on the paper's 2,128-row data.
- Exact split/scaler/MCAR condition hashes for seeds 0--4; exact environmental
  block hashes at 0.25/0.50/0.75 for those seeds.
- Exact edge indices, weights and degrees for none/Vessel/Destination/V+D on
  the paper input for all five seeds.
- Two-epoch CPU synthetic-data parity for GRAPE, R-GRAPE, OCA-GRAPE and
  OCA-R-GRAPE, both transductive and strict inductive. Transductive prediction
  differences were zero; inductive results passed absolute tolerance 1e-7.
- One-epoch CPU synthetic-data GAIN parity, with exactly equal completions.
- Runner initialization and leakage audit on the original input.
- The table-only analysis reproduces the manuscript's pooled Spearman values
  (LR 0.798, RF 0.686, XGBoost 0.655, LightGBM 0.704) from existing result tables.

These are targeted checks, not a rerun of the 20,000-epoch GPU experiments.
IGRM was not retrained during cleanup. No original data, training caches,
checkpoints or result tables are included here.

The fixed GAIN comparator weights and constant missing-category downstream
column are retained to preserve the recorded experiments; README documents both.
The authors still need to select a project licence and complete the final paper
citation before publishing the repository.
