# Third-party sources

## GRAPE

This project imports the original GRAPE model implementation from the included
`GRAPE-master/models` and `GRAPE-master/utils` directories. These upstream source
files have not been rewritten as part of the publication cleanup.

The original MIT licence and copyright are retained at `GRAPE-master/LICENSE`.
`GRAPE-master/README.md` is the upstream document, not this project's experiment
instructions. Cite the GRAPE paper by You et al. (2020), *Handling Missing Data
with Graph Representation Learning*, as identified in that upstream README.

The exact upstream GRAPE commit has not been established in this cleanup; do
not infer one from the directory name. The included source defines the backbone
version used by this release.

## IGRM

Repository: https://github.com/G-AILab/IGRM

Pinned commit: `5cfc955daa5d0f4bbdcbad1552cfd7493dfd5fd0`.

The runner downloads that repository separately; it is not bundled here.
Retain and comply with its upstream licence when obtaining or redistributing it.
The project adapter is labelled `IGRM_common_protocol` and discloses its common
mask/scaling/validation protocol and sparse friend-network adaptation.

## Other dependencies

NumPy, pandas, PyTorch, scikit-learn, PyTorch Geometric, torch-scatter, XGBoost and
LightGBM remain subject to their respective licences. They are installed as
dependencies, not copied into this release.
