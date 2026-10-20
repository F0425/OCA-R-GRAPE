"""Produce the paper's numerical summaries and utility correlations without plots."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def spearman(x: pd.Series, y: pd.Series) -> float:
    """Calculate Spearman rho as Pearson correlation of average ranks."""
    pair = pd.concat([x, y], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(pair) < 2 or pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
        return float("nan")
    return float(pair.iloc[:, 0].rank().corr(pair.iloc[:, 1].rank()))


def save(table: pd.DataFrame, out_dir: Path, filename: str) -> None:
    table.to_csv(out_dir / filename, index=False, encoding="utf-8-sig")


def summarize(imputation: pd.DataFrame, downstream: pd.DataFrame, groups: pd.DataFrame, out_dir: Path) -> None:
    """Keep suites distinct and average rates within each seed before overall SD."""
    keys = ["experiment_suite", "evaluation_protocol", "imputer", "nominal_missing_rate"]
    metrics = ["MAE_scaled_clipped", "RMSE_scaled_clipped"]
    table = imputation.groupby(keys, dropna=False)[metrics].agg(["mean", "std"]).reset_index()
    table.columns = ["_".join(filter(None, c)) if isinstance(c, tuple) else c for c in table.columns]
    save(table, out_dir, "imputation_by_condition.csv")

    seed_keys = ["experiment_suite", "evaluation_protocol", "imputer", "seed"]
    by_seed = imputation.groupby(seed_keys, dropna=False)[metrics].mean().reset_index()
    overall = by_seed.groupby(seed_keys[:-1], dropna=False)[metrics].agg(["mean", "std"]).reset_index()
    overall.columns = ["_".join(filter(None, c)) if isinstance(c, tuple) else c for c in overall.columns]
    save(overall, out_dir, "imputation_overall_across_seeds.csv")

    group_keys = ["experiment_suite", "evaluation_protocol", "imputer", "source_group"]
    group_seed = groups.groupby(group_keys + ["seed"], dropna=False)[metrics].mean().reset_index()
    group_table = group_seed.groupby(group_keys, dropna=False)[metrics].agg(["mean", "std"]).reset_index()
    group_table.columns = ["_".join(filter(None, c)) if isinstance(c, tuple) else c for c in group_table.columns]
    save(group_table, out_dir, "source_group_overall_across_seeds.csv")

    group_rate = groups[groups.experiment_suite.eq("main_mcar")].groupby(
        ["source_group", "imputer", "nominal_missing_rate"], as_index=False
    )["MAE_scaled_clipped"].mean()
    ref = group_rate[group_rate.imputer.eq("GRAPE")].drop(columns="imputer").rename(
        columns={"MAE_scaled_clipped": "GRAPE_MAE"})
    full = group_rate[group_rate.imputer.eq("OCA-R-GRAPE")].merge(
        ref, on=["source_group", "nominal_missing_rate"], validate="one_to_one")
    full["relative_MAE_reduction_percent"] = (
        100 * (full.GRAPE_MAE - full.MAE_scaled_clipped) / full.GRAPE_MAE.replace(0, np.nan))
    save(full, out_dir, "source_group_relative_improvement_by_rate.csv")

    ds_keys = keys + ["predictor"]
    ds_table = downstream.groupby(ds_keys, dropna=False)[["MAE", "RMSE", "R2"]].agg(["mean", "std"]).reset_index()
    ds_table.columns = ["_".join(filter(None, c)) if isinstance(c, tuple) else c for c in ds_table.columns]
    save(ds_table, out_dir, "downstream_by_condition.csv")


def utility_correlations(imputation: pd.DataFrame, downstream: pd.DataFrame, out_dir: Path) -> None:
    """Pair each prediction error with its seed-matched complete-data reference."""
    main = downstream[downstream.experiment_suite.eq("main_mcar")].copy()
    if main.empty:
        return
    refs = downstream[
        downstream.experiment_suite.eq("complete_data_reference")
        & downstream.imputer.eq("Complete-Data Reference")
    ][["seed", "predictor", "RMSE"]].rename(columns={"RMSE": "reference_RMSE"})
    if refs.empty:
        raise ValueError("The Spearman analysis requires the complete-data reference suite.")
    paired = main.merge(refs, on=["seed", "predictor"], how="left", validate="many_to_one")
    if paired.reference_RMSE.isna().any():
        raise ValueError("Missing seed/predictor complete-data reference; cannot calculate matched degradation.")
    paired["delta_RMSE"] = paired.RMSE - paired.reference_RMSE
    imp = imputation[imputation.experiment_suite.eq("main_mcar")][
        ["seed", "imputer", "nominal_missing_rate", "MAE_scaled_clipped"]
    ]
    paired = paired.merge(imp, on=["seed", "imputer", "nominal_missing_rate"],
                          how="left", validate="many_to_one")
    if paired.MAE_scaled_clipped.isna().any():
        raise ValueError("Downstream conditions lack matching imputation MAEs.")
    save(paired, out_dir, "imputation_downstream_matched_values.csv")
    means = paired.groupby(["predictor", "imputer", "nominal_missing_rate"], as_index=False).agg(
        imputation_MAE=("MAE_scaled_clipped", "mean"),
        downstream_delta_RMSE=("delta_RMSE", "mean"),
        n_seeds=("seed", "nunique"),
    )
    save(means, out_dir, "imputation_downstream_condition_means.csv")
    details, summaries = [], []
    for predictor, part in means.groupby("predictor"):
        rhos = []
        for rate, at_rate in part.groupby("nominal_missing_rate"):
            rho = spearman(at_rate.imputation_MAE, at_rate.downstream_delta_RMSE)
            rhos.append(rho)
            details.append({"predictor": predictor, "nominal_missing_rate": rate,
                            "spearman_rho": rho, "n_method_conditions": len(at_rate)})
        finite = np.asarray([rho for rho in rhos if np.isfinite(rho)])
        summaries.append({
            "predictor": predictor,
            "pooled_condition_mean_rho": spearman(part.imputation_MAE, part.downstream_delta_RMSE),
            "pooled_n_method_rate_conditions": len(part),
            "median_within_rate_rho": float(np.median(finite)) if len(finite) else float("nan"),
            "n_missing_rates": len(rhos),
        })
    save(pd.DataFrame(details), out_dir, "utility_correlation_by_rate.csv")
    save(pd.DataFrame(summaries), out_dir, "utility_correlation_summary.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, default=Path("paper_tables"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    imp = pd.read_csv(args.results_dir / "ALL_imputation_summary.csv")
    ds = pd.read_csv(args.results_dir / "ALL_downstream_summary.csv")
    groups = pd.read_csv(args.results_dir / "ALL_source_group_summary.csv")
    summarize(imp, ds, groups, args.out_dir)
    utility_correlations(imp, ds, args.out_dir)
    print(f"Tables written to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
