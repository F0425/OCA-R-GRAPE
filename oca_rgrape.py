"""OCA-R-GRAPE models, preprocessing and evaluation for the paper experiments."""
from __future__ import annotations

import argparse
import logging
import math
import os
import random
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

GRAPE_CODE_DIR = Path(__file__).resolve().parent / "GRAPE-master"
if GRAPE_CODE_DIR.exists() and str(GRAPE_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(GRAPE_CODE_DIR))

try:
    from torch_geometric.data import Data as PyGData
    from models.gnn_model import get_gnn as get_original_grape_gnn
    from models.prediction_model import MLPNet as OriginalGrapeMLPNet
except Exception:
    PyGData = None
    get_original_grape_gnn = None
    OriginalGrapeMLPNet = None

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

TARGET_OPTIONS = ["fuel_total", "use_hfo", "use_dogo", "use_lfo"]

VOYAGE_CATEGORICAL_SUPPORT = ["vessel_code", "end_port_code", "end_is_eu"]
VOYAGE_LEVEL_ENRICHED_NUMERIC_SUPPORT = [
    "cargo_volume", "total_teu", "heavy_ctns", "total_distance",
    "voyage_duration_h", "ais_distance_nm", "ais_sog_sailing_time_mean_kn",
    "ais_sailing_time_ratio", "ais_draft_time_mean_m",
    "ship_age_years", "ship_deadweight_t", "ship_teu_capacity",
    "ship_design_draught_m", "ship_service_speed_kn", "main_power_kw",
    "wind_speed_time_mean_ms", "headwind_mean_ms", "crosswind_abs_mean_ms",
    "headwind_time_ratio", "crosswind_time_ratio",
    "wave_height_time_mean_m", "wave_height_p90_m",
    "current_speed_time_mean_ms", "headcurrent_mean_ms", "crosscurrent_abs_mean_ms",
    "headcurrent_time_ratio", "crosscurrent_time_ratio",
]
ALL_VOYAGE_NUMERIC_SUPPORT = list(VOYAGE_LEVEL_ENRICHED_NUMERIC_SUPPORT)
RELATION_MODE_OPTIONS = ["none", "vessel", "destination", "vessel_destination"]
FINAL_RELATION_MODE = "vessel_destination"

SOURCE_GROUP_FEATURES = {
    "MRV / Transport": ["cargo_volume", "total_teu", "heavy_ctns", "total_distance"],
    "AIS": ["voyage_duration_h", "ais_distance_nm", "ais_sog_sailing_time_mean_kn", "ais_sailing_time_ratio", "ais_draft_time_mean_m"],
    "Ship": ["ship_age_years", "ship_deadweight_t", "ship_teu_capacity", "ship_design_draught_m", "ship_service_speed_kn", "main_power_kw"],
    "Wind": ["wind_speed_time_mean_ms", "headwind_mean_ms", "crosswind_abs_mean_ms"],
    "WindRatio": ["headwind_time_ratio", "crosswind_time_ratio"],
    "Wave": ["wave_height_time_mean_m", "wave_height_p90_m"],
    "Current": ["current_speed_time_mean_ms", "headcurrent_mean_ms", "crosscurrent_abs_mean_ms"],
    "CurrentRatio": ["headcurrent_time_ratio", "crosscurrent_time_ratio"],
}

DEFAULT_PREDICTORS = ["LR", "RF", "XGBoost", "LightGBM"]
LOGGER = logging.getLogger("MRV-OCA-RGRAPE")


@dataclass
class MinMaxState:
    data_min: np.ndarray
    data_range: np.ndarray


@dataclass
class SplitIndices:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


@dataclass
class GraphOutput:
    imputed_scaled: np.ndarray
    pred_mean_scaled: np.ndarray
    relation_degree: np.ndarray
    relation_diagnostics: Dict[str, object]


def make_model_args() -> argparse.Namespace:
    """Return model settings used by the paper runner; this module has no separate CLI."""
    return argparse.Namespace(
        excel_path="voyage_enriched_data_new.csv",
        sheet_name="all",
        data_schema="voyage_level",
        voyage_feature_set="base_ais_env_ship_enriched",
        target="fuel_total",
        split_mode="random",
        missing_mechanism="mcar",
        missing_rate=0.3,
        train_ratio=0.7,
        val_ratio=0.2,
        voyage_only=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
        out_dir="./mrv_oca_rgrape_results",
        verbose=False,
        graph_hidden_dim=32,
        graph_dropout=0.1,
        graph_lr=1e-3,
        graph_weight_decay=0.0,
        graph_max_epochs=20000,
        original_grape_model_types="EGSAGE_EGSAGE_EGSAGE",
        original_grape_node_dim=64,
        original_grape_edge_dim=64,
        original_grape_edge_mode=1,
        original_grape_activation="relu",
        original_grape_impute_hiddens="64",
        original_grape_impute_activation="relu",
        original_grape_concat_states=False,
        original_grape_post_hiddens=None,
        original_grape_norm_embs=None,
        original_grape_aggr="mean",
        original_grape_known=0.7,
        original_grape_node_mode=0,
        original_grape_eval_every=50,
        original_grape_prediction_batch_size=65536,
        relation_mode=FINAL_RELATION_MODE,
        vessel_neighbor_k=2,
        destination_neighbor_k=5,
        vessel_domain_weight=0.6,
        destination_domain_weight=0.4,
        oca_hidden_dim=64,
        oca_dropout=0.1,
        oca_residual_weight=1.0,
        oca_context_min_coverage=0.0,
        oca_adaptive_fusion=True,
        oca_fusion_temperature=8.0,
        oca_fusion_center=0.5,
        oca_global_missing_weight=1.0,
        oca_pair_batch_size=4096,
    )


def setup_logging(out_dir: Path, verbose: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    formatter = logging.Formatter(fmt="%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(out_dir / "run.log", mode="w", encoding="utf-8")
    fh.setFormatter(formatter)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    ch.setLevel(logging.INFO if verbose else logging.WARNING)
    LOGGER.addHandler(fh)
    LOGGER.addHandler(ch)
    LOGGER.propagate = False


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)






def parse_name_list(text: str, valid_names: Sequence[str], label: str) -> List[str]:
    aliases = {name.lower(): name for name in valid_names}
    parsed = []
    for raw in str(text).split(","):
        key = raw.strip()
        if not key:
            continue
        canonical = aliases.get(key.lower())
        if canonical is None:
            raise ValueError(f"Unknown {label}: {key}. Valid values: {','.join(valid_names)}")
        parsed.append(canonical)
    return parsed


def normalize_column_name(name: object) -> str:
    text = str(name).strip().lower()
    text = re.sub(r"[\s\-/\\()]+", "_", text)
    text = re.sub(r"[^0-9a-zA-Z_]+", "", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text








def load_table_any(path: str, sheet_name: Optional[str]) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")
    lower = path.lower()
    if lower.endswith(".csv"):
        return pd.read_csv(path)
    if lower.endswith(".xlsx") or lower.endswith(".xls"):
        xl = pd.ExcelFile(path)
        if sheet_name and sheet_name in xl.sheet_names:
            return pd.read_excel(path, sheet_name=sheet_name)
        best_sheet, best_score = None, -1
        for name in xl.sheet_names:
            preview = pd.read_excel(path, sheet_name=name)
            score = int(preview.shape[0]) * max(int(preview.shape[1]), 1)
            if score > best_score:
                best_sheet, best_score = name, score
        LOGGER.info("Using Excel sheet: %s", best_sheet)
        return pd.read_excel(path, sheet_name=best_sheet)
    raise ValueError(f"Unsupported input file type: {path}")


def copy_first_available_column(df: pd.DataFrame, target_col: str, source_cols: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for source_col in source_cols:
        if source_col in out.columns:
            out[target_col] = out[source_col]
            return out
    return out




def adapt_voyage_level_dataset_schema(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = df.copy()
    alias_map = {
        "fuel_total": ["primary_mrv_fuel_total", "mrv_sailing_fuel_total", "mrv_all_fuel_total"],
        "use_dogo": ["mrv_sailing_use_dogo", "mrv_all_use_dogo"],
        "use_lfo": ["mrv_sailing_use_lfo", "mrv_all_use_lfo"],
        "use_hfo": ["mrv_sailing_use_hfo", "mrv_all_use_hfo"],
        "cargo_volume": ["primary_mrv_cargo_volume"],
        "total_teu": ["primary_mrv_total_teu"],
        "heavy_ctns": ["primary_mrv_heavy_ctns"],
        "total_distance": ["primary_mrv_total_distance_nm", "mrv_all_distance_nm", "mrv_sailing_distance_nm", "ais_distance_nm"],
        "vessel_code": ["imo", "mmsi"],
        "end_port_code": ["primary_mrv_end_port_code", "ais_dest_mode", "ais_dest_at_end"],
        "end_date": ["primary_mrv_end_utc", "primary_mrv_end_local", "voyage_end_utc"],
        "start_date": ["primary_mrv_start_utc", "voyage_start_utc"],
    }
    for target, sources in alias_map.items():
        if target not in out.columns or out[target].isna().all():
            out = copy_first_available_column(out, target, sources)
    if "end_is_eu" not in out.columns:
        out["end_is_eu"] = np.nan
    out["__data_schema"] = "voyage_level"
    return out






def auto_convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    must_numeric = set(VOYAGE_LEVEL_ENRICHED_NUMERIC_SUPPORT + TARGET_OPTIONS + ["end_is_eu"])
    for col in out.columns:
        if col in must_numeric:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        elif out[col].dtype == object:
            converted = pd.to_numeric(out[col], errors="coerce")
            if converted.notna().mean() >= 0.90:
                out[col] = converted
    return out


def generate_fuel_total(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    fuel_cols = [c for c in ["use_dogo", "use_lfo", "use_hfo"] if c in out.columns]
    for col in fuel_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if fuel_cols and ("fuel_total" not in out.columns or pd.to_numeric(out["fuel_total"], errors="coerce").notna().sum() == 0):
        out["fuel_total"] = out[fuel_cols].fillna(0.0).sum(axis=1)
    return out


def drop_management_columns(df: pd.DataFrame) -> pd.DataFrame:
    management_cols = {"id", "row_id", "index", "unnamed_0", "created_at", "updated_at", "remark", "remarks", "comment", "comments", "audit_status", "status", "file_name", "source_file", "sheet"}
    keep = set(ALL_VOYAGE_NUMERIC_SUPPORT + VOYAGE_CATEGORICAL_SUPPORT + TARGET_OPTIONS)
    keep.update(["end_date", "arrival_date", "arrive_date", "ata", "eta", "departure_date", "start_date", "etd", "date"])
    return df.drop(columns=[c for c in df.columns if c in management_cols and c not in keep], errors="ignore")


def apply_leakage_drop(df: pd.DataFrame, target: str) -> pd.DataFrame:
    out = df.copy()
    leakage_cols = {"fuel_total", "use_hfo", "use_dogo", "use_lfo", "fuel_consumption", "total_fuel_consumption", "co2", "co2_emission", "co2_emissions", "ghg_intensity", "eua", "ets_cost"}
    allowed = set(ALL_VOYAGE_NUMERIC_SUPPORT + VOYAGE_CATEGORICAL_SUPPORT + [target])
    allowed.update(["end_date", "arrival_date", "arrive_date", "ata", "eta", "departure_date", "start_date", "etd", "date"])
    return out.drop(columns=[c for c in leakage_cols if c in out.columns and c not in allowed], errors="ignore")


def load_and_prepare_mrv(args: argparse.Namespace) -> pd.DataFrame:
    df = load_table_any(args.excel_path, args.sheet_name).copy()
    df.columns = [normalize_column_name(c) for c in df.columns]
    df = adapt_voyage_level_dataset_schema(df, args)
    df = drop_management_columns(df)
    df = auto_convert_numeric_columns(df)
    df = generate_fuel_total(df)
    if args.voyage_only and "total_distance" in df.columns:
        df["total_distance"] = pd.to_numeric(df["total_distance"], errors="coerce")
        df = df[df["total_distance"] > 0].copy()
    df = apply_leakage_drop(df, args.target)
    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' is not available after cleaning.")
    df[args.target] = pd.to_numeric(df[args.target], errors="coerce")
    df = df[df[args.target].notna()].reset_index(drop=True)
    for col in VOYAGE_CATEGORICAL_SUPPORT:
        if col in df.columns:
            df[col] = df[col].astype("object").where(df[col].notna(), np.nan)
    return df


def get_numeric_support_candidates(args: argparse.Namespace) -> List[str]:
    """Return the paper's 27 numerical imputation features."""
    if args.voyage_feature_set != "base_ais_env_ship_enriched":
        raise ValueError("Only the paper's base_ais_env_ship_enriched feature set is supported.")
    return list(VOYAGE_LEVEL_ENRICHED_NUMERIC_SUPPORT)


def resolve_support_features(df: pd.DataFrame, args: argparse.Namespace) -> Tuple[List[str], List[str]]:
    numeric_candidates = get_numeric_support_candidates(args)
    numeric_features = []
    for col in numeric_candidates:
        if col in df.columns and pd.to_numeric(df[col], errors="coerce").notna().sum() > 0:
            numeric_features.append(col)
    categorical_features = [c for c in VOYAGE_CATEGORICAL_SUPPORT if c in df.columns]
    if not numeric_features:
        raise ValueError("No numeric support features are available.")
    skipped = [c for c in numeric_candidates if c not in numeric_features]
    if skipped:
        LOGGER.warning("Unavailable/all-NaN numeric support columns skipped: %s", skipped)
    return numeric_features, categorical_features




def make_train_val_test_split(df: pd.DataFrame, args: argparse.Namespace, seed: int) -> SplitIndices:
    """Preserve the paper's two-stage random split (approximately 56/14/30)."""
    if args.split_mode != "random":
        raise ValueError("The paper uses random segment-level partitions only.")
    indices = np.arange(len(df))
    train_val_idx, test_idx = train_test_split(indices, train_size=args.train_ratio, random_state=seed, shuffle=True)
    train_idx, val_idx = train_test_split(train_val_idx, train_size=1.0 - args.val_ratio, random_state=seed + 17, shuffle=True)
    return SplitIndices(np.sort(train_idx), np.sort(val_idx), np.sort(test_idx))


def assert_split_integrity(split: SplitIndices, n_rows: int) -> None:
    """Fail closed when train, validation, and test row sets are not disjoint and exhaustive."""
    train = set(np.asarray(split.train_idx, dtype=int).tolist())
    val = set(np.asarray(split.val_idx, dtype=int).tolist())
    test = set(np.asarray(split.test_idx, dtype=int).tolist())
    assert not (train & val), "Train and validation rows overlap."
    assert not (train & test), "Train and test rows overlap."
    assert not (val & test), "Validation and test rows overlap."
    assert train | val | test == set(range(int(n_rows))), "Split indices are not exhaustive."


def assert_downstream_target_isolated(target: str, numeric_features: Sequence[str]) -> None:
    """Ensure fuel targets cannot enter imputation, graph, relation, scaler, or OCA inputs."""
    normalized = {normalize_column_name(name) for name in numeric_features}
    forbidden = {normalize_column_name(name) for name in TARGET_OPTIONS}
    assert normalize_column_name(target) not in normalized, f"Downstream target '{target}' entered imputation features."
    assert not (normalized & forbidden), f"Fuel target columns entered imputation features: {sorted(normalized & forbidden)}"


def assert_artificial_targets_hidden(
    X_missing_scaled: np.ndarray,
    original_observed_mask: np.ndarray,
    artificial_missing_mask: np.ndarray,
) -> None:
    """Verify artificial labels originate from observed cells and are absent from model inputs."""
    assert artificial_missing_mask.shape == X_missing_scaled.shape == original_observed_mask.shape
    assert np.all(original_observed_mask[artificial_missing_mask]), "Artificial mask includes naturally missing cells."
    assert not np.any(np.isfinite(X_missing_scaled[artificial_missing_mask])), "Artificial ground truth leaked into input values."


def fit_minmax_train(X_raw: np.ndarray, train_idx: np.ndarray) -> MinMaxState:
    train = X_raw[train_idx].astype(float)
    data_min = np.nanmin(np.where(np.isfinite(train), train, np.nan), axis=0)
    data_max = np.nanmax(np.where(np.isfinite(train), train, np.nan), axis=0)
    data_min[~np.isfinite(data_min)] = 0.0
    data_max[~np.isfinite(data_max)] = 1.0
    data_range = data_max - data_min
    data_range[~np.isfinite(data_range)] = 1.0
    data_range[data_range == 0.0] = 1.0
    return MinMaxState(data_min=data_min, data_range=data_range)


def minmax_transform(X_raw: np.ndarray, state: MinMaxState) -> np.ndarray:
    return (X_raw.astype(float) - state.data_min) / state.data_range


def safe_col_means(X: np.ndarray, fallback: float = 0.0) -> np.ndarray:
    means = np.nanmean(X, axis=0)
    means[~np.isfinite(means)] = fallback
    return means


def fill_nan_with_means(X: np.ndarray, means: np.ndarray) -> np.ndarray:
    out = X.copy()
    rows, cols = np.where(~np.isfinite(out))
    if len(rows) > 0:
        out[rows, cols] = means[cols]
    return out


def make_row_mask(n: int, idx: np.ndarray) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    mask[np.asarray(idx, dtype=int)] = True
    return mask


def get_source_group_feature_indices(numeric_features: Sequence[str]) -> Dict[str, List[int]]:
    fmap = {str(feature): idx for idx, feature in enumerate(numeric_features)}
    return {group: [fmap[name] for name in names if name in fmap] for group, names in SOURCE_GROUP_FEATURES.items()}








def make_artificial_missing_mask(X_scaled_true: np.ndarray, original_observed_mask: np.ndarray, train_idx: np.ndarray, missing_rate: float, mechanism: str, seed: int, numeric_features: Optional[Sequence[str]] = None, args: Optional[argparse.Namespace] = None) -> np.ndarray:
    """Generate the original nested MCAR draws; environmental blocks belong to the runner."""
    if mechanism != "mcar":
        raise ValueError("Use the paper runner for environmental-block experiments; this function supports MCAR only.")
    rng = np.random.default_rng(seed)
    mask = np.zeros_like(original_observed_mask, dtype=bool)
    if missing_rate <= 0:
        return mask
    candidates = np.argwhere(original_observed_mask)
    if len(candidates) == 0:
        return mask
    selected = candidates[rng.random(len(candidates)) < missing_rate]
    if len(selected) == 0:
        selected = candidates[rng.choice(len(candidates), size=1)]
    mask[selected[:, 0], selected[:, 1]] = True
    return mask


def make_categorical_onehot(df: pd.DataFrame, categorical_features: Sequence[str], train_idx: np.ndarray) -> np.ndarray:
    if not categorical_features:
        return np.zeros((len(df), 0), dtype=np.float32)
    from sklearn.preprocessing import OneHotEncoder
    cat_df = df[list(categorical_features)].copy()
    for col in categorical_features:
        train_col = cat_df.iloc[train_idx][col].astype("object")
        mode = train_col.dropna().mode()
        fill = mode.iloc[0] if len(mode) > 0 else "__MISSING__"
        cat_df[col] = cat_df[col].astype("object").where(cat_df[col].notna(), fill).astype(str)
    try:
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        enc = OneHotEncoder(handle_unknown="ignore", sparse=False)
    enc.fit(cat_df.iloc[train_idx])
    return np.asarray(enc.transform(cat_df), dtype=np.float32)


def finalize_imputed_scaled(X_imputed_scaled: np.ndarray, X_missing_scaled: np.ndarray, train_idx: np.ndarray) -> np.ndarray:
    out = np.asarray(X_imputed_scaled, dtype=float).copy()
    observed = np.isfinite(X_missing_scaled)
    out[observed] = X_missing_scaled[observed]
    means = safe_col_means(X_missing_scaled[train_idx], fallback=0.0)
    rows, cols = np.where(~np.isfinite(out))
    if len(rows) > 0:
        out[rows, cols] = means[cols]
    return out


def run_mean_imputer(X_missing_scaled: np.ndarray, train_idx: np.ndarray) -> np.ndarray:
    means = safe_col_means(X_missing_scaled[train_idx], fallback=0.0)
    return finalize_imputed_scaled(fill_nan_with_means(X_missing_scaled, means), X_missing_scaled, train_idx)


def prepare_train_for_sklearn_imputer(X_missing_scaled: np.ndarray, train_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    X_train = X_missing_scaled[train_idx].copy()
    means = safe_col_means(X_train, fallback=0.0)
    all_nan_cols = ~np.isfinite(X_train).any(axis=0)
    if np.any(all_nan_cols):
        X_train[:, all_nan_cols] = means[all_nan_cols]
    return X_train, means


def run_knn_imputer(X_missing_scaled: np.ndarray, train_idx: np.ndarray, n_neighbors: int = 5) -> np.ndarray:
    X_train, means = prepare_train_for_sklearn_imputer(X_missing_scaled, train_idx)
    X_all = X_missing_scaled.copy()
    all_nan_cols = ~np.isfinite(X_train).any(axis=0)
    if np.any(all_nan_cols):
        X_all[:, all_nan_cols] = means[all_nan_cols]
    k = max(1, min(n_neighbors, len(train_idx)))
    try:
        imputer = KNNImputer(n_neighbors=k, weights="distance")
        imputer.fit(X_train)
        imputed = imputer.transform(X_all)
    except Exception as exc:
        LOGGER.warning("KNN failed (%s). Falling back to Mean.", exc)
        imputed = run_mean_imputer(X_missing_scaled, train_idx)
    return finalize_imputed_scaled(imputed, X_missing_scaled, train_idx)


def run_mice_imputer(X_missing_scaled: np.ndarray, train_idx: np.ndarray, seed: int) -> np.ndarray:
    X_train, means = prepare_train_for_sklearn_imputer(X_missing_scaled, train_idx)
    X_all = X_missing_scaled.copy()
    all_nan_cols = ~np.isfinite(X_train).any(axis=0)
    if np.any(all_nan_cols):
        X_all[:, all_nan_cols] = means[all_nan_cols]
    try:
        imputer = IterativeImputer(random_state=seed, max_iter=30, initial_strategy="mean", sample_posterior=False, skip_complete=False)
        imputer.fit(X_train)
        imputed = imputer.transform(X_all)
    except Exception as exc:
        LOGGER.warning("MICE failed (%s). Falling back to Mean.", exc)
        imputed = run_mean_imputer(X_missing_scaled, train_idx)
    return finalize_imputed_scaled(imputed, X_missing_scaled, train_idx)


class GainGenerator(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, dim), nn.Sigmoid())
    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, m], dim=1))


class GainDiscriminator(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, dim), nn.Sigmoid())
    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, h], dim=1))


def make_gain_feature_weights(numeric_features: Sequence[str]) -> np.ndarray:
    """Preserve fixed reconstruction weights of the GAIN comparator used in the experiments."""
    fixed = {"total_distance": 2.5, "cargo_volume": 1.5, "total_teu": 1.4, "heavy_ctns": 1.3}
    weights = np.asarray([fixed.get(name, 1.0) for name in numeric_features], dtype=float)
    return (weights / float(np.mean(weights))).astype(np.float32)


def run_gain_imputer(X_true_scaled: np.ndarray, X_missing_scaled: np.ndarray, train_idx: np.ndarray, artificial_missing_mask: np.ndarray, args: argparse.Namespace, seed: int, numeric_features: Optional[Sequence[str]] = None, y: Optional[np.ndarray] = None, split: Optional[SplitIndices] = None) -> np.ndarray:
    set_random_seed(seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    _, d = X_missing_scaled.shape
    if d == 0:
        return X_missing_scaled.copy()
    means = safe_col_means(X_missing_scaled[train_idx], fallback=0.0)
    X_base = fill_nan_with_means(X_missing_scaled, means)
    X_train_true = X_true_scaled[train_idx].copy()
    X_train_missing = X_missing_scaled[train_idx].copy()
    M_train = np.isfinite(X_train_missing).astype(np.float32)
    true_avail = np.isfinite(X_train_true).astype(np.float32)
    X_true_filled = fill_nan_with_means(X_train_true, means)
    X_missing_filled = fill_nan_with_means(X_train_missing, means)
    hidden_dim = max(32, args.graph_hidden_dim)
    gen = GainGenerator(d, hidden_dim).to(device)
    disc = GainDiscriminator(d, hidden_dim).to(device)
    opt_g = torch.optim.Adam(gen.parameters(), lr=args.graph_lr, weight_decay=args.graph_weight_decay)
    opt_d = torch.optim.Adam(disc.parameters(), lr=args.graph_lr, weight_decay=args.graph_weight_decay)
    X_true_t = torch.tensor(X_true_filled, dtype=torch.float32, device=device)
    X_miss_t = torch.tensor(X_missing_filled, dtype=torch.float32, device=device)
    M_t = torch.tensor(M_train, dtype=torch.float32, device=device)
    true_avail_t = torch.tensor(true_avail, dtype=torch.float32, device=device)
    artificial_t = torch.tensor(artificial_missing_mask[train_idx].astype(np.float32), dtype=torch.float32, device=device)
    fw = torch.tensor(make_gain_feature_weights(numeric_features or [f"feature_{j}" for j in range(d)]).reshape(1, -1), dtype=torch.float32, device=device)
    batch_size = max(1, min(128, len(train_idx)))
    epochs = max(1, min(int(args.graph_max_epochs), 1000))
    hint_rate, eps = 0.9, 1e-8
    for _epoch in range(1, epochs + 1):
        perm = torch.randperm(len(train_idx), device=device)
        for start in range(0, len(train_idx), batch_size):
            batch = perm[start:start + batch_size]
            x_true, x_miss, m = X_true_t[batch], X_miss_t[batch], M_t[batch]
            true_avail_b, artificial_b = true_avail_t[batch], artificial_t[batch]
            z = torch.rand_like(x_miss)
            x_tilde = m * x_miss + (1.0 - m) * z
            g_sample = gen(x_tilde, m)
            x_hat = m * x_miss + (1.0 - m) * g_sample
            b = (torch.rand_like(m) < hint_rate).float()
            hint = b * m + 0.5 * (1.0 - b)
            d_prob = disc(x_hat.detach(), hint)
            d_loss = -torch.mean(m * torch.log(d_prob + eps) + (1.0 - m) * torch.log(1.0 - d_prob + eps))
            opt_d.zero_grad(); d_loss.backward(); opt_d.step()
            g_sample = gen(x_tilde, m)
            x_hat = m * x_miss + (1.0 - m) * g_sample
            d_prob = disc(x_hat, hint)
            adv_loss = -torch.mean((1.0 - m) * torch.log(d_prob + eps))
            obs_weight = fw * m * true_avail_b
            obs_mse = torch.sum(obs_weight * ((g_sample - x_true) ** 2)) / torch.clamp(torch.sum(obs_weight), min=1.0)
            hidden_weight = fw * artificial_b * true_avail_b
            hidden_mse = torch.sum(hidden_weight * ((g_sample - x_true) ** 2)) / torch.clamp(torch.sum(hidden_weight), min=1.0)
            g_loss = adv_loss + 10.0 * obs_mse + 10.0 * hidden_mse
            opt_g.zero_grad(); g_loss.backward(); opt_g.step()
    gen.eval()
    with torch.no_grad():
        X_all = torch.tensor(X_base, dtype=torch.float32, device=device)
        M_all = torch.tensor(np.isfinite(X_missing_scaled).astype(np.float32), dtype=torch.float32, device=device)
        G_all = gen(X_all, M_all).cpu().numpy()
    output = X_missing_scaled.copy()
    missing = ~np.isfinite(output)
    output[missing] = G_all[missing]
    return finalize_imputed_scaled(output, X_missing_scaled, train_idx)


def add_directed_pair(edge_dict: Dict[Tuple[int, int], float], i: int, j: int, weight: float) -> None:
    if i == j:
        return
    key = (int(i), int(j))
    edge_dict[key] = max(edge_dict.get(key, 0.0), float(weight))


def add_undirected_pair(edge_dict: Dict[Tuple[int, int], float], i: int, j: int, weight: float) -> None:
    add_directed_pair(edge_dict, i, j, weight)
    add_directed_pair(edge_dict, j, i, weight)




def merge_domain_edges(edge_dict: Dict[Tuple[int, int], float], candidate_edges: Dict[Tuple[int, int], float]) -> int:
    before = len(edge_dict)
    for (src, dst), w in candidate_edges.items():
        add_directed_pair(edge_dict, src, dst, w)
    return len(edge_dict) - before




def build_ordered_group_edges(temp: pd.DataFrame, group_col: str, neighbor_k: int, base_weight: float, args: argparse.Namespace) -> Dict[Tuple[int, int], float]:
    """Connect up to k records on either side in retained input order with fixed weights."""
    edge_dict: Dict[Tuple[int, int], float] = {}
    if neighbor_k <= 0 or temp.empty:
        return edge_dict
    for _, group in temp.groupby(group_col, dropna=False):
        idxs = group["_idx"].to_numpy(dtype=int)
        for pos, i in enumerate(idxs):
            for local_pos in range(max(0, pos - neighbor_k), min(len(idxs), pos + neighbor_k + 1)):
                j = int(idxs[local_pos])
                if int(i) != j:
                    add_undirected_pair(edge_dict, int(i), j, float(base_weight))
    return edge_dict


def build_vessel_domain_edges(df: pd.DataFrame, args: argparse.Namespace) -> Dict[Tuple[int, int], float]:
    """Build paper Vessel connections using retained input order."""
    if "vessel_code" not in df.columns or args.vessel_neighbor_k <= 0:
        return {}
    temp = pd.DataFrame({"_idx": np.arange(len(df)), "_group": df["vessel_code"].astype("object")})
    temp = temp.sort_values(["_group", "_idx"], kind="mergesort")
    return build_ordered_group_edges(temp, "_group", int(args.vessel_neighbor_k), float(args.vessel_domain_weight), args)


def resolve_destination_group_series(df: pd.DataFrame) -> Tuple[Optional[pd.Series], str]:
    """Use destination metadata only; an origin-destination pair is never substituted."""
    for column in ["end_port_code", "end_port"]:
        if column in df.columns:
            return df[column].astype(str), column
    return None, "none"


def build_destination_domain_edges(df: pd.DataFrame, args: argparse.Namespace) -> Tuple[Dict[Tuple[int, int], float], str]:
    """Build fixed-weight Destination connections using retained input order."""
    key, source = resolve_destination_group_series(df)
    if key is None or args.destination_neighbor_k <= 0:
        return {}, source
    valid = key.notna() & (key.astype(str).str.lower() != "nan")
    temp = pd.DataFrame({"_idx": np.arange(len(df)), "_group": key})[valid.to_numpy()].copy()
    temp = temp.sort_values(["_group", "_idx"], kind="mergesort")
    return build_ordered_group_edges(temp, "_group", int(args.destination_neighbor_k), float(args.destination_domain_weight), args), source










def _logical_relation_pairs(edges: Dict[Tuple[int, int], float]) -> set[Tuple[int, int]]:
    """Collapse bidirectional relation edges into unordered voyage pairs."""
    return {(min(int(src), int(dst)), max(int(src), int(dst))) for src, dst in edges if int(src) != int(dst)}


def _relation_distribution_diagnostics(
    edge_dict: Dict[Tuple[int, int], float], n_nodes: int
) -> Dict[str, float]:
    """Summarize final directed weights and unweighted/weighted node degrees."""
    weights = np.asarray(list(edge_dict.values()), dtype=float)
    degree = np.zeros(int(n_nodes), dtype=float)
    weighted_degree = np.zeros(int(n_nodes), dtype=float)
    for (src, _), weight in edge_dict.items():
        degree[int(src)] += 1.0
        weighted_degree[int(src)] += float(weight)

    def stats(prefix: str, values: np.ndarray) -> Dict[str, float]:
        if values.size == 0:
            return {
                f"{prefix}_mean": 0.0, f"{prefix}_std": 0.0,
                f"{prefix}_min": 0.0, f"{prefix}_max": 0.0,
            }
        return {
            f"{prefix}_mean": float(np.mean(values)),
            f"{prefix}_std": float(np.std(values)),
            f"{prefix}_min": float(np.min(values)),
            f"{prefix}_max": float(np.max(values)),
        }

    out: Dict[str, float] = {}
    out.update(stats("relation_weight", weights))
    out.update(stats("node_degree", degree))
    out.update(stats("weighted_node_degree", weighted_degree))
    return out


def build_relation_edges(df: pd.DataFrame, X_missing_scaled: np.ndarray, train_idx: np.ndarray, args: argparse.Namespace, force_relation: bool = False, numeric_features: Optional[Sequence[str]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    """Merge paper Vessel/Destination relations with the original maximum-weight rule."""
    n = len(df)
    mode = args.relation_mode
    if mode not in RELATION_MODE_OPTIONS:
        raise ValueError(f"Unsupported paper relation mode: {mode}")
    edge_dict: Dict[Tuple[int, int], float] = {}
    vessel_edges = {}
    destination_edges = {}
    destination_source = "none"
    if force_relation and mode in {"vessel", "vessel_destination"}:
        vessel_edges = build_vessel_domain_edges(df, args)
    if force_relation and mode in {"destination", "vessel_destination"}:
        destination_edges, destination_source = build_destination_domain_edges(df, args)
    merge_domain_edges(edge_dict, vessel_edges)
    merge_domain_edges(edge_dict, destination_edges)
    vessel_pairs = _logical_relation_pairs(vessel_edges)
    destination_pairs = _logical_relation_pairs(destination_edges)
    diagnostics = {
        "relation_mode": mode, "n_nodes": n, "vessel_edges": len(vessel_edges),
        "destination_edges": len(destination_edges), "destination_key_source": destination_source,
        "standalone_vessel_pair_count": len(vessel_pairs),
        "standalone_destination_pair_count": len(destination_pairs),
        "vessel_destination_overlap_pair_count": len(vessel_pairs & destination_pairs),
        "merge_rule": "max_weight",
    }
    raw_degree = np.zeros(n, dtype=np.float32)
    for (src, _), weight in edge_dict.items():
        raw_degree[src] += weight
    diagnostics["n_relation_edges"] = int(len(edge_dict))
    diagnostics["directed_relation_edge_count"] = int(len(edge_dict))
    diagnostics["final_merged_unique_pair_count"] = len(_logical_relation_pairs(edge_dict))
    diagnostics["mean_relation_degree"] = float(np.mean(raw_degree)) if len(raw_degree) else 0.0
    diagnostics["max_relation_degree"] = float(np.max(raw_degree)) if len(raw_degree) else 0.0
    diagnostics.update(_relation_distribution_diagnostics(edge_dict, n))
    degree = raw_degree.copy()
    max_degree = float(np.max(degree)) if len(degree) else 0.0
    if max_degree > 0:
        degree = degree / max_degree
    if not edge_dict:
        return np.empty((2, 0), dtype=np.int64), np.empty((0,), dtype=np.float32), degree, diagnostics
    pairs = np.array(list(edge_dict.keys()), dtype=np.int64)
    weights = np.array([edge_dict[tuple(pair)] for pair in pairs], dtype=np.float32)
    return pairs.T, weights, degree, diagnostics


def require_original_grape_dependencies() -> None:
    if PyGData is None or get_original_grape_gnn is None or OriginalGrapeMLPNet is None:
        raise RuntimeError("GRAPE/R-GRAPE/OCA methods require torch_geometric and local GRAPE-master/models.")


def parse_original_grape_hidden_layers(text: str) -> List[int]:
    if text is None or str(text).strip() == "":
        return []
    return [int(item) for item in str(text).split("_") if item.strip()]


def make_original_grape_node_features(n_rows: int, n_features: int, node_mode: int) -> np.ndarray:
    if int(node_mode) == 0:
        feature_node = np.zeros((n_features, n_features), dtype=np.float32)
        feature_node[np.arange(n_features), np.arange(n_features)] = 1.0
        sample_node = np.ones((n_rows, n_features), dtype=np.float32)
        return np.vstack([sample_node, feature_node])
    feature_node = np.zeros((n_features, n_features + 1), dtype=np.float32)
    feature_node[np.arange(n_features), np.arange(n_features) + 1] = 1.0
    sample_node = np.zeros((n_rows, n_features + 1), dtype=np.float32)
    sample_node[:, 0] = 1.0
    return np.vstack([sample_node, feature_node])


def cell_pairs_to_original_grape_edges(pairs: np.ndarray, values: np.ndarray, n_rows: int) -> Tuple[np.ndarray, np.ndarray]:
    if len(pairs) == 0:
        return np.empty((2, 0), dtype=np.int64), np.empty((0, 1), dtype=np.float32)
    src = pairs[:, 0].astype(np.int64)
    dst = (n_rows + pairs[:, 1]).astype(np.int64)
    edge_index = np.vstack([np.concatenate([src, dst]), np.concatenate([dst, src])]).astype(np.int64)
    edge_attr = np.concatenate([values, values]).astype(np.float32).reshape(-1, 1)
    return edge_index, edge_attr


def make_original_grape_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        model_types=str(args.original_grape_model_types),
        post_hiddens=args.original_grape_post_hiddens,
        concat_states=bool(args.original_grape_concat_states),
        norm_embs=args.original_grape_norm_embs,
        aggr=str(args.original_grape_aggr),
        node_dim=int(args.original_grape_node_dim),
        edge_dim=int(args.original_grape_edge_dim),
        edge_mode=int(args.original_grape_edge_mode),
        gnn_activation=str(args.original_grape_activation),
        dropout=float(args.graph_dropout),
    )


def make_original_grape_message_graph(value_edge_index: torch.Tensor, value_edge_attr: torch.Tensor, relation_edge_index: torch.Tensor, relation_edge_attr: torch.Tensor, known_prob: float) -> Tuple[torch.Tensor, torch.Tensor]:
    half = int(value_edge_attr.shape[0] // 2)
    if half > 0 and float(known_prob) < 1.0:
        keep_half = torch.rand(half, device=value_edge_attr.device) < float(known_prob)
        if int(keep_half.sum().item()) == 0:
            keep_half = torch.ones(half, dtype=torch.bool, device=value_edge_attr.device)
        keep = torch.cat([keep_half, keep_half], dim=0)
        edge_index, edge_attr = value_edge_index[:, keep], value_edge_attr[keep]
    else:
        edge_index, edge_attr = value_edge_index, value_edge_attr
    if relation_edge_index.numel() > 0:
        edge_index = torch.cat([edge_index, relation_edge_index], dim=1)
        edge_attr = torch.cat([edge_attr, relation_edge_attr], dim=0)
    return edge_index, edge_attr


def predict_original_grape_pairs(node_embeddings: torch.Tensor, impute_model: nn.Module, pairs: torch.Tensor, n_rows: int) -> torch.Tensor:
    sample_idx = pairs[:, 0]
    feature_idx = n_rows + pairs[:, 1]
    return impute_model([node_embeddings[sample_idx], node_embeddings[feature_idx]])[:, 0]


def weighted_original_grape_mse(pred: torch.Tensor, target: torch.Tensor, pairs: torch.Tensor, feature_weights: torch.Tensor) -> torch.Tensor:
    if pred.numel() == 0:
        return torch.zeros((), dtype=torch.float32, device=feature_weights.device)
    weights = feature_weights[pairs[:, 1]]
    return torch.sum(weights * (pred - target) ** 2) / torch.clamp(torch.sum(weights), min=1.0)


def tensor_pairs_and_values(pairs_np: np.ndarray, values_np: np.ndarray, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.tensor(pairs_np, dtype=torch.long, device=device), torch.tensor(values_np.astype(np.float32), dtype=torch.float32, device=device)


class ObservedContextAdapter(nn.Module):
    """Decoder-side row-context adapter. It does not replace the original GRAPE/R-GRAPE backbone.

    Base prediction is still produced by the original GRAPE imputation MLP. The adapter builds a
    target-aware context vector from other observed features in the same row and applies a gated
    residual correction. This is intended to improve low-missingness cases where within-row
    observed features carry strong conditional information.
    """
    def __init__(
        self,
        node_dim: int,
        hidden_dim: int,
        dropout: float,
        residual_weight: float = 1.0,
        adaptive_fusion: bool = True,
        fusion_temperature: float = 8.0,
        fusion_center: float = 0.5,
        global_missing_rate: float = 0.0,
        global_missing_weight: float = 1.0,
        context_min_coverage: float = 0.0,
    ):
        super().__init__()
        self.node_dim = int(node_dim)
        self.hidden_dim = int(hidden_dim)
        self.residual_weight = float(residual_weight)
        self.adaptive_fusion = bool(adaptive_fusion)
        self.fusion_temperature = float(fusion_temperature)
        self.fusion_center = float(fusion_center)
        self.global_missing_rate = float(np.clip(global_missing_rate, 0.0, 1.0))
        self.global_missing_weight = float(np.clip(global_missing_weight, 0.0, 1.0))
        self.context_min_coverage = float(np.clip(context_min_coverage, 0.0, 1.0))
        self.value_encoder = nn.Sequential(
            nn.Linear(node_dim + 1, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, node_dim), nn.ReLU()
        )
        self.query = nn.Linear(node_dim, node_dim)
        self.key = nn.Linear(node_dim, node_dim)
        self.value = nn.Linear(node_dim, node_dim)
        self.residual = nn.Sequential(
            nn.Linear(node_dim * 3 + 1, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1)
        )
        self.gate = nn.Sequential(
            nn.Linear(node_dim * 3 + 1, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1)
        )

    def adaptive_lambda(self, coverage: torch.Tensor) -> torch.Tensor:
        """Explicitly downweight OCA when row context coverage or global observed rate is low."""
        if not self.adaptive_fusion:
            return torch.ones_like(coverage)
        coverage = torch.clamp(coverage, 0.0, 1.0)
        lam = torch.sigmoid(self.fusion_temperature * (coverage - self.fusion_center))
        if self.context_min_coverage > 0.0:
            lam = torch.where(coverage >= self.context_min_coverage, lam, torch.zeros_like(lam))
        global_factor = (1.0 - self.global_missing_weight) + self.global_missing_weight * (1.0 - self.global_missing_rate)
        return torch.clamp(lam * float(global_factor), 0.0, 1.0)

    def row_context(
        self,
        node_embeddings: torch.Tensor,
        X_context: torch.Tensor,
        observed_mask: torch.Tensor,
        pairs: torch.Tensor,
        n_rows: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sample_idx = pairs[:, 0]
        target_feat_idx = pairs[:, 1]
        feat_emb = node_embeddings[n_rows:]
        target_emb = feat_emb[target_feat_idx]
        row_values = X_context[sample_idx]
        row_mask = observed_mask[sample_idx].clone()
        row_mask.scatter_(1, target_feat_idx.unsqueeze(1), False)
        coverage = row_mask.float().mean(dim=1, keepdim=True)
        B, d = row_values.shape
        feat_expand = feat_emb.unsqueeze(0).expand(B, -1, -1)
        value_input = torch.cat([feat_expand, row_values.unsqueeze(2)], dim=2)
        encoded = self.value_encoder(value_input.reshape(B * d, self.node_dim + 1)).reshape(B, d, self.node_dim)
        q = self.query(target_emb).unsqueeze(1)
        k = self.key(encoded)
        scores = torch.sum(q * k, dim=2) / math.sqrt(max(1, self.node_dim))
        # Stable masked softmax. Avoid in-place boolean indexing on CUDA because large
        # OCA pair batches may trigger asynchronous launch failures on some GPUs.
        scores = scores.masked_fill(~row_mask, -1e4)
        attn = torch.softmax(scores, dim=1) * row_mask.float()
        attn = attn / attn.sum(dim=1, keepdim=True).clamp_min(1e-6)
        v = self.value(encoded)
        ctx = torch.sum(attn.unsqueeze(2) * v, dim=1)
        return ctx, coverage

    def forward(
        self,
        base_pred: torch.Tensor,
        node_embeddings: torch.Tensor,
        X_context: torch.Tensor,
        observed_mask: torch.Tensor,
        pairs: torch.Tensor,
        n_rows: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sample_h = node_embeddings[pairs[:, 0]]
        target_h = node_embeddings[n_rows + pairs[:, 1]]
        ctx, coverage = self.row_context(node_embeddings, X_context, observed_mask, pairs, n_rows)
        adapter_input = torch.cat([sample_h, target_h, ctx, coverage], dim=1)
        delta = self.residual(adapter_input).squeeze(1)
        gate = torch.sigmoid(self.gate(adapter_input).squeeze(1))
        coverage_flat = coverage.squeeze(1)
        fusion_lambda = self.adaptive_lambda(coverage_flat)
        pred = base_pred + self.residual_weight * fusion_lambda * gate * delta
        return pred, delta, gate, coverage_flat, fusion_lambda


def predict_oca_pairs(node_embeddings: torch.Tensor, impute_model: nn.Module, oca: ObservedContextAdapter, X_context: torch.Tensor, observed_mask: torch.Tensor, pairs: torch.Tensor, n_rows: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    base = predict_original_grape_pairs(node_embeddings, impute_model, pairs, n_rows)
    pred, _delta, gate, coverage, fusion_lambda = oca(base, node_embeddings, X_context, observed_mask, pairs, n_rows)
    return pred, gate, coverage, fusion_lambda


def predict_oca_pairs_batched_no_grad(
    node_embeddings: torch.Tensor,
    impute_model: nn.Module,
    oca: ObservedContextAdapter,
    X_context: torch.Tensor,
    observed_mask: torch.Tensor,
    pairs: torch.Tensor,
    n_rows: int,
    batch_size: int,
) -> torch.Tensor:
    """Predict OCA pairs in small chunks for validation/inference."""
    preds: List[torch.Tensor] = []
    bs = max(1, int(batch_size))
    for start in range(0, int(pairs.shape[0]), bs):
        chunk = pairs[start : start + bs]
        pred_chunk, _, _, _ = predict_oca_pairs(node_embeddings, impute_model, oca, X_context, observed_mask, chunk, n_rows)
        preds.append(pred_chunk)
    if not preds:
        return torch.empty((0,), dtype=torch.float32, device=node_embeddings.device)
    return torch.cat(preds, dim=0)


def backward_weighted_oca_mse_chunks(
    node_embeddings: torch.Tensor,
    impute_model: nn.Module,
    oca: ObservedContextAdapter,
    X_context: torch.Tensor,
    observed_mask: torch.Tensor,
    pairs: torch.Tensor,
    labels: torch.Tensor,
    feature_weights: torch.Tensor,
    n_rows: int,
    batch_size: int,
    retain_graph_for_last_chunk: bool,
) -> float:
    """Backpropagate one weighted OCA MSE term chunk by chunk.

    The original R-GRAPE backbone is unchanged. This only avoids building a single huge
    decoder-side OCA computation graph for all artificial/observed cells at once.
    """
    if pairs.numel() == 0:
        return 0.0
    denom = torch.clamp(torch.sum(feature_weights[pairs[:, 1]]), min=1.0)
    bs = max(1, int(batch_size))
    total_value = 0.0
    n_chunks = int(math.ceil(int(pairs.shape[0]) / float(bs)))
    for chunk_id, start in enumerate(range(0, int(pairs.shape[0]), bs)):
        end = min(start + bs, int(pairs.shape[0]))
        chunk_pairs = pairs[start:end]
        chunk_labels = labels[start:end]
        pred_chunk, _, _, _ = predict_oca_pairs(node_embeddings, impute_model, oca, X_context, observed_mask, chunk_pairs, n_rows)
        weights = feature_weights[chunk_pairs[:, 1]]
        loss_chunk = torch.sum(weights * (pred_chunk - chunk_labels) ** 2) / denom
        is_last_chunk_of_term = chunk_id == n_chunks - 1
        loss_chunk.backward(retain_graph=(not is_last_chunk_of_term) or retain_graph_for_last_chunk)
        total_value += float(loss_chunk.detach().item())
    return total_value


def predict_oca_full_matrix(model: nn.Module, impute_model: nn.Module, oca: ObservedContextAdapter, x: torch.Tensor, value_edge_index: torch.Tensor, value_edge_attr: torch.Tensor, relation_edge_index: torch.Tensor, relation_edge_attr: torch.Tensor, X_context: torch.Tensor, observed_mask: torch.Tensor, n_rows: int, n_features: int, batch_size: int, device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval(); impute_model.eval(); oca.eval()
    all_pairs = np.array([(i, j) for i in range(n_rows) for j in range(n_features)], dtype=np.int64)
    pred = np.empty((len(all_pairs),), dtype=np.float32)
    gate = np.empty((len(all_pairs),), dtype=np.float32)
    coverage = np.empty((len(all_pairs),), dtype=np.float32)
    fusion_lambda = np.empty((len(all_pairs),), dtype=np.float32)
    with torch.no_grad():
        edge_index, edge_attr = make_original_grape_message_graph(value_edge_index, value_edge_attr, relation_edge_index, relation_edge_attr, known_prob=1.0)
        emb = model(x, edge_attr, edge_index)
        for start in range(0, len(all_pairs), max(1, int(batch_size))):
            batch_np = all_pairs[start:start + max(1, int(batch_size))]
            batch = torch.tensor(batch_np, dtype=torch.long, device=device)
            p, g, c, lam = predict_oca_pairs(emb, impute_model, oca, X_context, observed_mask, batch, n_rows)
            pred[start:start + len(batch_np)] = p.detach().cpu().numpy()
            gate[start:start + len(batch_np)] = g.detach().cpu().numpy()
            coverage[start:start + len(batch_np)] = c.detach().cpu().numpy()
            fusion_lambda[start:start + len(batch_np)] = lam.detach().cpu().numpy()
    return (
        pred.reshape(n_rows, n_features),
        gate.reshape(n_rows, n_features),
        coverage.reshape(n_rows, n_features),
        fusion_lambda.reshape(n_rows, n_features),
    )


def evaluate_original_or_oca_pairs(model: nn.Module, impute_model: nn.Module, x: torch.Tensor, value_edge_index: torch.Tensor, value_edge_attr: torch.Tensor, relation_edge_index: torch.Tensor, relation_edge_attr: torch.Tensor, pairs: torch.Tensor, labels: torch.Tensor, n_rows: int, oca: Optional[ObservedContextAdapter] = None, X_context: Optional[torch.Tensor] = None, observed_mask: Optional[torch.Tensor] = None) -> Tuple[float, float]:
    if pairs.numel() == 0:
        return float("nan"), float("nan")
    model.eval(); impute_model.eval()
    if oca is not None: oca.eval()
    with torch.no_grad():
        edge_index, edge_attr = make_original_grape_message_graph(value_edge_index, value_edge_attr, relation_edge_index, relation_edge_attr, known_prob=1.0)
        emb = model(x, edge_attr, edge_index)
        if oca is None:
            pred = predict_original_grape_pairs(emb, impute_model, pairs, n_rows)
        else:
            assert X_context is not None and observed_mask is not None
            # Validation may include many pairs; chunk the decoder-side OCA path to avoid CUDA launch/memory failures.
            pred = predict_oca_pairs_batched_no_grad(
                emb,
                impute_model,
                oca,
                X_context,
                observed_mask,
                pairs,
                n_rows,
                batch_size=4096,
            )
        mae = F.l1_loss(pred, labels).item()
        rmse = math.sqrt(F.mse_loss(pred, labels).item())
    return float(mae), float(rmse)


def build_original_grape_models(args: argparse.Namespace, n: int, d: int, x: torch.Tensor, device: torch.device) -> Tuple[nn.Module, nn.Module, int]:
    data = PyGData(x=x, edge_attr_dim=1)
    model = get_original_grape_gnn(data, make_original_grape_args(args)).to(device)
    hiddens = parse_original_grape_hidden_layers(str(args.original_grape_impute_hiddens))
    n_layers = len(str(args.original_grape_model_types).split("_"))
    if bool(args.original_grape_concat_states):
        impute_input_dim = int(args.original_grape_node_dim) * n_layers * 2
    else:
        impute_input_dim = int(args.original_grape_node_dim) * 2
    impute_model = OriginalGrapeMLPNet(impute_input_dim, 1, hidden_layer_sizes=hiddens, hidden_activation=str(args.original_grape_impute_activation), dropout=float(args.graph_dropout)).to(device)
    adapter_node_dim = impute_input_dim // 2
    return model, impute_model, adapter_node_dim


def run_original_bipartite_graph_imputer(X_true_scaled: np.ndarray, X_missing_scaled: np.ndarray, split: SplitIndices, artificial_missing_mask: np.ndarray, df: pd.DataFrame, args: argparse.Namespace, seed: int, method_name: str = "GRAPE", force_relation_enhanced: bool = False, numeric_features: Optional[Sequence[str]] = None, y: Optional[np.ndarray] = None, use_oca: bool = False, missing_rate: Optional[float] = None) -> GraphOutput:
    require_original_grape_dependencies()
    assert_split_integrity(split, len(df))
    assert_artificial_targets_hidden(X_missing_scaled, np.isfinite(X_true_scaled), artificial_missing_mask)
    set_random_seed(seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    n, d = X_missing_scaled.shape

    relation_edge_index_np, relation_edge_weight_np, relation_degree, relation_diagnostics = build_relation_edges(
        df=df, X_missing_scaled=X_missing_scaled, train_idx=split.train_idx, args=args,
        force_relation=force_relation_enhanced, numeric_features=numeric_features,
    )
    relation_diagnostics.update({"method_name": method_name, "graph_arch": "original_bipartite", "use_oca": bool(use_oca)})

    observed_after_mask = np.isfinite(X_missing_scaled)
    observed_pairs_np = np.argwhere(observed_after_mask)
    observed_values_np = X_missing_scaled[observed_after_mask].astype(np.float32)
    value_edge_index_np, value_edge_attr_np = cell_pairs_to_original_grape_edges(observed_pairs_np, observed_values_np, n)

    train_mask = make_row_mask(n, split.train_idx)
    val_mask = make_row_mask(n, split.val_idx)
    test_mask = make_row_mask(n, split.test_idx)
    train_artificial_mask = artificial_missing_mask & train_mask[:, None]
    val_artificial_mask = artificial_missing_mask & val_mask[:, None]
    test_artificial_mask = artificial_missing_mask & test_mask[:, None]
    train_observed_mask = observed_after_mask & train_mask[:, None]

    artificial_pairs, artificial_labels = tensor_pairs_and_values(np.argwhere(train_artificial_mask), X_true_scaled[train_artificial_mask].astype(np.float32), device)
    observed_pairs_train, observed_labels_train = tensor_pairs_and_values(np.argwhere(train_observed_mask), X_true_scaled[train_observed_mask].astype(np.float32), device)
    val_pairs, val_labels = tensor_pairs_and_values(np.argwhere(val_artificial_mask), X_true_scaled[val_artificial_mask].astype(np.float32), device)

    if artificial_pairs.numel() == 0 and observed_pairs_train.numel() == 0:
        fallback = run_mean_imputer(X_missing_scaled, split.train_idx)
        return GraphOutput(fallback, fallback.copy(), relation_degree, relation_diagnostics)

    x_np = make_original_grape_node_features(n, d, int(args.original_grape_node_mode))
    x = torch.tensor(x_np, dtype=torch.float32, device=device)
    value_edge_index = torch.tensor(value_edge_index_np, dtype=torch.long, device=device)
    value_edge_attr = torch.tensor(value_edge_attr_np, dtype=torch.float32, device=device)
    if relation_edge_index_np.size > 0:
        rel_src = relation_edge_index_np.astype(np.int64)
        # Convert observation-observation relation edge from row indices to sample-node indices.
        relation_edge_index = torch.tensor(rel_src, dtype=torch.long, device=device)
        relation_edge_attr = torch.tensor(relation_edge_weight_np.reshape(-1, 1), dtype=torch.float32, device=device)
    else:
        relation_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        relation_edge_attr = torch.empty((0, 1), dtype=torch.float32, device=device)

    model, impute_model, adapter_node_dim = build_original_grape_models(args, n, d, x, device)
    oca = None
    if use_oca:
        oca = ObservedContextAdapter(
            node_dim=adapter_node_dim,
            hidden_dim=max(8, int(args.oca_hidden_dim)),
            dropout=float(args.oca_dropout),
            residual_weight=float(args.oca_residual_weight),
            adaptive_fusion=bool(getattr(args, "oca_adaptive_fusion", True)),
            fusion_temperature=float(getattr(args, "oca_fusion_temperature", 8.0)),
            fusion_center=float(getattr(args, "oca_fusion_center", 0.5)),
            global_missing_rate=float(args.missing_rate if missing_rate is None else missing_rate),
            global_missing_weight=float(getattr(args, "oca_global_missing_weight", 1.0)),
            context_min_coverage=float(getattr(args, "oca_context_min_coverage", 0.0)),
        ).to(device)

    feature_weights_np = np.ones(d, dtype=np.float32)
    feature_weights = torch.tensor(feature_weights_np, dtype=torch.float32, device=device)

    oca_train_mode = "joint"
    params = list(model.parameters()) + list(impute_model.parameters()) + ([] if oca is None else list(oca.parameters()))
    optimizer = torch.optim.Adam(params, lr=float(args.graph_lr), weight_decay=float(args.graph_weight_decay))
    best_score, best_epoch = float("inf"), 0
    best_state = None
    eval_every = max(1, int(args.original_grape_eval_every))
    loss_scope = "combined"

    means = safe_col_means(X_missing_scaled[split.train_idx], fallback=0.0)
    X_context_np = fill_nan_with_means(X_missing_scaled, means).astype(np.float32)
    X_context_t = torch.tensor(X_context_np, dtype=torch.float32, device=device)
    observed_mask_t = torch.tensor(observed_after_mask.astype(bool), dtype=torch.bool, device=device)

    if args.verbose:
        LOGGER.info(
            "%s | original GRAPE backbone | relation_edges=%d | OCA=%s | oca_train_mode=%s | train_artificial=%d | train_observed=%d",
            method_name,
            int(relation_edge_attr.shape[0]),
            bool(use_oca),
            oca_train_mode if use_oca else "none",
            int(train_artificial_mask.sum()),
            int(train_observed_mask.sum()),
        )

    # Jointly optimize the graph encoder, decoder and OCA when present.
    for epoch in range(1, int(args.graph_max_epochs) + 1):
        model.train(); impute_model.train()
        if oca is not None:
            oca.train()
        edge_index, edge_attr = make_original_grape_message_graph(value_edge_index, value_edge_attr, relation_edge_index, relation_edge_attr, known_prob=float(args.original_grape_known))
        emb = model(x, edge_attr, edge_index)
        optimizer.zero_grad()
        if oca is None:
            losses: List[torch.Tensor] = []
            if artificial_pairs.numel() > 0:
                pred_art = predict_original_grape_pairs(emb, impute_model, artificial_pairs, n)
                losses.append(weighted_original_grape_mse(pred_art, artificial_labels, artificial_pairs, feature_weights))
            if observed_pairs_train.numel() > 0:
                pred_obs = predict_original_grape_pairs(emb, impute_model, observed_pairs_train, n)
                losses.append(weighted_original_grape_mse(pred_obs, observed_labels_train, observed_pairs_train, feature_weights))
            if not losses and observed_pairs_train.numel() > 0:
                pred_obs = predict_original_grape_pairs(emb, impute_model, observed_pairs_train, n)
                losses.append(weighted_original_grape_mse(pred_obs, observed_labels_train, observed_pairs_train, feature_weights))
            loss = torch.stack(losses).sum()
            loss.backward()
            loss_value = float(loss.detach().item())
        else:
            # OCA creates B x d x hidden tensors for pair-level row context. Backpropagate pair losses
            # in chunks so GPU memory stays bounded. This does not change the R-GRAPE graph structure.
            pair_terms: List[Tuple[torch.Tensor, torch.Tensor]] = []
            if artificial_pairs.numel() > 0:
                pair_terms.append((artificial_pairs, artificial_labels))
            if observed_pairs_train.numel() > 0:
                pair_terms.append((observed_pairs_train, observed_labels_train))
            if not pair_terms and observed_pairs_train.numel() > 0:
                pair_terms.append((observed_pairs_train, observed_labels_train))
            loss_value = 0.0
            for term_id, (term_pairs, term_labels) in enumerate(pair_terms):
                loss_value += backward_weighted_oca_mse_chunks(
                    node_embeddings=emb,
                    impute_model=impute_model,
                    oca=oca,
                    X_context=X_context_t,
                    observed_mask=observed_mask_t,
                    pairs=term_pairs,
                    labels=term_labels,
                    feature_weights=feature_weights,
                    n_rows=n,
                    batch_size=int(getattr(args, "oca_pair_batch_size", 4096)),
                    retain_graph_for_last_chunk=(term_id < len(pair_terms) - 1),
                )
        torch.nn.utils.clip_grad_norm_(params, 5.0); optimizer.step()

        if epoch == 1 or epoch % eval_every == 0 or epoch == int(args.graph_max_epochs):
            eval_oca = oca
            val_mae, val_rmse = evaluate_original_or_oca_pairs(model, impute_model, x, value_edge_index, value_edge_attr, relation_edge_index, relation_edge_attr, val_pairs, val_labels, n, oca=eval_oca, X_context=X_context_t, observed_mask=observed_mask_t)
            score = val_mae if np.isfinite(val_mae) else float(loss_value)
            if score < best_score:
                best_score, best_epoch = float(score), int(epoch)
                best_state = {
                    "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    "impute_model": {k: v.detach().cpu().clone() for k, v in impute_model.state_dict().items()},
                    "oca": None if oca is None else {k: v.detach().cpu().clone() for k, v in oca.state_dict().items()},
                }
            if args.verbose:
                LOGGER.info("%s epoch %d/%d | loss=%.6f | val_mae=%.6f | val_rmse=%.6f | best_epoch=%d", method_name, epoch, int(args.graph_max_epochs), float(loss_value), float(val_mae), float(val_rmse), best_epoch)

    if best_state is not None:
        model.load_state_dict(best_state["model"]); impute_model.load_state_dict(best_state["impute_model"])
        if oca is not None and best_state["oca"] is not None:
            oca.load_state_dict(best_state["oca"])



    if oca is None:
        pred_mean = predict_original_grape_full_matrix(model, impute_model, x, value_edge_index, value_edge_attr, relation_edge_index, relation_edge_attr, n, d, int(args.original_grape_prediction_batch_size), device)
        gate_mean = float("nan"); gate_std = float("nan")
        coverage_mean = float("nan"); coverage_std = float("nan")
        fusion_lambda_mean = float("nan"); fusion_lambda_std = float("nan")
        correction_abs_mean = float("nan"); correction_mean = float("nan"); correction_std = float("nan")
    else:
        pred_mean, gate_mat, coverage_mat, fusion_lambda_mat = predict_oca_full_matrix(model, impute_model, oca, x, value_edge_index, value_edge_attr, relation_edge_index, relation_edge_attr, X_context_t, observed_mask_t, n, d, int(args.original_grape_prediction_batch_size), device)
        base_pred_mean = predict_original_grape_full_matrix(model, impute_model, x, value_edge_index, value_edge_attr, relation_edge_index, relation_edge_attr, n, d, int(args.original_grape_prediction_batch_size), device)
        focus = test_artificial_mask if np.any(test_artificial_mask) else ~np.isfinite(X_missing_scaled)
        if not np.any(focus):
            focus = np.ones_like(X_missing_scaled, dtype=bool)
        gate_values = gate_mat[focus]
        coverage_values = coverage_mat[focus]
        fusion_lambda_values = fusion_lambda_mat[focus]
        correction_values = (pred_mean - base_pred_mean)[focus]
        gate_mean = float(np.nanmean(gate_values)); gate_std = float(np.nanstd(gate_values))
        coverage_mean = float(np.nanmean(coverage_values)); coverage_std = float(np.nanstd(coverage_values))
        fusion_lambda_mean = float(np.nanmean(fusion_lambda_values)); fusion_lambda_std = float(np.nanstd(fusion_lambda_values))
        correction_abs_mean = float(np.nanmean(np.abs(correction_values)))
        correction_mean = float(np.nanmean(correction_values)); correction_std = float(np.nanstd(correction_values))

    imputed = X_missing_scaled.copy()
    missing = ~np.isfinite(imputed)
    imputed[missing] = pred_mean[missing]
    imputed = finalize_imputed_scaled(imputed, X_missing_scaled, split.train_idx)
    relation_diagnostics.update({
        "best_epoch": int(best_epoch), "best_validation_MAE": float(best_score) if np.isfinite(best_score) else float("nan"),
        "best_valid_mae": float(best_score) if np.isfinite(best_score) else float("nan"),
        "final_training_epoch": int(args.graph_max_epochs),
        "loss_scope": loss_scope, "n_value_edges": int(value_edge_attr.shape[0]),
        "n_train_artificial_cells": int(train_artificial_mask.sum()), "n_train_observed_cells": int(train_observed_mask.sum()),
        "n_val_artificial_cells": int(val_artificial_mask.sum()), "n_test_artificial_cells": int(test_artificial_mask.sum()),
        "original_grape_model_types": str(args.original_grape_model_types), "original_grape_known": float(args.original_grape_known),
        "oca_pair_batch_size": int(getattr(args, "oca_pair_batch_size", 4096)),
        "oca_train_mode": oca_train_mode if use_oca else "none",
        "oca_diagnostic_scope": "test_artificial_cells",
        "oca_mean_gate_on_missing": gate_mean, "oca_std_gate_on_missing": gate_std,
        "oca_mean_context_coverage_on_missing": coverage_mean, "oca_std_context_coverage_on_missing": coverage_std,
        "oca_adaptive_fusion": bool(getattr(args, "oca_adaptive_fusion", True)),
        "oca_fusion_temperature": float(getattr(args, "oca_fusion_temperature", 8.0)),
        "oca_fusion_center": float(getattr(args, "oca_fusion_center", 0.5)),
        "oca_global_missing_weight": float(getattr(args, "oca_global_missing_weight", 1.0)),
        "oca_mean_adaptive_lambda_on_missing": fusion_lambda_mean,
        "oca_std_adaptive_lambda_on_missing": fusion_lambda_std,
        "oca_mean_abs_correction_on_missing": correction_abs_mean,
        "oca_mean_signed_correction_on_missing": correction_mean,
        "oca_std_correction_on_missing": correction_std,
    })
    return GraphOutput(imputed_scaled=imputed, pred_mean_scaled=pred_mean, relation_degree=relation_degree, relation_diagnostics=relation_diagnostics)


def predict_original_grape_full_matrix(model: nn.Module, impute_model: nn.Module, x: torch.Tensor, value_edge_index: torch.Tensor, value_edge_attr: torch.Tensor, relation_edge_index: torch.Tensor, relation_edge_attr: torch.Tensor, n_rows: int, n_features: int, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval(); impute_model.eval()
    all_pairs = np.array([(i, j) for i in range(n_rows) for j in range(n_features)], dtype=np.int64)
    pred = np.empty((len(all_pairs),), dtype=np.float32)
    with torch.no_grad():
        edge_index, edge_attr = make_original_grape_message_graph(value_edge_index, value_edge_attr, relation_edge_index, relation_edge_attr, known_prob=1.0)
        emb = model(x, edge_attr, edge_index)
        for start in range(0, len(all_pairs), max(1, int(batch_size))):
            batch_np = all_pairs[start:start + max(1, int(batch_size))]
            batch = torch.tensor(batch_np, dtype=torch.long, device=device)
            pred[start:start + len(batch_np)] = predict_original_grape_pairs(emb, impute_model, batch, n_rows).detach().cpu().numpy()
    return pred.reshape(n_rows, n_features)


def compute_metric_pair(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if valid.sum() == 0:
        return float("nan"), float("nan")
    mae = mean_absolute_error(y_true[valid], y_pred[valid])
    rmse = math.sqrt(mean_squared_error(y_true[valid], y_pred[valid]))
    return float(mae), float(rmse)


def compute_imputation_records(X_true_scaled: np.ndarray, X_imp_scaled: np.ndarray, scaler_state: MinMaxState, test_artificial_mask: np.ndarray, numeric_features: Sequence[str], seed: int, missing_rate: float, imputer: str, args: argparse.Namespace) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    X_imp_clip = np.clip(X_imp_scaled, 0.0, 1.0)
    mask = test_artificial_mask
    mae_clip, rmse_clip = compute_metric_pair(X_true_scaled[mask], X_imp_clip[mask])
    overall = {
        "seed": seed, "missing_rate": missing_rate, "missing_mechanism": args.missing_mechanism,
        "feature_set": args.voyage_feature_set, "data_schema": args.data_schema, "imputer": imputer,
        "target": args.target, "split_mode": args.split_mode, "voyage_only": bool(args.voyage_only),
        "MAE_scaled_clipped": mae_clip, "RMSE_scaled_clipped": rmse_clip,
        "n_masked_cells": int(mask.sum()),
    }
    per_feature = []
    for j, feature in enumerate(numeric_features):
        f_mask = test_artificial_mask[:, j]
        f_mae_clip, f_rmse_clip = compute_metric_pair(X_true_scaled[f_mask, j], X_imp_clip[f_mask, j])
        per_feature.append({
            "seed": seed, "missing_rate": missing_rate, "missing_mechanism": args.missing_mechanism,
            "feature_set": args.voyage_feature_set, "imputer": imputer, "feature": feature,
            "MAE_scaled_clipped": f_mae_clip, "RMSE_scaled_clipped": f_rmse_clip,
            "n_masked_cells": int(f_mask.sum()),
        })
    return overall, per_feature


def compute_source_group_imputation_records(X_true_scaled: np.ndarray, X_imp_scaled: np.ndarray, test_artificial_mask: np.ndarray, numeric_features: Sequence[str], seed: int, missing_rate: float, imputer: str, args: argparse.Namespace) -> List[Dict[str, object]]:
    records = []
    group_indices = get_source_group_feature_indices(numeric_features)
    X_imp_clip = np.clip(X_imp_scaled, 0.0, 1.0)
    for group, cols in group_indices.items():
        if not cols:
            continue
        group_mask = np.zeros_like(test_artificial_mask, dtype=bool)
        group_mask[:, cols] = test_artificial_mask[:, cols]
        mae, rmse = compute_metric_pair(X_true_scaled[group_mask], X_imp_clip[group_mask])
        records.append({
            "seed": seed, "missing_rate": missing_rate, "missing_mechanism": args.missing_mechanism,
            "feature_set": args.voyage_feature_set, "relation_mode": args.relation_mode,
            "imputer": imputer, "source_group": group, "MAE_scaled_clipped": mae,
            "RMSE_scaled_clipped": rmse, "n_masked_cells": int(group_mask.sum()),
        })
    return records




def compute_downstream_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if valid.sum() == 0:
        return {"MAE": float("nan"), "RMSE": float("nan"), "R2": float("nan")}
    yt, yp = y_true[valid], y_pred[valid]
    return {
        "MAE": float(mean_absolute_error(yt, yp)),
        "RMSE": float(math.sqrt(mean_squared_error(yt, yp))),
        "R2": float(r2_score(yt, yp)) if len(yt) >= 2 else float("nan"),
    }


def make_downstream_regressor(predictor: str, seed: int):
    if predictor == "LR":
        return LinearRegression()
    if predictor == "RF":
        return RandomForestRegressor(n_estimators=300, max_depth=None, min_samples_leaf=2, n_jobs=-1, random_state=seed)
    if predictor == "XGBoost":
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, objective="reg:squarederror", tree_method="hist", random_state=seed, n_jobs=-1)
    if predictor == "LightGBM":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31, subsample=0.9, colsample_bytree=0.9, random_state=seed, n_jobs=-1, verbose=-1)
    raise ValueError(f"Unsupported predictor: {predictor}")


def run_downstream_predictor(X_numeric_scaled_imputed: np.ndarray, cat_onehot: np.ndarray, y: np.ndarray, split: SplitIndices, seed: int, missing_rate: float, imputer: str, predictor: str, args: argparse.Namespace) -> Dict[str, object]:
    X = np.hstack([np.clip(X_numeric_scaled_imputed, 0.0, 1.0), cat_onehot]).astype(float)
    model = make_downstream_regressor(predictor, seed)
    model.fit(X[split.train_idx], y[split.train_idx])
    y_pred = np.asarray(model.predict(X[split.test_idx]), dtype=float)
    return {
        "seed": seed, "missing_rate": missing_rate, "missing_mechanism": args.missing_mechanism,
        "feature_set": args.voyage_feature_set, "imputer": imputer, "predictor": predictor,
        "model_combo": f"{imputer}+{predictor}", "target": args.target, **compute_downstream_metrics(y[split.test_idx], y_pred),
    }










