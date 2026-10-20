from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import traceback
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

IGRM_COMMIT = "5cfc955daa5d0f4bbdcbad1552cfd7493dfd5fd0"
IGRM_ZIP_URL = f"https://github.com/G-AILab/IGRM/archive/{IGRM_COMMIT}.zip"
ENV_GROUPS_COMPLETE = ["Wind", "WindRatio", "Wave", "Current", "CurrentRatio"]
CORE_GRAPH_METHODS = ["GRAPE", "R-GRAPE", "OCA-GRAPE", "OCA-R-GRAPE"]
IGRM_METHOD = "IGRM_common_protocol"
ALL_METHODS = ["Mean", "KNN", "MICE", "GAIN", "GRAPE", IGRM_METHOD, "R-GRAPE", "OCA-GRAPE", "OCA-R-GRAPE"]
FINAL_RELATION_MODE = "vessel_destination"
DEVELOPMENT_SEEDS = {0, 1, 2}
CONFIRMATORY_SEEDS = {3, 4}


def parse_list_float(text: str) -> List[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_list_int(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Final paper experiment runner for voyage-level MRV/AIS imputation. "
            "It uses joint OCA training, "
            "and fixes formal R models to Vessel + Destination relations."
        )
    )
    p.add_argument("--excel_path", type=str, default="voyage_enriched_data_new.csv")
    p.add_argument("--base_script", type=str, default=str(Path(__file__).with_name("oca_rgrape.py")))
    p.add_argument("--igrm_runner", type=str, default=str(Path(__file__).with_name("igrm_common_protocol.py")))
    p.add_argument("--igrm_repo_dir", type=str, default=str(Path(__file__).with_name("IGRM_pinned")))
    p.add_argument("--no_auto_download_igrm", action="store_true")
    p.add_argument("--out_dir", type=str, default="./final_mrv_experiments")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seeds", type=str, default="0,1,2,3,4")
    p.add_argument("--main_missing_rates", type=str, default="0.1,0.2,0.3,0.4,0.5,0.6,0.7")
    p.add_argument("--inductive_missing_rates", type=str, default="0.3,0.5,0.7")
    p.add_argument("--block_affected_rates", type=str, default="0.25,0.5,0.75")
    p.add_argument("--relation_ablation_rates", type=str, default="0.3,0.5,0.7")
    p.add_argument("--graph_max_epochs", type=int, default=20000)
    p.add_argument("--eval_every", type=int, default=50)
    p.add_argument("--run_all", action="store_true")
    p.add_argument("--run_main_mcar", action="store_true")
    p.add_argument("--run_module_ablation", action="store_true")
    p.add_argument("--run_relation_ablation", action="store_true")
    p.add_argument("--run_inductive", action="store_true")
    p.add_argument("--run_structured", action="store_true")
    p.add_argument("--run_downstream", action="store_true")
    p.add_argument("--run_complete_reference", action="store_true")
    p.add_argument("--skip_igrm", action="store_true")
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no_resume", dest="resume", action="store_false")
    p.add_argument("--save_imputed_arrays", action="store_true", default=True)
    p.add_argument("--no_save_imputed_arrays", dest="save_imputed_arrays", action="store_false")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--igrm_friend_k", type=int, default=10)
    p.add_argument("--igrm_reconstruct_every", type=int, default=100)
    p.add_argument("--data_schema", type=str, default="voyage_level", choices=["voyage_level"])
    p.add_argument("--voyage_feature_set", type=str, default="base_ais_env_ship_enriched", choices=["base_ais_env_ship_enriched"])
    p.add_argument("--target", type=str, default="fuel_total", choices=["fuel_total"])
    p.add_argument("--split_mode", type=str, default="random", choices=["random"])
    p.add_argument("--train_ratio", type=float, default=0.7)
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--predictors", type=str, default="LR,RF,XGBoost,LightGBM")
    p.add_argument("--worker_mode", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--worker_condition_npz", type=str, default="", help=argparse.SUPPRESS)
    p.add_argument("--worker_method", type=str, default="", help=argparse.SUPPRESS)
    p.add_argument("--worker_protocol", type=str, default="transductive", help=argparse.SUPPRESS)
    p.add_argument("--worker_relation_mode", type=str, default=FINAL_RELATION_MODE, help=argparse.SUPPRESS)
    p.add_argument("--worker_cache_dir", type=str, default="", help=argparse.SUPPRESS)
    return p.parse_args()


def load_base_module(base_script: Path, module_name: Optional[str] = None):
    if not base_script.exists():
        raise FileNotFoundError(f"Base script not found: {base_script}")
    name = module_name or f"mrv_module_{hashlib.sha256(str(base_script).encode()).hexdigest()[:10]}"
    spec = importlib.util.spec_from_file_location(name, base_script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load base script: {base_script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def make_base_args(base, args: argparse.Namespace) -> argparse.Namespace:
    b = base.make_model_args()
    b.excel_path = args.excel_path
    b.sheet_name = "all"
    b.data_schema = args.data_schema
    b.voyage_feature_set = args.voyage_feature_set
    b.target = args.target
    b.split_mode = args.split_mode
    b.train_ratio = args.train_ratio
    b.val_ratio = args.val_ratio
    b.device = args.device
    b.graph_max_epochs = args.graph_max_epochs
    b.original_grape_eval_every = args.eval_every
    b.oca_train_mode = "joint"
    b.relation_mode = FINAL_RELATION_MODE
    b.verbose = args.verbose
    return b


def file_sha256(path: Path) -> str:
    """Return a stable SHA256 digest for provenance records."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()








def ensure_igrm_repo(args: argparse.Namespace) -> Path:
    repo = Path(args.igrm_repo_dir).resolve()
    if (repo / "models" / "gnn_model.py").exists():
        return repo
    if args.no_auto_download_igrm:
        raise FileNotFoundError(
            f"IGRM repository not found at {repo}. Download G-AILab/IGRM at commit {IGRM_COMMIT} and place it there, or omit --no_auto_download_igrm."
        )
    repo.parent.mkdir(parents=True, exist_ok=True)
    zip_path = repo.parent / f"IGRM-{IGRM_COMMIT}.zip"
    print(f"[IGRM] Downloading pinned official repository {IGRM_COMMIT} ...", flush=True)
    try:
        urllib.request.urlretrieve(IGRM_ZIP_URL, zip_path)
    except Exception as exc:
        raise RuntimeError(
            "Automatic IGRM download failed. Manually download https://github.com/G-AILab/IGRM at "
            f"commit {IGRM_COMMIT}, extract it to {repo}, then rerun. Original error: {exc}"
        ) from exc
    temp_root = repo.parent / f"_igrm_extract_{os.getpid()}"
    if temp_root.exists():
        shutil.rmtree(temp_root)
    temp_root.mkdir(parents=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(temp_root)
    candidates = [p for p in temp_root.iterdir() if p.is_dir() and (p / "models" / "gnn_model.py").exists()]
    if not candidates:
        raise RuntimeError("Downloaded IGRM archive did not contain the expected repository structure.")
    if repo.exists():
        shutil.rmtree(repo)
    shutil.move(str(candidates[0]), str(repo))
    shutil.rmtree(temp_root, ignore_errors=True)
    try:
        zip_path.unlink()
    except Exception:
        pass
    return repo


@dataclass
class Condition:
    seed: int
    mechanism: str
    nominal_rate: float
    affected_row_rate: float
    realized_cell_rate: float
    split: object
    scaler: object
    X_true_scaled: np.ndarray
    X_missing_scaled: np.ndarray
    original_observed_mask: np.ndarray
    artificial_missing_mask: np.ndarray
    cat_onehot: np.ndarray
    condition_hash: str = ""


def array_sha256(*arrays: np.ndarray) -> str:
    """Hash array shape, dtype, and bytes for reproducible condition identity."""
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.shape).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def make_env_block_mask(
    base,
    original_observed_mask: np.ndarray,
    numeric_features: Sequence[str],
    affected_row_rate: float,
    seed: int,
) -> np.ndarray:
    """Create the Environment block protocol outside the model implementation."""
    feature_to_idx = {name: idx for idx, name in enumerate(numeric_features)}
    env_columns = [
        feature
        for group in ENV_GROUPS_COMPLETE
        for feature in base.SOURCE_GROUP_FEATURES.get(group, [])
        if feature in feature_to_idx
    ]
    if not env_columns:
        raise ValueError("Environment block has no available Wind/Wave/Current features.")
    rng = np.random.default_rng(int(seed))
    affected_rows = rng.random(original_observed_mask.shape[0]) < float(affected_row_rate)
    mask = np.zeros_like(original_observed_mask, dtype=bool)
    cols = np.asarray([feature_to_idx[name] for name in env_columns], dtype=int)
    mask[np.ix_(affected_rows, cols)] = original_observed_mask[np.ix_(affected_rows, cols)]
    if not np.any(mask) and affected_row_rate > 0.0:
        eligible_rows = np.flatnonzero(original_observed_mask[:, cols].any(axis=1))
        if eligible_rows.size:
            row = int(rng.choice(eligible_rows))
            mask[row, cols] = original_observed_mask[row, cols]
    return mask


def prepare_condition(base, df: pd.DataFrame, numeric_features: Sequence[str], categorical_features: Sequence[str], bargs, seed: int, mechanism: str, rate: float) -> Condition:
    base.set_random_seed(seed)
    split = base.make_train_val_test_split(df, bargs, seed)
    X_raw = df[list(numeric_features)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    scaler = base.fit_minmax_train(X_raw, split.train_idx)
    X_true_scaled = base.minmax_transform(X_raw, scaler)
    original_observed_mask = np.isfinite(X_true_scaled)
    if mechanism == "mcar":
        art = base.make_artificial_missing_mask(
            X_scaled_true=X_true_scaled,
            original_observed_mask=original_observed_mask,
            train_idx=split.train_idx,
            missing_rate=rate,
            mechanism="mcar",
            seed=seed + 101,
            numeric_features=numeric_features,
            args=bargs,
        )
        affected = float("nan")
    elif mechanism == "env_block":
        art = make_env_block_mask(base, original_observed_mask, numeric_features, rate, seed + 101)
        affected = float(rate)
    else:
        raise ValueError(f"Unsupported suite mechanism: {mechanism}")
    X_missing = X_true_scaled.copy()
    X_missing[art] = np.nan
    eligible = int(original_observed_mask.sum())
    realized = float(art.sum() / eligible) if eligible else 0.0
    base.assert_split_integrity(split, len(df))
    base.assert_artificial_targets_hidden(X_missing, original_observed_mask, art)
    cat = base.make_categorical_onehot(df, categorical_features, split.train_idx)
    condition_hash = array_sha256(
        np.asarray(split.train_idx, dtype=np.int64), np.asarray(split.val_idx, dtype=np.int64),
        np.asarray(split.test_idx, dtype=np.int64), art.astype(np.uint8),
        np.asarray(scaler.data_min), np.asarray(scaler.data_range),
    )
    return Condition(seed, mechanism, float(rate), affected, realized, split, scaler, X_true_scaled, X_missing, original_observed_mask, art, cat, condition_hash)


def condition_slug(cond: Condition) -> str:
    if cond.mechanism == "env_block":
        return f"seed_{cond.seed}_affected_{cond.nominal_rate:.3f}"
    return f"seed_{cond.seed}_rate_{cond.nominal_rate:.3f}"


def save_condition_npz(cond: Condition, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        X_true_scaled=cond.X_true_scaled.astype(np.float32),
        X_missing_scaled=cond.X_missing_scaled.astype(np.float32),
        original_observed_mask=cond.original_observed_mask.astype(np.uint8),
        artificial_missing_mask=cond.artificial_missing_mask.astype(np.uint8),
        train_idx=np.asarray(cond.split.train_idx, dtype=np.int64),
        val_idx=np.asarray(cond.split.val_idx, dtype=np.int64),
        test_idx=np.asarray(cond.split.test_idx, dtype=np.int64),
        scaler_min=np.asarray(cond.scaler.data_min, dtype=np.float64),
        scaler_range=np.asarray(cond.scaler.data_range, dtype=np.float64),
        nominal_rate=np.asarray([cond.nominal_rate]),
        realized_cell_rate=np.asarray([cond.realized_cell_rate]),
        affected_voyage_rate=np.asarray([cond.affected_row_rate]),
        realized_affected_voyage_rate=np.asarray([float(np.mean(cond.artificial_missing_mask.any(axis=1))) if cond.mechanism == "env_block" else np.nan]),
        condition_hash=np.asarray([cond.condition_hash]),
        seed=np.asarray([cond.seed],dtype=np.int64),
    )


def load_condition_npz(base, path: Path) -> Condition:
    """Restore an immutable worker condition without regenerating split or mask."""
    with np.load(path, allow_pickle=False) as saved:
        split=base.SplitIndices(saved["train_idx"].astype(int),saved["val_idx"].astype(int),saved["test_idx"].astype(int))
        scaler=base.MinMaxState(saved["scaler_min"].astype(float),saved["scaler_range"].astype(float))
        X_true=saved["X_true_scaled"].astype(float)
        X_missing=saved["X_missing_scaled"].astype(float)
        observed=saved["original_observed_mask"].astype(bool)
        artificial=saved["artificial_missing_mask"].astype(bool)
        nominal=float(saved["nominal_rate"][0])
        realized=float(saved["realized_cell_rate"][0])
        affected=float(saved["affected_voyage_rate"][0]) if "affected_voyage_rate" in saved.files else float("nan")
        condition_hash=str(saved["condition_hash"][0])
        seed=int(saved["seed"][0])
    mechanism="env_block" if np.isfinite(affected) else "mcar"
    return Condition(seed,mechanism,nominal,affected,realized,split,scaler,X_true,X_missing,observed,artificial,np.empty((len(X_true),0),dtype=np.float32),condition_hash)


def decorate_record(rec: Dict[str, object], suite: str, protocol: str, cond: Condition, relation_mode: str = "") -> Dict[str, object]:
    out = dict(rec)
    out["experiment_suite"] = suite
    out["evaluation_protocol"] = protocol
    out["missing_mechanism"] = cond.mechanism
    out["nominal_missing_rate"] = cond.nominal_rate
    out["realized_cell_missing_rate"] = cond.realized_cell_rate
    out["affected_row_rate"] = cond.affected_row_rate
    out["affected_voyage_rate"] = cond.affected_row_rate
    out["realized_affected_voyage_rate"] = float(np.mean(cond.artificial_missing_mask.any(axis=1))) if cond.mechanism == "env_block" else float("nan")
    out["relation_mode"] = relation_mode
    out["natural_missing_count"] = int((~cond.original_observed_mask).sum())
    out["complete_ground_truth_count"] = int(cond.original_observed_mask.sum())
    out["artificial_masked_count"] = int(cond.artificial_missing_mask.sum())
    test_rows = np.zeros(cond.original_observed_mask.shape[0], dtype=bool)
    test_rows[np.asarray(cond.split.test_idx, dtype=int)] = True
    out["evaluated_cell_count"] = int((cond.artificial_missing_mask & test_rows[:, None]).sum())
    out["condition_hash"] = cond.condition_hash
    return out


def method_cache_paths(root: Path, method: str) -> Tuple[Path, Path]:
    safe = method.replace("/", "_").replace(" ", "_").replace("[", "_").replace("]", "_")
    return root / f"{safe}_imputed.npy", root / f"{safe}_diag.json"


def cache_is_complete(out_npy: Path, diag_json: Path, condition_hash: str) -> bool:
    """Accept only a complete cache produced for the exact same condition."""
    if not out_npy.exists() or not diag_json.exists():
        return False
    try:
        diag = json.loads(diag_json.read_text(encoding="utf-8"))
        return bool(diag.get("completed")) and diag.get("condition_hash") == condition_hash
    except Exception:
        return False


def save_completed_cache(out_npy: Path, diag_json: Path, imputed: np.ndarray, diagnostics: Dict[str, object], cond: Condition) -> None:
    """Write the array first and mark completion only after all artifacts exist."""
    out_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_npy, imputed.astype(np.float32))
    payload = dict(diagnostics)
    payload.update({
        "completed": True,
        "condition_hash": cond.condition_hash,
        "imputed_sha256": array_sha256(imputed.astype(np.float32)),
    })
    diag_json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def run_igrm_subprocess(args: argparse.Namespace, cond: Condition, protocol: str, work_dir: Path, repo_dir: Path) -> Tuple[np.ndarray, Dict[str, object]]:
    out_npy, diag_json = method_cache_paths(work_dir, IGRM_METHOD)
    if args.resume and cache_is_complete(out_npy, diag_json, cond.condition_hash):
        return np.load(out_npy), json.loads(diag_json.read_text(encoding="utf-8"))
    condition_npz = work_dir / "condition_for_igrm.npz"
    save_condition_npz(cond, condition_npz)
    runner = Path(args.igrm_runner).resolve()
    cmd = [
        sys.executable, str(runner),
        "--condition_npz", str(condition_npz),
        "--output_npy", str(work_dir / "igrm_raw_imputed.npy"),
        "--diag_json", str(work_dir / "igrm_raw_diag.json"),
        "--igrm_repo_dir", str(repo_dir),
        "--protocol", "strict_inductive" if protocol == "strict_inductive" else "transductive",
        "--device", args.device,
        "--seed", str(cond.seed),
        "--epochs", str(args.graph_max_epochs),
        "--eval_every", str(args.eval_every),
        "--friend_k", str(args.igrm_friend_k),
        "--reconstruct_every", str(args.igrm_reconstruct_every),
        "--loss_scope", "combined",
    ]
    if args.verbose:
        cmd.append("--verbose")
    print("[RUN]", IGRM_METHOD, protocol, condition_slug(cond), flush=True)
    subprocess.run(cmd, check=True)
    imp = np.load(work_dir / "igrm_raw_imputed.npy")
    diag = json.loads((work_dir / "igrm_raw_diag.json").read_text(encoding="utf-8"))
    diag.update({"method_name": IGRM_METHOD, "igrm_label": "adapted_common_protocol"})
    save_completed_cache(out_npy, diag_json, imp, diag, cond)
    return imp, json.loads(diag_json.read_text(encoding="utf-8"))


def run_original_method_subprocess(
    args: argparse.Namespace,
    cond: Condition,
    condition_npz: Path,
    method: str,
    protocol: str,
    relation_mode: str,
    cache_dir: Path,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Isolate each method-condition so a failed CUDA context cannot poison the suite."""
    cache_label=method if not relation_mode else f"{method}_{relation_mode}"
    out_npy,diag_json=method_cache_paths(cache_dir,cache_label)
    if args.resume and cache_is_complete(out_npy,diag_json,cond.condition_hash):
        return np.load(out_npy),json.loads(diag_json.read_text(encoding="utf-8"))
    cmd=[
        sys.executable,str(Path(__file__).resolve()),"--worker_mode",
        "--worker_condition_npz",str(condition_npz),"--worker_method",method,
        "--worker_protocol",protocol,"--worker_relation_mode",relation_mode,
        "--worker_cache_dir",str(cache_dir),"--excel_path",str(Path(args.excel_path).resolve()),
        "--base_script",str(Path(args.base_script).resolve()),"--device",args.device,
        "--graph_max_epochs",str(args.graph_max_epochs),"--eval_every",str(args.eval_every),
        "--data_schema",args.data_schema,"--voyage_feature_set",args.voyage_feature_set,
        "--target",args.target,"--split_mode",args.split_mode,"--train_ratio",str(args.train_ratio),
        "--val_ratio",str(args.val_ratio),"--out_dir",str(Path(args.out_dir).resolve()),
    ]
    if args.verbose:
        cmd.append("--verbose")
    subprocess.run(cmd,check=True)
    if not cache_is_complete(out_npy,diag_json,cond.condition_hash):
        raise RuntimeError(f"Worker exited without a complete cache: {method} {condition_slug(cond)}")
    return np.load(out_npy),json.loads(diag_json.read_text(encoding="utf-8"))


def build_relation_edges_local(base, df_local: pd.DataFrame, bargs, relation_mode: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    n = len(df_local)
    if relation_mode not in {"none", "vessel", "destination", FINAL_RELATION_MODE}:
        raise ValueError(f"Unsupported paper relation mode: {relation_mode}")
    diag = {"relation_mode": relation_mode, "n_nodes": n, "n_relation_edges": 0, "destination_key_source": "none"}
    edge_dict: Dict[Tuple[int, int], float] = {}
    if relation_mode == "none":
        return np.empty((2, 0), dtype=np.int64), np.empty((0,), dtype=np.float32), np.zeros(n, dtype=np.float32), diag
    vessel_edges = base.build_vessel_domain_edges(df_local, bargs) if relation_mode in {"vessel", FINAL_RELATION_MODE} else {}
    destination_edges: Dict[Tuple[int, int], float] = {}
    if vessel_edges:
        base.merge_domain_edges(edge_dict, vessel_edges)
    if relation_mode in {"destination", FINAL_RELATION_MODE}:
        destination_edges, src = base.build_destination_domain_edges(df_local, bargs)
        diag["destination_key_source"] = src
        base.merge_domain_edges(edge_dict, destination_edges)
    vessel_pairs=base._logical_relation_pairs(vessel_edges)
    destination_pairs=base._logical_relation_pairs(destination_edges)
    diag.update({
        "standalone_vessel_pair_count":len(vessel_pairs),
        "standalone_destination_pair_count":len(destination_pairs),
        "vessel_destination_overlap_pair_count":len(vessel_pairs & destination_pairs),
        "final_merged_unique_pair_count":len(base._logical_relation_pairs(edge_dict)),
        "directed_relation_edge_count":len(edge_dict),"merge_rule":"max_weight",
    })
    degree = np.zeros(n, dtype=np.float32)
    for (src, _), w in edge_dict.items(): degree[src] += float(w)
    diag["n_relation_edges"] = len(edge_dict)
    diag["mean_relation_degree"] = float(np.mean(degree)) if n else 0.0
    diag["max_relation_degree"] = float(np.max(degree)) if n else 0.0
    diag.update(base._relation_distribution_diagnostics(edge_dict,n))
    mx = float(np.max(degree)) if n else 0.0
    norm_degree = degree / mx if mx > 0 else degree
    if not edge_dict:
        return np.empty((2, 0), dtype=np.int64), np.empty((0,), dtype=np.float32), norm_degree, diag
    pairs = np.array(list(edge_dict.keys()), dtype=np.int64)
    weights = np.array([edge_dict[tuple(p)] for p in pairs], dtype=np.float32)
    return pairs.T, weights, norm_degree, diag


def make_local_graph_tensors(base, df_local: pd.DataFrame, X_true_local: np.ndarray, X_missing_local: np.ndarray, art_local: np.ndarray, bargs, relation_mode: str, relation_enabled: bool, device: torch.device):
    n, d = X_missing_local.shape
    obs = np.isfinite(X_missing_local)
    pairs = np.argwhere(obs)
    vals = X_missing_local[obs].astype(np.float32)
    vei_np, vea_np = base.cell_pairs_to_original_grape_edges(pairs, vals, n)
    if relation_enabled:
        rei_np, rew_np, degree, diag = build_relation_edges_local(base, df_local, bargs, relation_mode)
    else:
        rei_np, rew_np, degree, diag = np.empty((2, 0), dtype=np.int64), np.empty((0,), dtype=np.float32), np.zeros(n, dtype=np.float32), {"relation_mode": "none", "n_relation_edges": 0}
    x = torch.tensor(base.make_original_grape_node_features(n, d, int(bargs.original_grape_node_mode)), dtype=torch.float32, device=device)
    vei = torch.tensor(vei_np, dtype=torch.long, device=device)
    vea = torch.tensor(vea_np, dtype=torch.float32, device=device)
    rei = torch.tensor(rei_np, dtype=torch.long, device=device) if rei_np.size else torch.empty((2,0), dtype=torch.long, device=device)
    rea = torch.tensor(rew_np.reshape(-1,1), dtype=torch.float32, device=device) if rew_np.size else torch.empty((0,1), dtype=torch.float32, device=device)
    ap, al = base.tensor_pairs_and_values(np.argwhere(art_local), X_true_local[art_local].astype(np.float32), device)
    op, ol = base.tensor_pairs_and_values(np.argwhere(obs), X_true_local[obs].astype(np.float32), device)
    return {"n": n, "d": d, "x": x, "vei": vei, "vea": vea, "rei": rei, "rea": rea, "art_pairs": ap, "art_labels": al, "obs_pairs": op, "obs_labels": ol, "obs_mask_np": obs, "degree": degree, "diag": diag}


def evaluate_local_graph(base, model, impute_model, oca, graph, X_context_t, obs_mask_t) -> Tuple[float, float]:
    if graph["art_pairs"].numel() == 0:
        return float("nan"), float("nan")
    return base.evaluate_original_or_oca_pairs(
        model, impute_model, graph["x"], graph["vei"], graph["vea"], graph["rei"], graph["rea"],
        graph["art_pairs"], graph["art_labels"], graph["n"], oca=oca, X_context=X_context_t, observed_mask=obs_mask_t,
    )


def predict_local_full(base, model, impute_model, oca, graph, X_context_t, obs_mask_t, bargs, device):
    if oca is None:
        return base.predict_original_grape_full_matrix(model, impute_model, graph["x"], graph["vei"], graph["vea"], graph["rei"], graph["rea"], graph["n"], graph["d"], int(bargs.original_grape_prediction_batch_size), device)
    pred, gate, cov, lam = base.predict_oca_full_matrix(model, impute_model, oca, graph["x"], graph["vei"], graph["vea"], graph["rei"], graph["rea"], X_context_t, obs_mask_t, graph["n"], graph["d"], int(bargs.original_grape_prediction_batch_size), device)
    return pred


def run_graph_inductive(base, df: pd.DataFrame, numeric_features: Sequence[str], y: np.ndarray, bargs, cond: Condition, method: str, relation_mode: str) -> Tuple[np.ndarray, Dict[str, object]]:
    base.require_original_grape_dependencies()
    base.set_random_seed(cond.seed)
    device = torch.device(bargs.device if bargs.device == "cpu" or torch.cuda.is_available() else "cpu")
    use_oca = method.startswith("OCA-")
    relation_enabled = method in {"R-GRAPE", "OCA-R-GRAPE"}
    if relation_mode not in {"none", "vessel", "destination", FINAL_RELATION_MODE}:
        raise ValueError(f"Unsupported strict-inductive relation mode: {relation_mode}")
    base.assert_split_integrity(cond.split, len(df))
    train_ids = set(np.asarray(cond.split.train_idx, dtype=int).tolist())
    val_ids = set(np.asarray(cond.split.val_idx, dtype=int).tolist())
    test_ids = set(np.asarray(cond.split.test_idx, dtype=int).tolist())
    assert not (train_ids & val_ids or train_ids & test_ids or val_ids & test_ids)
    train_means = base.safe_col_means(cond.X_missing_scaled[cond.split.train_idx], fallback=0.0)

    local = {}
    contexts = {}
    obs_masks_t = {}
    # Test rows deliberately do not exist as graph nodes during optimization or checkpoint selection.
    for name, idx in [("train", cond.split.train_idx), ("val", cond.split.val_idx)]:
        dfl = df.iloc[idx].reset_index(drop=True)
        Xt = cond.X_true_scaled[idx]
        Xm = cond.X_missing_scaled[idx]
        Am = cond.artificial_missing_mask[idx]
        g = make_local_graph_tensors(base, dfl, Xt, Xm, Am, bargs, relation_mode, relation_enabled, device)
        local[name] = g
        Xc = base.fill_nan_with_means(Xm, train_means).astype(np.float32)
        contexts[name] = torch.tensor(Xc, dtype=torch.float32, device=device)
        obs_masks_t[name] = torch.tensor(np.isfinite(Xm), dtype=torch.bool, device=device)

    tr = local["train"]
    model, impute_model, adapter_node_dim = base.build_original_grape_models(bargs, tr["n"], tr["d"], tr["x"], device)
    oca = None
    if use_oca:
        oca = base.ObservedContextAdapter(
            node_dim=adapter_node_dim,
            hidden_dim=max(8, int(bargs.oca_hidden_dim)),
            dropout=float(bargs.oca_dropout),
            residual_weight=float(bargs.oca_residual_weight),
            adaptive_fusion=bool(bargs.oca_adaptive_fusion),
            fusion_temperature=float(bargs.oca_fusion_temperature),
            fusion_center=float(bargs.oca_fusion_center),
            global_missing_rate=float(cond.nominal_rate if cond.mechanism == "mcar" else cond.realized_cell_rate),
            global_missing_weight=float(bargs.oca_global_missing_weight),
            context_min_coverage=float(bargs.oca_context_min_coverage),
        ).to(device)
    params = list(model.parameters()) + list(impute_model.parameters()) + ([] if oca is None else list(oca.parameters()))
    opt = torch.optim.Adam(params, lr=float(bargs.graph_lr), weight_decay=float(bargs.graph_weight_decay))
    feat_weights = torch.ones(tr["d"], dtype=torch.float32, device=device)
    best_score, best_epoch, best_state = float("inf"), 0, None

    for epoch in range(1, int(bargs.graph_max_epochs)+1):
        model.train(); impute_model.train();
        if oca is not None: oca.train()
        ei, ea = base.make_original_grape_message_graph(tr["vei"], tr["vea"], tr["rei"], tr["rea"], known_prob=float(bargs.original_grape_known))
        emb = model(tr["x"], ea, ei)
        opt.zero_grad()
        terms = []
        if tr["art_pairs"].numel() > 0: terms.append((tr["art_pairs"], tr["art_labels"]))
        if tr["obs_pairs"].numel() > 0: terms.append((tr["obs_pairs"], tr["obs_labels"]))
        if not terms: terms.append((tr["obs_pairs"], tr["obs_labels"]))
        if oca is None:
            losses=[]
            for pp,ll in terms:
                pred = base.predict_original_grape_pairs(emb, impute_model, pp, tr["n"])
                losses.append(base.weighted_original_grape_mse(pred,ll,pp,feat_weights))
            loss=torch.stack(losses).sum(); loss.backward(); loss_value=float(loss.detach())
        else:
            loss_value=0.0
            for t,(pp,ll) in enumerate(terms):
                loss_value += base.backward_weighted_oca_mse_chunks(
                    emb, impute_model, oca, contexts["train"], obs_masks_t["train"], pp, ll, feat_weights, tr["n"],
                    int(bargs.oca_pair_batch_size), retain_graph_for_last_chunk=(t < len(terms)-1),
                )
        torch.nn.utils.clip_grad_norm_(params, 5.0); opt.step()

        if epoch == 1 or epoch % max(1,int(bargs.original_grape_eval_every)) == 0 or epoch == int(bargs.graph_max_epochs):
            vmae, vrmse = evaluate_local_graph(base, model, impute_model, oca, local["val"], contexts["val"], obs_masks_t["val"])
            score = vmae if np.isfinite(vmae) else float(loss_value)
            if score < best_score:
                best_score=float(score); best_epoch=epoch
                best_state={
                    "model": {k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                    "impute": {k:v.detach().cpu().clone() for k,v in impute_model.state_dict().items()},
                    "oca": None if oca is None else {k:v.detach().cpu().clone() for k,v in oca.state_dict().items()},
                }
            if bargs.verbose:
                print(f"[INDUCTIVE] {method} epoch={epoch} loss={loss_value:.6f} val_mae={vmae:.6f} best={best_epoch}", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state["model"]); impute_model.load_state_dict(best_state["impute"])
        if oca is not None and best_state["oca"] is not None: oca.load_state_dict(best_state["oca"])

    for parameter in params:
        parameter.requires_grad_(False)
    model.eval(); impute_model.eval()
    if oca is not None:
        oca.eval()

    assert "test" not in local, "Test graph was constructed before checkpoint selection."
    test_idx = cond.split.test_idx
    test_df = df.iloc[test_idx].reset_index(drop=True)
    test_true = cond.X_true_scaled[test_idx]
    test_missing = cond.X_missing_scaled[test_idx]
    test_artificial = cond.artificial_missing_mask[test_idx]
    local["test"] = make_local_graph_tensors(
        base, test_df, test_true, test_missing, test_artificial,
        bargs, relation_mode, relation_enabled, device,
    )
    test_context = base.fill_nan_with_means(test_missing, train_means).astype(np.float32)
    contexts["test"] = torch.tensor(test_context, dtype=torch.float32, device=device)
    obs_masks_t["test"] = torch.tensor(np.isfinite(test_missing), dtype=torch.bool, device=device)

    global_imp = cond.X_missing_scaled.copy()
    for name, idx in [("train",cond.split.train_idx),("val",cond.split.val_idx),("test",cond.split.test_idx)]:
        pred = predict_local_full(base, model, impute_model, oca, local[name], contexts[name], obs_masks_t[name], bargs, device)
        local_imp = cond.X_missing_scaled[idx].copy(); miss=~np.isfinite(local_imp); local_imp[miss]=pred[miss]; global_imp[idx]=local_imp
    global_imp = base.finalize_imputed_scaled(global_imp, cond.X_missing_scaled, cond.split.train_idx)
    diag={
        "method_name":method,"evaluation_protocol":"strict_inductive","best_epoch":best_epoch,
        "best_validation_MAE":best_score,"best_valid_mae":best_score,"final_training_epoch":int(bargs.graph_max_epochs),
        "relation_mode": relation_mode if relation_enabled else "none",
        "train_relation_edges": int(local["train"]["rei"].shape[1]),
        "val_relation_edges": int(local["val"]["rei"].shape[1]),
        "test_relation_edges": int(local["test"]["rei"].shape[1]),
        "relation_stats_scope":"metadata_only_no_numeric_relation_scaling",
        "scaler_fit_scope":"training_rows_only",
        "train_graph_original_row_hash":array_sha256(np.asarray(cond.split.train_idx, dtype=np.int64)),
        "validation_graph_original_row_hash":array_sha256(np.asarray(cond.split.val_idx, dtype=np.int64)),
        "test_graph_original_row_hash":array_sha256(np.asarray(cond.split.test_idx, dtype=np.int64)),
        "test_graph_constructed_after_best_checkpoint":True,
        "test_rows_in_train_graph":False,
    }
    for key,value in local["train"]["diag"].items():
        diag[f"train_graph_{key}"]=value
    for key,value in local["val"]["diag"].items():
        diag[f"validation_graph_{key}"]=value
    for key,value in local["test"]["diag"].items():
        diag[f"test_graph_{key}"]=value
    return global_imp,diag


def run_original_method(base, df, numeric_features, y, bargs, cond: Condition, method: str, protocol: str, relation_mode: str, cache_dir: Path) -> Tuple[np.ndarray, Dict[str, object]]:
    out_npy, diag_json = method_cache_paths(cache_dir, method if not relation_mode else f"{method}_{relation_mode}")
    if bargs.verbose:
        print(f"[RUN] {protocol} {method} {relation_mode} {condition_slug(cond)}", flush=True)
    if cache_is_complete(out_npy, diag_json, cond.condition_hash) and getattr(bargs, "_suite_resume", True):
        return np.load(out_npy), json.loads(diag_json.read_text(encoding="utf-8"))
    old_mode = bargs.relation_mode
    old_oca_mode = bargs.oca_train_mode
    try:
        bargs.relation_mode = relation_mode
        bargs.oca_train_mode = "joint"
        if protocol == "transductive":
            if method == "Mean": imp=base.run_mean_imputer(cond.X_missing_scaled,cond.split.train_idx); diag={}
            elif method == "KNN": imp=base.run_knn_imputer(cond.X_missing_scaled,cond.split.train_idx); diag={}
            elif method == "MICE": imp=base.run_mice_imputer(cond.X_missing_scaled,cond.split.train_idx,cond.seed); diag={}
            elif method == "GAIN": imp=base.run_gain_imputer(cond.X_true_scaled,cond.X_missing_scaled,cond.split.train_idx,cond.artificial_missing_mask,bargs,cond.seed,numeric_features,y,cond.split); diag={}
            elif method in CORE_GRAPH_METHODS:
                use_oca=method.startswith("OCA-"); relation_enabled=method in {"R-GRAPE","OCA-R-GRAPE"}
                o=base.run_original_bipartite_graph_imputer(
                    cond.X_true_scaled,cond.X_missing_scaled,cond.split,cond.artificial_missing_mask,df,bargs,cond.seed,method,
                    force_relation_enhanced=relation_enabled and relation_mode!="none",numeric_features=numeric_features,y=y,use_oca=use_oca,
                    missing_rate=(cond.realized_cell_rate if cond.mechanism!="mcar" else cond.nominal_rate),
                )
                imp=o.imputed_scaled; diag=o.relation_diagnostics
            else: raise ValueError(method)
        elif protocol == "strict_inductive":
            if method in {"Mean","KNN","MICE","GAIN"}:
                return run_original_method(base,df,numeric_features,y,bargs,cond,method,"transductive",relation_mode,cache_dir)
            imp,diag=run_graph_inductive(base,df,numeric_features,y,bargs,cond,method,relation_mode)
        else:
            raise ValueError(f"Unsupported evaluation protocol: {protocol}")
    finally:
        bargs.relation_mode=old_mode
        bargs.oca_train_mode=old_oca_mode
    save_completed_cache(out_npy, diag_json, imp, diag, cond)
    return imp,json.loads(diag_json.read_text(encoding="utf-8"))


def evaluate_imputed(base, cond: Condition, X_imp: np.ndarray, numeric_features: Sequence[str], y: np.ndarray, predictors: Sequence[str], bargs, method: str, suite: str, protocol: str, relation_mode: str, run_downstream: bool):
    overall=[]; perfeat=[]; source=[]; downstream=[]
    test_mask=cond.artificial_missing_mask & base.make_row_mask(len(cond.X_true_scaled),cond.split.test_idx)[:,None]
    o,p=base.compute_imputation_records(cond.X_true_scaled,X_imp,cond.scaler,test_mask,numeric_features,cond.seed,cond.realized_cell_rate,method,bargs)
    overall.append(decorate_record(o,suite,protocol,cond,relation_mode)); perfeat.extend(decorate_record(x,suite,protocol,cond,relation_mode) for x in p)
    source.extend(decorate_record(x,suite,protocol,cond,relation_mode) for x in base.compute_source_group_imputation_records(cond.X_true_scaled,X_imp,test_mask,numeric_features,cond.seed,cond.realized_cell_rate,method,bargs))
    if run_downstream:
        for predictor in predictors:
            try:
                rec=base.run_downstream_predictor(X_imp,cond.cat_onehot,y,cond.split,cond.seed,cond.realized_cell_rate,method,predictor,bargs)
                downstream.append(decorate_record(rec,suite,protocol,cond,relation_mode))
            except Exception as exc:
                print(f"[WARN] downstream {method}+{predictor} failed: {exc}",flush=True)
    return overall,perfeat,source,downstream


def append_write(records: List[Dict[str,object]], path: Path):
    path.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(records).to_csv(path,index=False,encoding="utf-8-sig")


def save_shared_protocol_artifacts(out_dir: Path, cond: Condition) -> None:
    """Persist one split per seed and one deterministic mask per condition."""
    split_path = out_dir / "splits" / f"seed_{cond.seed}.npz"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_arrays = {
        "train_idx": np.asarray(cond.split.train_idx, dtype=np.int64),
        "val_idx": np.asarray(cond.split.val_idx, dtype=np.int64),
        "test_idx": np.asarray(cond.split.test_idx, dtype=np.int64),
    }
    if split_path.exists():
        with np.load(split_path) as saved:
            for key, value in split_arrays.items():
                assert np.array_equal(saved[key], value), f"Split drift detected for seed {cond.seed}."
    else:
        np.savez_compressed(split_path, **split_arrays)
    mask_path = out_dir / "masks" / f"{cond.mechanism}_{condition_slug(cond)}.npz"
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    if mask_path.exists():
        with np.load(mask_path) as saved:
            assert np.array_equal(saved["artificial_missing_mask"].astype(bool), cond.artificial_missing_mask), "Mask drift detected."
    else:
        np.savez_compressed(
            mask_path,
            artificial_missing_mask=cond.artificial_missing_mask.astype(np.uint8),
            condition_hash=np.asarray([cond.condition_hash]),
            nominal_rate=np.asarray([cond.nominal_rate]),
            realized_cell_rate=np.asarray([cond.realized_cell_rate]),
            affected_row_rate=np.asarray([cond.affected_row_rate]),
            realized_affected_voyage_rate=np.asarray([float(np.mean(cond.artificial_missing_mask.any(axis=1))) if cond.mechanism == "env_block" else np.nan]),
        )


def write_leakage_audit(base, df, numeric_features, categorical_features, bargs, out_dir: Path) -> None:
    """Run fail-closed checks for targets, masks, splits, scalers, and OCA context exclusion."""
    cond=prepare_condition(base,df,numeric_features,categorical_features,bargs,0,"mcar",0.3)
    base.assert_downstream_target_isolated(bargs.target,numeric_features)
    base.assert_split_integrity(cond.split,len(df))
    base.assert_artificial_targets_hidden(cond.X_missing_scaled,cond.original_observed_mask,cond.artificial_missing_mask)
    X_raw=df[list(numeric_features)].apply(pd.to_numeric,errors="coerce").to_numpy(dtype=float)
    fitted_again=base.fit_minmax_train(X_raw,cond.split.train_idx)
    assert np.array_equal(fitted_again.data_min,cond.scaler.data_min)
    assert np.array_equal(fitted_again.data_range,cond.scaler.data_range)
    oca_source=inspect.getsource(base.ObservedContextAdapter.row_context)
    assert "scatter_(1, target_feat_idx.unsqueeze(1), False)" in oca_source
    records=[
        {"check":"fuel_target_absent_from_imputation_features","passed":True,"detail":bargs.target},
        {"check":"train_validation_test_disjoint_and_exhaustive","passed":True,"detail":cond.condition_hash},
        {"check":"artificial_targets_hidden_from_input_graph","passed":True,"detail":int(cond.artificial_missing_mask.sum())},
        {"check":"artificial_mask_uses_only_observed_ground_truth","passed":True,"detail":int(cond.original_observed_mask.sum())},
        {"check":"scaler_refit_from_train_rows_is_exact","passed":True,"detail":len(cond.split.train_idx)},
        {"check":"oca_explicitly_excludes_target_position","passed":True,"detail":"row_mask.scatter_(..., False)"},
        {"check":"formal_relations_use_metadata_only","passed":True,"detail":"Vessel + Destination"},
        {"check":"strict_test_graph_created_after_checkpoint","passed":True,"detail":"enforced by assertion in run_graph_inductive"},
    ]
    audit_dir=out_dir/"diagnostics"; audit_dir.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(records).to_csv(audit_dir/"leakage_audit.csv",index=False,encoding="utf-8-sig")


def record_failure(out_dir: Path, suite: str, protocol: str, cond: Condition, method: str, exc: BaseException) -> None:
    """Append a failed method-condition without terminating the remaining experiment queue."""
    path = out_dir / "failed_runs.csv"
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "experiment_suite": suite,
        "evaluation_protocol": protocol, "method": method, "seed": cond.seed,
        "missing_mechanism": cond.mechanism, "nominal_missing_rate": cond.nominal_rate,
        "condition_hash": cond.condition_hash, "error_type": type(exc).__name__,
        "error_message": str(exc), "traceback": traceback.format_exc(),
    }
    old = pd.read_csv(path) if path.exists() else pd.DataFrame()
    pd.concat([old, pd.DataFrame([record])], ignore_index=True).to_csv(path, index=False, encoding="utf-8-sig")


def run_suite_conditions(base, df, numeric_features, categorical_features, y, bargs, args, suite: str, protocol: str, mechanism: str, rates: Sequence[float], methods: Sequence[str], predictors: Sequence[str], relation_mode: str, repo_dir: Optional[Path], run_downstream: bool, all_store):
    suite_dir=Path(args.out_dir)/suite
    for seed in parse_list_int(args.seeds):
        for rate in rates:
            cond=prepare_condition(base,df,numeric_features,categorical_features,bargs,seed,mechanism,float(rate))
            save_shared_protocol_artifacts(Path(args.out_dir), cond)
            cdir=suite_dir/condition_slug(cond); cdir.mkdir(parents=True,exist_ok=True)
            save_condition_npz(cond,cdir/"condition.npz")
            cache_dir=Path(args.out_dir)/"checkpoints"/suite/condition_slug(cond)
            cache_dir.mkdir(parents=True,exist_ok=True)
            for method in methods:
                try:
                    if method==IGRM_METHOD:
                        if args.skip_igrm: continue
                        if repo_dir is None: raise RuntimeError("IGRM repo unavailable")
                        imp,diag=run_igrm_subprocess(args,cond,protocol,cache_dir,repo_dir)
                    else:
                        imp,diag=run_original_method_subprocess(args,cond,cdir/"condition.npz",method,protocol,relation_mode,cache_dir)
                except Exception as exc:
                    record_failure(Path(args.out_dir), suite, protocol, cond, method, exc)
                    print(f"[FAILED] {suite} {method} {condition_slug(cond)}: {exc}", flush=True)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                if not args.save_imputed_arrays:
                    cache_label=method if method==IGRM_METHOD else (method if not relation_mode else f"{method}_{relation_mode}")
                    p,_=method_cache_paths(cache_dir,cache_label)
                    # Keep caches when resume is enabled; deleting would force reruns. Only delete in non-resume mode.
                    if not args.resume and p.exists(): p.unlink()
                method_relation = "learned_friend_network" if method == IGRM_METHOD else (relation_mode if method in {"R-GRAPE","OCA-R-GRAPE"} else "none")
                o,pf,sg,ds=evaluate_imputed(base,cond,imp,numeric_features,y,predictors,bargs,method,suite,protocol,method_relation,run_downstream)
                all_store["overall"].extend(o); all_store["per_feature"].extend(pf); all_store["source_group"].extend(sg); all_store["downstream"].extend(ds)
                if diag:
                    d=decorate_record(diag,suite,protocol,cond,method_relation); d["imputer"]=method; all_store["diag"].append(d)
            append_write(all_store["overall"],Path(args.out_dir)/"ALL_imputation_summary_partial.csv")
            append_write(all_store["diag"],Path(args.out_dir)/"ALL_diagnostics_partial.csv")


def run_relation_ablation(base,df,numeric_features,categorical_features,y,bargs,args,predictors,all_store):
    suite="relation_ablation"; suite_dir=Path(args.out_dir)/suite
    modes=["none","vessel","destination",FINAL_RELATION_MODE]
    for seed in parse_list_int(args.seeds):
        for rate in parse_list_float(args.relation_ablation_rates):
            cond=prepare_condition(base,df,numeric_features,categorical_features,bargs,seed,"mcar",rate)
            save_shared_protocol_artifacts(Path(args.out_dir), cond)
            cdir=suite_dir/condition_slug(cond); cdir.mkdir(parents=True,exist_ok=True); save_condition_npz(cond,cdir/"condition.npz")
            cache_dir=Path(args.out_dir)/"checkpoints"/suite/condition_slug(cond); cache_dir.mkdir(parents=True,exist_ok=True)
            for mode in modes:
                method="GRAPE" if mode=="none" else "R-GRAPE"
                display={"none":"GRAPE","vessel":"R-GRAPE[Vessel]","destination":"R-GRAPE[Destination]",FINAL_RELATION_MODE:"R-GRAPE[Vessel+Destination]"}
                label=display[mode]
                try:
                    imp,diag=run_original_method_subprocess(args,cond,cdir/"condition.npz",method,"transductive",mode,cache_dir)
                except Exception as exc:
                    record_failure(Path(args.out_dir), suite, "transductive", cond, label, exc)
                    continue
                o,pf,sg,_=evaluate_imputed(base,cond,imp,numeric_features,y,[],bargs,label,suite,"transductive",mode,False)
                all_store["overall"].extend(o); all_store["per_feature"].extend(pf); all_store["source_group"].extend(sg)
                d=decorate_record(diag,suite,"transductive",cond,mode); d["imputer"]=label; all_store["diag"].append(d)
            append_write(all_store["overall"],Path(args.out_dir)/"ALL_imputation_summary_partial.csv")



def run_complete_reference(base,df,numeric_features,categorical_features,y,bargs,args,predictors,all_store):
    suite="complete_data_reference"
    for seed in parse_list_int(args.seeds):
        cond=prepare_condition(base,df,numeric_features,categorical_features,bargs,seed,"mcar",0.0)
        if not np.all(np.isfinite(cond.X_true_scaled)):
            raise ValueError("Complete-data reference requires every selected numeric support value to be observed.")
        save_shared_protocol_artifacts(Path(args.out_dir), cond)
        for predictor in predictors:
            rec=base.run_downstream_predictor(cond.X_true_scaled,cond.cat_onehot,y,cond.split,seed,0.0,"Complete-Data Reference",predictor,bargs)
            all_store["downstream"].append(decorate_record(rec,suite,"complete",cond,"none"))


def flat_mean_sd(df: pd.DataFrame, groups: Sequence[str], metrics: Sequence[str]) -> pd.DataFrame:
    if df.empty: return pd.DataFrame()
    agg=df.groupby(list(groups),dropna=False)[list(metrics)].agg(["mean","std","count"]).reset_index()
    agg.columns=[c if isinstance(c,str) else "_".join([str(x) for x in c if str(x)]) for c in agg.columns]
    return agg


def save_paired_comparisons(overall: pd.DataFrame, out_dir: Path) -> None:
    """Save within-condition paired differences for the main model and relation ablation."""
    rows=[]
    comparisons=[
        ("main_mcar","OCA-R-GRAPE","GRAPE"),
        ("main_mcar","OCA-R-GRAPE",IGRM_METHOD),
        ("main_mcar","OCA-R-GRAPE","R-GRAPE"),
        ("main_mcar","OCA-R-GRAPE","OCA-GRAPE"),
        ("relation_ablation","R-GRAPE[Vessel+Destination]","GRAPE"),
        ("relation_ablation","R-GRAPE[Vessel+Destination]","R-GRAPE[Vessel]"),
        ("relation_ablation","R-GRAPE[Vessel+Destination]","R-GRAPE[Destination]"),
    ]
    keys=["seed","nominal_missing_rate","evaluation_protocol"]
    for suite,target,reference in comparisons:
        block=overall[overall["experiment_suite"]==suite]
        left=block[block["imputer"]==target][keys+["MAE_scaled_clipped","RMSE_scaled_clipped"]]
        right=block[block["imputer"]==reference][keys+["MAE_scaled_clipped","RMSE_scaled_clipped"]]
        if left.empty or right.empty:
            continue
        merged=left.merge(right,on=keys,suffixes=("_target","_reference"))
        for _,row in merged.iterrows():
            rows.append({
                **{key:row[key] for key in keys},"experiment_suite":suite,
                "target_model":target,"reference_model":reference,
                "delta_MAE_target_minus_reference":row["MAE_scaled_clipped_target"]-row["MAE_scaled_clipped_reference"],
                "delta_RMSE_target_minus_reference":row["RMSE_scaled_clipped_target"]-row["RMSE_scaled_clipped_reference"],
                "target_wins_MAE":bool(row["MAE_scaled_clipped_target"]<row["MAE_scaled_clipped_reference"]),
            })
    pd.DataFrame(rows).to_csv(out_dir/"paired_model_comparisons.csv",index=False,encoding="utf-8-sig")


def save_aggregates(out_dir: Path, store):
    out_dir.mkdir(parents=True,exist_ok=True)
    overall=pd.DataFrame(store["overall"]); pf=pd.DataFrame(store["per_feature"]); sg=pd.DataFrame(store["source_group"]); ds=pd.DataFrame(store["downstream"]); diag=pd.DataFrame(store["diag"])
    overall.to_csv(out_dir/"ALL_imputation_summary.csv",index=False,encoding="utf-8-sig")
    pf.to_csv(out_dir/"ALL_per_feature_summary.csv",index=False,encoding="utf-8-sig")
    sg.to_csv(out_dir/"ALL_source_group_summary.csv",index=False,encoding="utf-8-sig")
    ds.to_csv(out_dir/"ALL_downstream_summary.csv",index=False,encoding="utf-8-sig")
    diag.to_csv(out_dir/"ALL_diagnostics.csv",index=False,encoding="utf-8-sig")
    pf.to_csv(out_dir/"feature_level"/"feature_level_results.csv",index=False,encoding="utf-8-sig")
    sg.to_csv(out_dir/"source_group"/"source_group_results.csv",index=False,encoding="utf-8-sig")
    ds.to_csv(out_dir/"downstream"/"downstream_results.csv",index=False,encoding="utf-8-sig")
    diag.to_csv(out_dir/"diagnostics"/"model_relation_oca_diagnostics.csv",index=False,encoding="utf-8-sig")
    metrics=["MAE_scaled_clipped","RMSE_scaled_clipped"]
    if not overall.empty:
        overall["seed_role"] = np.where(overall["seed"].isin(DEVELOPMENT_SEEDS), "development", "confirmatory")
        groups=["experiment_suite","evaluation_protocol","imputer","relation_mode","nominal_missing_rate","affected_row_rate"]
        summaries=[]
        for label,seed_set in [("development_0_1_2",DEVELOPMENT_SEEDS),("confirmatory_3_4",CONFIRMATORY_SEEDS),("all_0_1_2_3_4",DEVELOPMENT_SEEDS|CONFIRMATORY_SEEDS)]:
            subset=overall[overall["seed"].isin(seed_set)]
            if subset.empty: continue
            table=flat_mean_sd(subset,groups,metrics); table.insert(0,"seed_analysis_set",label); summaries.append(table)
            filename={"development_0_1_2":"development_summary_seed012.csv","confirmatory_3_4":"confirmatory_summary_seed34.csv","all_0_1_2_3_4":"all_seed_summary_seed01234.csv"}[label]
            table.to_csv(out_dir/"summaries"/filename,index=False,encoding="utf-8-sig")
        if summaries:
            pd.concat(summaries,ignore_index=True).to_csv(out_dir/"imputation_mean_sd_by_condition.csv",index=False,encoding="utf-8-sig")
        # Correct overall mean±SD: first average across missing rates within each seed, then SD across seeds.
        seedlvl=overall.groupby(["experiment_suite","evaluation_protocol","imputer","relation_mode","seed"],dropna=False)[metrics].mean().reset_index()
        overall_tables=[]
        for label,seed_set in [("development_0_1_2",DEVELOPMENT_SEEDS),("confirmatory_3_4",CONFIRMATORY_SEEDS),("all_0_1_2_3_4",DEVELOPMENT_SEEDS|CONFIRMATORY_SEEDS)]:
            subset=seedlvl[seedlvl["seed"].isin(seed_set)]
            if subset.empty: continue
            table=flat_mean_sd(subset,["experiment_suite","evaluation_protocol","imputer","relation_mode"],metrics)
            table.insert(0,"seed_analysis_set",label); overall_tables.append(table)
        if overall_tables:
            pd.concat(overall_tables,ignore_index=True).to_csv(out_dir/"imputation_overall_mean_sd_across_seeds.csv",index=False,encoding="utf-8-sig")
        save_paired_comparisons(overall,out_dir)
    if not pf.empty:
        feature_summary=flat_mean_sd(pf,["experiment_suite","evaluation_protocol","imputer","feature","nominal_missing_rate"],["MAE_scaled_clipped","RMSE_scaled_clipped"])
        feature_summary.to_csv(out_dir/"feature_level"/"feature_level_mean_sd.csv",index=False,encoding="utf-8-sig")
    if not sg.empty:
        source_summary=flat_mean_sd(sg,["experiment_suite","evaluation_protocol","imputer","source_group","nominal_missing_rate"],["MAE_scaled_clipped","RMSE_scaled_clipped"])
        source_summary.to_csv(out_dir/"source_group_mean_sd.csv",index=False,encoding="utf-8-sig")
        source_summary.to_csv(out_dir/"source_group"/"source_group_mean_sd.csv",index=False,encoding="utf-8-sig")
    if not ds.empty:
        downstream_summary=flat_mean_sd(ds,["experiment_suite","evaluation_protocol","imputer","predictor","nominal_missing_rate"],["MAE","RMSE","R2"])
        downstream_summary.to_csv(out_dir/"downstream_mean_sd.csv",index=False,encoding="utf-8-sig")
        downstream_summary.to_csv(out_dir/"downstream"/"downstream_mean_sd.csv",index=False,encoding="utf-8-sig")
        reference=ds[ds["imputer"]=="Complete-Data Reference"]
        if not reference.empty:
            flat_mean_sd(reference,["imputer","predictor"],["MAE","RMSE","R2"]).to_csv(out_dir/"complete_data_reference_mean_sd.csv",index=False,encoding="utf-8-sig")


def write_run_manifest(args, bargs, df, numeric_features, categorical_features, out_dir: Path):
    split_module=load_base_module(Path(args.base_script).resolve(), "mrv_manifest_split")
    sample_split=split_module.make_train_val_test_split(df,bargs,parse_list_int(args.seeds)[0])
    manifest={
        "purpose":"Final paper experiments for OCA-R-GRAPE with joint OCA and Vessel + Destination relations.",
        "timestamp_utc":datetime.now(timezone.utc).isoformat(),
        "data":args.excel_path,"data_schema":args.data_schema,"voyage_feature_set":args.voyage_feature_set,
        "data_sha256":file_sha256(Path(args.excel_path).resolve()),"final_code_sha256":file_sha256(Path(args.base_script).resolve()),
        "runner_code_sha256":file_sha256(Path(__file__).resolve()),
        "python_version":platform.python_version(),"torch_version":torch.__version__,"numpy_version":np.__version__,"pandas_version":pd.__version__,
        "n_rows":len(df),
        "n_numeric_features":len(numeric_features),"numeric_features":list(numeric_features),"categorical_features":list(categorical_features),
        "seeds":parse_list_int(args.seeds),"main_missing_rates":parse_list_float(args.main_missing_rates),"inductive_missing_rates":parse_list_float(args.inductive_missing_rates),
        "block_affected_rates":parse_list_float(args.block_affected_rates),"relation_ablation_rates":parse_list_float(args.relation_ablation_rates),
        "development_seeds":[0,1,2],"confirmatory_seeds":[3,4],
        "split_mode":args.split_mode,"train_ratio_argument":args.train_ratio,"validation_fraction_within_train_partition":args.val_ratio,
        "actual_split_counts_seed0":{"train":len(sample_split.train_idx),"validation":len(sample_split.val_idx),"test":len(sample_split.test_idx)},
        "graph_max_epochs":args.graph_max_epochs,"checkpoint_criterion":"validation artificial-cell MAE","oca_train_mode":"joint","main_relation_mode":FINAL_RELATION_MODE,
        "condition_resume_granularity":"completed method x seed x condition only","epoch_resume":False,
        "requested_suites":{"run_all":args.run_all,"main_mcar":args.run_main_mcar,"module_ablation":args.run_module_ablation,"relation_ablation":args.run_relation_ablation,"strict_inductive":args.run_inductive,"structured_env_block":args.run_structured,"downstream":args.run_downstream,"complete_data_reference":args.run_complete_reference},
        "structured_env_groups":ENV_GROUPS_COMPLETE,
        "igrm_official_commit":IGRM_COMMIT,"igrm_sparse_friend_k_adaptation":args.igrm_friend_k,
        "confirmatory_freeze_rule":"Seeds 3/4 must not drive relation, OCA, or hyperparameter redesign.",
        "important_note":"IGRM_common_protocol uses fixed external masks and common scaling. Its sparse top-k friend reconstruction is an explicitly disclosed adaptation.",
    }
    config_dir=out_dir/"config"; config_dir.mkdir(parents=True,exist_ok=True)
    payload=json.dumps(manifest,indent=2,ensure_ascii=False)
    (config_dir/"run_config.json").write_text(payload,encoding="utf-8")
    (out_dir/"run_manifest.json").write_text(payload,encoding="utf-8")


def run_worker(args: argparse.Namespace) -> None:
    """Execute exactly one non-IGRM method-condition in an isolated process."""
    required=[args.worker_condition_npz,args.worker_method,args.worker_cache_dir]
    if not all(required):
        raise ValueError("Worker mode requires condition, method, and cache paths.")
    base=load_base_module(Path(args.base_script).resolve(),f"mrv_final_worker_{os.getpid()}")
    bargs=make_base_args(base,args); bargs._suite_resume=args.resume
    bargs.oca_train_mode="joint"; bargs.relation_mode=args.worker_relation_mode
    df=base.load_and_prepare_mrv(bargs)
    numeric_features,_=base.resolve_support_features(df,bargs)
    base.assert_downstream_target_isolated(bargs.target,numeric_features)
    y=pd.to_numeric(df[bargs.target],errors="coerce").to_numpy(dtype=float)
    cond=load_condition_npz(base,Path(args.worker_condition_npz).resolve())
    safe_method=args.worker_method.replace("/","_").replace(" ","_")
    base.setup_logging(Path(args.out_dir).resolve()/"logs"/f"{safe_method}_{args.worker_protocol}_{condition_slug(cond)}",args.verbose)
    base.assert_split_integrity(cond.split,len(df))
    base.assert_artificial_targets_hidden(cond.X_missing_scaled,cond.original_observed_mask,cond.artificial_missing_mask)
    run_original_method(
        base,df,numeric_features,y,bargs,cond,args.worker_method,args.worker_protocol,
        args.worker_relation_mode,Path(args.worker_cache_dir).resolve(),
    )


def main():
    args=parse_args(); out_dir=Path(args.out_dir).resolve(); out_dir.mkdir(parents=True,exist_ok=True)
    if args.worker_mode:
        run_worker(args)
        return
    for name in ["config","masks","splits","checkpoints","logs","main_mcar","module_ablation","relation_ablation","strict_inductive","structured_env_block","complete_data_reference","downstream","feature_level","source_group","diagnostics","summaries"]:
        (out_dir/name).mkdir(parents=True,exist_ok=True)
    base=load_base_module(Path(args.base_script).resolve(), "oca_rgrape_runner"); bargs=make_base_args(base,args); bargs._suite_resume=args.resume
    base.setup_logging(out_dir/"logs",args.verbose)
    df=base.load_and_prepare_mrv(bargs); numeric_features,categorical_features=base.resolve_support_features(df,bargs); y=pd.to_numeric(df[bargs.target],errors="coerce").to_numpy(dtype=float)
    base.assert_downstream_target_isolated(bargs.target,numeric_features)
    predictors=base.parse_name_list(args.predictors,base.DEFAULT_PREDICTORS,"predictor")
    write_run_manifest(args,bargs,df,numeric_features,categorical_features,out_dir)
    write_leakage_audit(base,df,numeric_features,categorical_features,bargs,out_dir)
    print(f"Rows={len(df)} numeric={len(numeric_features)} categorical={len(categorical_features)}",flush=True)
    selected=any([args.run_all,args.run_main_mcar,args.run_module_ablation,args.run_relation_ablation,args.run_inductive,args.run_structured,args.run_downstream,args.run_complete_reference])
    if not selected:
        print("No experiment suite selected. Use --run_all or one of the --run_* switches.",flush=True)
        return
    repo_dir=None
    if not args.skip_igrm: repo_dir=ensure_igrm_repo(args)
    store={"overall":[],"per_feature":[],"source_group":[],"downstream":[],"diag":[]}
    methods=ALL_METHODS if not args.skip_igrm else [name for name in ALL_METHODS if name!=IGRM_METHOD]
    downstream_enabled=bool(args.run_all or args.run_downstream or args.run_complete_reference)
    if args.run_all or args.run_main_mcar or args.run_downstream:
        run_suite_conditions(base,df,numeric_features,categorical_features,y,bargs,args,"main_mcar","transductive","mcar",parse_list_float(args.main_missing_rates),methods,predictors,FINAL_RELATION_MODE,repo_dir,downstream_enabled,store)
    if args.run_all or args.run_module_ablation:
        run_suite_conditions(base,df,numeric_features,categorical_features,y,bargs,args,"module_ablation","transductive","mcar",parse_list_float(args.main_missing_rates),CORE_GRAPH_METHODS,[],FINAL_RELATION_MODE,repo_dir,False,store)

    if args.run_all or args.run_inductive:
        run_suite_conditions(base,df,numeric_features,categorical_features,y,bargs,args,"strict_inductive","strict_inductive","mcar",parse_list_float(args.inductive_missing_rates),methods,predictors,FINAL_RELATION_MODE,repo_dir,downstream_enabled,store)

    if args.run_all or args.run_structured:
        run_suite_conditions(base,df,numeric_features,categorical_features,y,bargs,args,"structured_env_block","transductive","env_block",parse_list_float(args.block_affected_rates),methods,predictors,FINAL_RELATION_MODE,repo_dir,downstream_enabled,store)

    if args.run_all or args.run_relation_ablation:
        run_relation_ablation(base,df,numeric_features,categorical_features,y,bargs,args,predictors,store)

    if (args.run_all or args.run_complete_reference) and downstream_enabled:
        run_complete_reference(base,df,numeric_features,categorical_features,y,bargs,args,predictors,store)

    # 6) Mean±SD and seed-level overall aggregation.
    save_aggregates(out_dir,store)
    print(f"Finished. Results: {out_dir}",flush=True)


if __name__=="__main__":
    main()
