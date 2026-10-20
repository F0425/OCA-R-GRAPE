from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="IGRM adapter under a fixed external missing-data protocol.")
    p.add_argument("--condition_npz", required=True)
    p.add_argument("--output_npy", required=True)
    p.add_argument("--diag_json", required=True)
    p.add_argument("--igrm_repo_dir", required=True)
    p.add_argument("--protocol", choices=["transductive", "strict_inductive"], default="transductive")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=20000)
    p.add_argument("--eval_every", type=int, default=50)
    p.add_argument("--known", type=float, default=0.7)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--model_types", default="EGSAGE_EGSAGE_EGSAGE")
    p.add_argument("--node_dim", type=int, default=64)
    p.add_argument("--edge_dim", type=int, default=64)
    p.add_argument("--edge_mode", type=int, default=1)
    p.add_argument("--gnn_activation", default="relu")
    p.add_argument("--impute_hiddens", default="64")
    p.add_argument("--impute_activation", default="relu")
    p.add_argument("--aggr", default="mean")
    p.add_argument("--node_mode", type=int, default=0)
    p.add_argument("--reconstruct_every", type=int, default=100)
    p.add_argument("--friend_k", type=int, default=10)
    p.add_argument("--friend_temperature", type=float, default=0.2)
    p.add_argument("--loss_scope", choices=["combined"], default="combined")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_node_features(n_rows: int, n_features: int, node_mode: int) -> np.ndarray:
    if int(node_mode) == 0:
        feat = np.zeros((n_features, n_features), dtype=np.float32)
        feat[np.arange(n_features), np.arange(n_features)] = 1.0
        obs = np.ones((n_rows, n_features), dtype=np.float32)
        return np.vstack([obs, feat])
    feat = np.zeros((n_features, n_features + 1), dtype=np.float32)
    feat[np.arange(n_features), np.arange(n_features) + 1] = 1.0
    obs = np.zeros((n_rows, n_features + 1), dtype=np.float32)
    obs[:, 0] = 1.0
    return np.vstack([obs, feat])


def pairs_to_edges(pairs: np.ndarray, values: np.ndarray, n_rows: int) -> Tuple[np.ndarray, np.ndarray]:
    if len(pairs) == 0:
        return np.empty((2, 0), dtype=np.int64), np.empty((0, 1), dtype=np.float32)
    src = pairs[:, 0].astype(np.int64)
    dst = (n_rows + pairs[:, 1]).astype(np.int64)
    ei = np.vstack([np.concatenate([src, dst]), np.concatenate([dst, src])])
    ea = np.concatenate([values, values]).astype(np.float32).reshape(-1, 1)
    return ei, ea


def make_message_graph(edge_index: torch.Tensor, edge_attr: torch.Tensor, known: float) -> Tuple[torch.Tensor, torch.Tensor]:
    half = int(edge_attr.shape[0] // 2)
    if half > 0 and known < 1.0:
        keep_h = torch.rand(half, device=edge_attr.device) < float(known)
        if int(keep_h.sum()) == 0:
            keep_h[:] = True
        keep = torch.cat([keep_h, keep_h], dim=0)
        return edge_index[:, keep], edge_attr[keep]
    return edge_index, edge_attr


class SparseGCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, activation=None):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=False)
        self.activation = activation
        nn.init.xavier_uniform_(self.lin.weight)

    def forward(self, adj: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        z = self.lin(x)
        z = torch.sparse.mm(adj, z) if adj.is_sparse else adj @ z
        return self.activation(z) if self.activation is not None else z


class FriendGAE(nn.Module):
    """Two-layer deterministic GAE matching the friend-network reconstructor used by IGRM."""
    def __init__(self, dim_feats: int = 64, dim_h: int = 128, dim_z: int = 32):
        super().__init__()
        self.g1 = SparseGCNLayer(dim_feats, dim_h, activation=None)
        self.g2 = SparseGCNLayer(dim_h, dim_z, activation=F.relu)

    def forward(self, adj: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        h = self.g1(adj, features)
        z = self.g2(adj, h)
        return z @ z.T


def random_friend_edges(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    target = max(1, n)
    pairs = set()
    while len(pairs) < target and n > 1:
        i, j = rng.choice(n, size=2, replace=False).tolist()
        if i > j:
            i, j = j, i
        if i != j:
            pairs.add((int(i), int(j)))
    if not pairs:
        return np.empty((2, 0), dtype=np.int64)
    p = np.array(sorted(pairs), dtype=np.int64)
    return p.T


def normalized_sparse_adj(n: int, undirected_upper_edges: np.ndarray, device: torch.device) -> torch.Tensor:
    # IGRM's public code normalizes A+I symmetrically. Reproduce that operation without SciPy.
    if undirected_upper_edges.size:
        src = np.concatenate([undirected_upper_edges[0], undirected_upper_edges[1], np.arange(n)])
        dst = np.concatenate([undirected_upper_edges[1], undirected_upper_edges[0], np.arange(n)])
    else:
        src = np.arange(n)
        dst = np.arange(n)
    src_t = torch.tensor(src, dtype=torch.long, device=device)
    dst_t = torch.tensor(dst, dtype=torch.long, device=device)
    deg = torch.zeros(n, dtype=torch.float32, device=device)
    deg.index_add_(0, src_t, torch.ones_like(src_t, dtype=torch.float32))
    dinv = torch.rsqrt(torch.clamp(deg, min=1.0))
    values = dinv[src_t] * dinv[dst_t]
    indices = torch.stack([src_t, dst_t], dim=0)
    return torch.sparse_coo_tensor(indices, values, (n, n), device=device).coalesce()


def sparse_friend_from_logits(
    logits: torch.Tensor,
    friend_k: int,
    temperature: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reconstruct the friend network from IGRM affinity logits.

    friend_k <= 0 follows the public implementation closely by hard-sampling the full
    upper-triangular relaxed-Bernoulli adjacency. friend_k > 0 is a sparse top-k adaptation
    for voyage tables, where dense sampling may generate millions of friend edges.
    """
    n = int(logits.shape[0])
    scores = F.relu(logits)
    mx = torch.max(scores.detach())
    if not torch.isfinite(mx) or float(mx) <= 0:
        return torch.empty((2, 0), dtype=torch.long, device=logits.device), torch.empty((0,), dtype=torch.float32, device=logits.device)
    probs = torch.clamp(scores / mx, 0.0, 1.0)
    probs = probs - torch.diag_embed(torch.diagonal(probs))
    tri_probs = torch.triu(probs, diagonal=1)

    if int(friend_k) <= 0:
        eps = 1e-6
        p = torch.clamp(tri_probs, eps, 1.0 - eps)
        u = torch.rand_like(p)
        logistic_noise = torch.log(torch.clamp(u, eps, 1-eps)) - torch.log(torch.clamp(1-u, eps, 1-eps))
        relaxed = torch.sigmoid((torch.log(p) - torch.log1p(-p) + logistic_noise) / max(float(temperature), 1e-6))
        hard = (relaxed > 0.5)
        hard = torch.triu(hard, diagonal=1)
        edge_index = torch.nonzero(hard, as_tuple=False).T
        if edge_index.numel() == 0:
            return torch.empty((2,0),dtype=torch.long,device=logits.device), torch.empty((0,),dtype=torch.float32,device=logits.device)
        weights = probs[edge_index[0], edge_index[1]]
        return edge_index, weights

    flat = tri_probs.reshape(-1)
    n_possible = n * (n - 1) // 2
    keep = min(max(1, int(friend_k) * n // 2), max(1, n_possible))
    vals, idx = torch.topk(flat, k=min(keep, flat.numel()), largest=True, sorted=False)
    positive = vals > 0
    vals = vals[positive]
    idx = idx[positive]
    if idx.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long, device=logits.device), torch.empty((0,), dtype=torch.float32, device=logits.device)
    src = torch.div(idx, n, rounding_mode="floor")
    dst = idx % n
    edge_index = torch.stack([src, dst], dim=0)
    # Public IGRM uses normalized reconstructed affinity as friend-edge weight.
    return edge_index, vals


@dataclass
class LocalGraph:
    X_true: np.ndarray
    X_missing: np.ndarray
    artificial_mask: np.ndarray
    x: torch.Tensor
    value_edge_index: torch.Tensor
    value_edge_attr: torch.Tensor
    artificial_pairs: torch.Tensor
    artificial_labels: torch.Tensor
    observed_pairs: torch.Tensor
    observed_labels: torch.Tensor
    init_friend_adj: torch.Tensor
    n_rows: int
    n_features: int


def make_local_graph(X_true: np.ndarray, X_missing: np.ndarray, artificial_mask: np.ndarray, node_mode: int, seed: int, device: torch.device) -> LocalGraph:
    n, d = X_missing.shape
    obs = np.isfinite(X_missing)
    obs_pairs = np.argwhere(obs)
    obs_vals = X_missing[obs].astype(np.float32)
    ei_np, ea_np = pairs_to_edges(obs_pairs, obs_vals, n)
    art_pairs_np = np.argwhere(artificial_mask)
    art_vals_np = X_true[artificial_mask].astype(np.float32)
    x = torch.tensor(make_node_features(n, d, node_mode), dtype=torch.float32, device=device)
    ei = torch.tensor(ei_np, dtype=torch.long, device=device)
    ea = torch.tensor(ea_np, dtype=torch.float32, device=device)
    art_pairs = torch.tensor(art_pairs_np, dtype=torch.long, device=device)
    art_labels = torch.tensor(art_vals_np, dtype=torch.float32, device=device)
    obs_pairs_t = torch.tensor(obs_pairs, dtype=torch.long, device=device)
    obs_labels_t = torch.tensor(X_true[obs].astype(np.float32), dtype=torch.float32, device=device)
    initial = random_friend_edges(n, seed)
    init_adj = normalized_sparse_adj(n, initial, device)
    return LocalGraph(X_true, X_missing, artificial_mask, x, ei, ea, art_pairs, art_labels, obs_pairs_t, obs_labels_t, init_adj, n, d)


def get_model_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        model_types=args.model_types,
        post_hiddens=None,
        concat_states=False,
        norm_embs=None,
        aggr=args.aggr,
        node_dim=args.node_dim,
        edge_dim=args.edge_dim,
        edge_mode=args.edge_mode,
        gnn_activation=args.gnn_activation,
        dropout=args.dropout,
    )


def hidden_list(text: str) -> List[int]:
    return [] if str(text).strip() == "" else [int(v) for v in str(text).split("_") if v.strip()]


def predict_pairs(emb: torch.Tensor, impute_model: nn.Module, pairs: torch.Tensor, n_rows: int) -> torch.Tensor:
    if pairs.numel() == 0:
        return torch.empty((0,), dtype=torch.float32, device=emb.device)
    return impute_model([emb[pairs[:, 0]], emb[n_rows + pairs[:, 1]]])[:, 0]


def augment_embeddings(model: nn.Module, gae: FriendGAE, graph: LocalGraph, full_value_edges: bool, args: argparse.Namespace, friend_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, require_friend_grad: bool = False) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    ei, ea = (graph.value_edge_index, graph.value_edge_attr) if full_value_edges else make_message_graph(graph.value_edge_index, graph.value_edge_attr, args.known)
    emb = model(graph.x, ea, ei)
    if friend_cache is None:
        logits = gae(graph.init_friend_adj, emb[:graph.n_rows].clone())
        fei, few = sparse_friend_from_logits(logits, args.friend_k, args.friend_temperature)
    else:
        fei, few = friend_cache
        if not require_friend_grad:
            fei, few = fei.detach(), few.detach()
    if fei.numel() > 0:
        sample_aug = model.F_augmentation(emb[:graph.n_rows], few, fei)
        emb = emb.clone()
        emb[:graph.n_rows] = sample_aug
    emb = model.node_post_mlp(emb)
    return emb, (fei, few)


def eval_artificial(model: nn.Module, impute_model: nn.Module, gae: FriendGAE, graph: LocalGraph, args: argparse.Namespace) -> Tuple[float, float]:
    if graph.artificial_pairs.numel() == 0:
        return float("nan"), float("nan")
    model.eval(); impute_model.eval(); gae.eval()
    with torch.no_grad():
        emb, _ = augment_embeddings(model, gae, graph, True, args, None, False)
        pred = predict_pairs(emb, impute_model, graph.artificial_pairs, graph.n_rows)
        mae = F.l1_loss(pred, graph.artificial_labels).item()
        rmse = math.sqrt(F.mse_loss(pred, graph.artificial_labels).item())
    return float(mae), float(rmse)


def predict_full(model: nn.Module, impute_model: nn.Module, gae: FriendGAE, graph: LocalGraph, args: argparse.Namespace, batch_size: int = 65536) -> np.ndarray:
    model.eval(); impute_model.eval(); gae.eval()
    pairs_np = np.array([(i, j) for i in range(graph.n_rows) for j in range(graph.n_features)], dtype=np.int64)
    out = np.empty(len(pairs_np), dtype=np.float32)
    with torch.no_grad():
        emb, _ = augment_embeddings(model, gae, graph, True, args, None, False)
        for start in range(0, len(pairs_np), batch_size):
            pp = torch.tensor(pairs_np[start:start+batch_size], dtype=torch.long, device=emb.device)
            pred = predict_pairs(emb, impute_model, pp, graph.n_rows)
            out[start:start+len(pp)] = pred.detach().cpu().numpy()
    return out.reshape(graph.n_rows, graph.n_features)


def build_models(repo_dir: Path, graph: LocalGraph, args: argparse.Namespace, device: torch.device):
    if not (repo_dir / "models" / "gnn_model.py").exists():
        raise FileNotFoundError(f"IGRM repository not found or incomplete: {repo_dir}")
    sys.path.insert(0, str(repo_dir))
    from torch_geometric.data import Data as PyGData
    from models.gnn_model import get_gnn
    from models.prediction_model import MLPNet
    data = PyGData(x=graph.x, edge_attr_dim=1)
    model = get_gnn(data, get_model_args(args)).to(device)
    input_dim = args.node_dim * 2
    impute_model = MLPNet(input_dim, 1, hidden_layer_sizes=hidden_list(args.impute_hiddens), hidden_activation=args.impute_activation, dropout=args.dropout).to(device)
    gae = FriendGAE(args.node_dim, 128, 32).to(device)
    return model, impute_model, gae


def train_model(train_graph: LocalGraph, val_graph: LocalGraph, repo_dir: Path, args: argparse.Namespace, device: torch.device):
    model, impute_model, gae = build_models(repo_dir, train_graph, args, device)
    params = list(model.parameters()) + list(impute_model.parameters())
    opt = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    gae_opt = torch.optim.Adam(gae.parameters(), lr=0.001)
    best_score = float("inf")
    best_epoch = 0
    best_state = None
    friend_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    for epoch in range(1, args.epochs + 1):
        model.train(); impute_model.train(); gae.train()
        opt.zero_grad(); gae_opt.zero_grad()
        reconstruct = friend_cache is None or ((epoch - 1) % max(1, args.reconstruct_every) == 0)
        emb, new_cache = augment_embeddings(
            model, gae, train_graph, False, args,
            friend_cache=None if reconstruct else friend_cache,
            require_friend_grad=reconstruct,
        )
        friend_cache = new_cache

        losses = []
        if train_graph.artificial_pairs.numel() > 0:
            pa = predict_pairs(emb, impute_model, train_graph.artificial_pairs, train_graph.n_rows)
            losses.append(F.mse_loss(pa, train_graph.artificial_labels))
        if train_graph.observed_pairs.numel() > 0:
            po = predict_pairs(emb, impute_model, train_graph.observed_pairs, train_graph.n_rows)
            losses.append(F.mse_loss(po, train_graph.observed_labels))
        if not losses:
            raise RuntimeError("IGRM has no training supervision under the supplied condition.")
        loss = torch.stack(losses).sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()
        if reconstruct:
            torch.nn.utils.clip_grad_norm_(gae.parameters(), 5.0)
            gae_opt.step()
            # The public implementation detaches reconstructed friend weights after their update epoch.
            friend_cache = (friend_cache[0].detach(), friend_cache[1].detach())

        if epoch == 1 or epoch % max(1, args.eval_every) == 0 or epoch == args.epochs:
            vmae, vrmse = eval_artificial(model, impute_model, gae, val_graph, args)
            score = vmae if np.isfinite(vmae) else float(loss.detach().item())
            if score < best_score:
                best_score = float(score); best_epoch = int(epoch)
                best_state = {
                    "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    "impute": {k: v.detach().cpu().clone() for k, v in impute_model.state_dict().items()},
                    "gae": {k: v.detach().cpu().clone() for k, v in gae.state_dict().items()},
                }
            if args.verbose:
                print(f"IGRM epoch {epoch}/{args.epochs} loss={float(loss.detach()):.6f} val_mae={vmae:.6f} val_rmse={vrmse:.6f} best={best_epoch}", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state["model"])
        impute_model.load_state_dict(best_state["impute"])
        gae.load_state_dict(best_state["gae"])
    return model, impute_model, gae, best_epoch, best_score


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    z = np.load(args.condition_npz, allow_pickle=False)
    X_true = z["X_true_scaled"].astype(float)
    X_missing = z["X_missing_scaled"].astype(float)
    artificial = z["artificial_missing_mask"].astype(bool)
    train_idx = z["train_idx"].astype(int)
    val_idx = z["val_idx"].astype(int)
    test_idx = z["test_idx"].astype(int)
    repo = Path(args.igrm_repo_dir).resolve()

    if args.protocol == "transductive":
        # One graph contains all rows, while supervision/checkpoint cells remain split by row.
        train_row = np.zeros(len(X_true), dtype=bool); train_row[train_idx] = True
        val_row = np.zeros(len(X_true), dtype=bool); val_row[val_idx] = True
        train_art = artificial & train_row[:, None]
        val_art = artificial & val_row[:, None]
        train_graph = make_local_graph(X_true, X_missing, train_art, args.node_mode, args.seed + 1, device)
        train_observed = np.isfinite(X_missing) & train_row[:, None]
        train_observed_pairs = np.argwhere(train_observed)
        train_graph.observed_pairs = torch.tensor(train_observed_pairs, dtype=torch.long, device=device)
        train_graph.observed_labels = torch.tensor(X_true[train_observed].astype(np.float32), dtype=torch.float32, device=device)
        # Validation uses the same transductive input graph but a different target set.
        val_graph = copy.copy(train_graph)
        val_graph.artificial_mask = val_art
        vp = np.argwhere(val_art)
        val_graph.artificial_pairs = torch.tensor(vp, dtype=torch.long, device=device)
        val_graph.artificial_labels = torch.tensor(X_true[val_art].astype(np.float32), dtype=torch.float32, device=device)
        model, impute, gae, best_epoch, best_score = train_model(train_graph, val_graph, repo, args, device)
        pred = predict_full(model, impute, gae, train_graph, args)
        imputed = X_missing.copy()
        miss = ~np.isfinite(imputed)
        imputed[miss] = pred[miss]
    else:
        # Strict unseen-voyage protocol: separate train/validation/test graphs.
        tr = make_local_graph(X_true[train_idx], X_missing[train_idx], artificial[train_idx], args.node_mode, args.seed + 11, device)
        va = make_local_graph(X_true[val_idx], X_missing[val_idx], artificial[val_idx], args.node_mode, args.seed + 17, device)
        model, impute, gae, best_epoch, best_score = train_model(tr, va, repo, args, device)
        for parameter in list(model.parameters()) + list(impute.parameters()) + list(gae.parameters()):
            parameter.requires_grad_(False)
        model.eval(); impute.eval(); gae.eval()
        te = make_local_graph(X_true[test_idx], X_missing[test_idx], artificial[test_idx], args.node_mode, args.seed + 23, device)
        imputed = X_missing.copy()
        for idx, graph in [(train_idx, tr), (val_idx, va), (test_idx, te)]:
            pred = predict_full(model, impute, gae, graph, args)
            local = X_missing[idx].copy()
            miss = ~np.isfinite(local)
            local[miss] = pred[miss]
            imputed[idx] = local

    Path(args.output_npy).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output_npy, imputed.astype(np.float32))
    diag = {
        "method_name": "IGRM_common_protocol",
        "implementation": "official-architecture-adapted-to-common-protocol",
        "official_repository": "G-AILab/IGRM",
        "official_commit": "5cfc955daa5d0f4bbdcbad1552cfd7493dfd5fd0",
        "protocol": args.protocol,
        "best_epoch": int(best_epoch),
        "best_validation_MAE": float(best_score),
        "best_valid_mae": float(best_score),
        "final_training_epoch": int(args.epochs),
        "friend_reconstruction_every": int(args.reconstruct_every),
        "friend_k_sparse_adaptation": int(args.friend_k),
        "sparse_friend_graph_adaptation": True,
        "fixed_external_mask": True,
        "common_train_fitted_scaling": True,
        "checkpoint_protocol": "minimum validation artificial-cell MAE",
        "observed_supervision_scope": "training_rows_only",
        "test_graph_constructed_after_best_checkpoint": bool(args.protocol == "strict_inductive"),
        "note": "Uses the public IGRM GNN/Friend-SAGE architecture with a disclosed sparse top-k friend-graph adaptation under fixed external masks and common train-fitted scaling.",
    }
    Path(args.diag_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.diag_json).write_text(json.dumps(diag, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
