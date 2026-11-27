"""缩减版 FlyWire 脑核：子图构建、rate-coded 前向、dopamine-gated col_gain。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from player.neuro.constants import (
    COMPLETENESS_CSV,
    CONNECTIVITY_PARQUET,
    DATA_DIR,
    DEFAULT_HOPS,
    DEFAULT_MAX_NEURONS,
    ELIG_DECAY,
    GROUPS_JSON,
    LEARN_RATE,
    MAX_GAIN,
    READOUT_ORDER,
    STIM_GROUPS,
    SUBGRAPH_CACHE,
    W_SCALE,
)


def _data_paths(data_dir: Optional[Path] = None) -> Tuple[Path, Path, Path, Path]:
    root = Path(data_dir or DATA_DIR)
    return (
        root / COMPLETENESS_CSV,
        root / CONNECTIVITY_PARQUET,
        root / GROUPS_JSON,
        root / SUBGRAPH_CACHE,
    )


def _load_groups(groups_path: Path, flyid2i: Dict[int, int]) -> Dict[str, np.ndarray]:
    raw = json.loads(groups_path.read_text())
    out = {}
    for k, ids in raw.items():
        idx = [flyid2i[int(i)] for i in ids if int(i) in flyid2i]
        out[k] = np.asarray(idx, dtype=np.int64)
    return out


def _bfs_subgraph(
    n: int,
    pre: np.ndarray,
    post: np.ndarray,
    seeds: np.ndarray,
    hops: int,
    max_neurons: int,
) -> np.ndarray:
    """无向邻接 BFS，返回选中的全局神经元下标（升序）。"""
    from collections import defaultdict, deque

    adj = defaultdict(list)
    for a, b in zip(pre.tolist(), post.tolist()):
        adj[a].append(b)
        adj[b].append(a)

    seen = set(int(x) for x in seeds.tolist())
    q = deque((int(x), 0) for x in seeds.tolist())
    while q and len(seen) < max_neurons:
        u, d = q.popleft()
        if d >= hops:
            continue
        for v in adj.get(u, []):
            if v not in seen:
                seen.add(v)
                q.append((v, d + 1))
                if len(seen) >= max_neurons:
                    break
    return np.array(sorted(seen), dtype=np.int64)


def build_or_load_subgraph(
    data_dir: Optional[Path] = None,
    hops: int = DEFAULT_HOPS,
    max_neurons: int = DEFAULT_MAX_NEURONS,
    force_rebuild: bool = False,
) -> dict:
    """
    返回字典：
      W_csr: scipy.sparse.csr_matrix (n_sub, n_sub)
      groups_local: name -> local indices
      stim_local / readout_local
      n, mean_gain plastic mask
    """
    from scipy import sparse
    import pandas as pd

    csv_path, pq_path, groups_path, cache_path = _data_paths(data_dir)
    cache_key = f"h{hops}_m{max_neurons}"
    meta_path = cache_path.with_suffix(f".{cache_key}.npz")
    npz_path = cache_path.with_suffix(f".{cache_key}.npz")

    if npz_path.exists() and not force_rebuild:
        blob = np.load(npz_path, allow_pickle=True)
        print(f"[brain] loaded subgraph cache {npz_path} n={int(blob['n'])}")
        return {
            "n": int(blob["n"]),
            "W_data": blob["W_data"],
            "W_indices": blob["W_indices"],
            "W_indptr": blob["W_indptr"],
            "groups_local": blob["groups_local"].item(),
            "plastic_local": blob["plastic_local"],
        }

    print(f"[brain] loading connectome from {pq_path} ...")
    comp = pd.read_csv(csv_path, index_col=0)
    fly_ids = [int(x) for x in comp.index]
    flyid2i = {fid: i for i, fid in enumerate(fly_ids)}
    n_full = len(fly_ids)

    conn = pd.read_parquet(pq_path)
    pre = conn["Presynaptic_Index"].to_numpy(dtype=np.int64)
    post = conn["Postsynaptic_Index"].to_numpy(dtype=np.int64)
    w = conn["Excitatory x Connectivity"].to_numpy(dtype=np.float32)

    groups = _load_groups(groups_path, flyid2i)
    seeds = np.unique(np.concatenate([v for v in groups.values() if len(v)]))
    if len(seeds) == 0:
        raise RuntimeError("No group neurons found in completeness CSV")

    keep = _bfs_subgraph(n_full, pre, post, seeds, hops=hops, max_neurons=max_neurons)
    global_to_local = {int(g): i for i, g in enumerate(keep.tolist())}
    n = len(keep)
    print(f"[brain] subgraph n={n} (from {n_full}), hops={hops}, max={max_neurons}")

    # filter edges inside subgraph
    mask = np.isin(pre, keep) & np.isin(post, keep)
    pre_l = np.array([global_to_local[int(x)] for x in pre[mask]], dtype=np.int64)
    post_l = np.array([global_to_local[int(x)] for x in post[mask]], dtype=np.int64)
    w_l = w[mask]
    # CSC/CSR: W[post, pre] = weight (column = presynaptic), matching fly-brain-snake
    W = sparse.csr_matrix((w_l, (post_l, pre_l)), shape=(n, n), dtype=np.float32)

    groups_local = {
        k: np.array([global_to_local[int(g)] for g in v if int(g) in global_to_local], dtype=np.int64)
        for k, v in groups.items()
    }
    plastic_parts = [
        groups_local.get("reward_sugar_left", np.array([], dtype=np.int64)),
        groups_local.get("reward_sugar_right", np.array([], dtype=np.int64)),
        groups_local.get("taste_sez", np.array([], dtype=np.int64)),
    ]
    plastic_parts = [p for p in plastic_parts if len(p)]
    plastic = np.unique(np.concatenate(plastic_parts)) if plastic_parts else np.array([], dtype=np.int64)

    np.savez_compressed(
        npz_path,
        n=n,
        W_data=W.data,
        W_indices=W.indices,
        W_indptr=W.indptr,
        groups_local=groups_local,
        plastic_local=plastic,
    )
    print(f"[brain] cached -> {npz_path}, nnz={W.nnz}")
    return {
        "n": n,
        "W_data": W.data,
        "W_indices": W.indices,
        "W_indptr": W.indptr,
        "groups_local": groups_local,
        "plastic_local": plastic,
    }


def _csr_to_torch(n, data, indices, indptr, device="cpu") -> torch.Tensor:
    """scipy CSR -> torch sparse CSR (PyTorch 2+) or COO fallback."""
    crow = torch.as_tensor(indptr, dtype=torch.int64, device=device)
    col = torch.as_tensor(indices, dtype=torch.int64, device=device)
    vals = torch.as_tensor(data, dtype=torch.float32, device=device)
    try:
        return torch.sparse_csr_tensor(crow, col, vals, size=(n, n))
    except Exception:
        # COO: (row, col) from CSR
        rows = np.repeat(np.arange(n), np.diff(indptr))
        i = torch.stack(
            [
                torch.as_tensor(rows, dtype=torch.int64, device=device),
                col,
            ]
        )
        return torch.sparse_coo_tensor(i, vals, size=(n, n)).coalesce()


class FlyBrainCore(nn.Module):
    """
    Rate-coded 缩减脑：
      x0 = inject(stim)
      h = phi(W @ (x * col_gain))  （2 步）
      rates = group_mean(h)
    方案二：apply_reward 更新 plastic 神经元的 col_gain（Hebbian）。
    """

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        hops: int = DEFAULT_HOPS,
        max_neurons: int = DEFAULT_MAX_NEURONS,
        steps: int = 2,
        device: str = "cpu",
        freeze_weights: bool = True,
    ):
        super().__init__()
        blob = build_or_load_subgraph(data_dir, hops=hops, max_neurons=max_neurons)
        self.n = int(blob["n"])
        self.steps = steps
        self.device_name = device

        from scipy import sparse as sp

        W_scipy = sp.csr_matrix(
            (blob["W_data"], blob["W_indices"], blob["W_indptr"]), shape=(self.n, self.n)
        )
        # 小子图用 dense 更快；大图保留 sparse
        self._use_dense = self.n <= 2048
        if self._use_dense:
            W_dense = torch.as_tensor(W_scipy.toarray(), dtype=torch.float32, device=device)
            self.register_buffer("W_dense", W_dense)
            self.W = None
        else:
            self.W = _csr_to_torch(
                self.n, blob["W_data"], blob["W_indices"], blob["W_indptr"], device=device
            )
            self.register_buffer("W_dense", torch.zeros(1, device=device))

        groups_local: Dict[str, np.ndarray] = blob["groups_local"]
        self.groups = {k: torch.as_tensor(v, dtype=torch.long, device=device) for k, v in groups_local.items()}

        plastic = blob["plastic_local"]
        self.register_buffer("plastic_idx", torch.as_tensor(plastic, dtype=torch.long, device=device))
        self.register_buffer("col_gain", torch.ones(self.n, dtype=torch.float32, device=device))
        self.register_buffer("elig", torch.zeros(self.n, dtype=torch.float32, device=device))

        self.stim_names = list(STIM_GROUPS)
        self.readout_names = list(READOUT_ORDER)
        self._readout_index_lists = self._build_readout_lists()

        if freeze_weights:
            self.col_gain.requires_grad_(False)

        self.mean_gain = 1.0

    def _build_readout_lists(self) -> List[torch.Tensor]:
        mapping = {
            "steer_left": ["steer_left"],
            "steer_right": ["steer_right"],
            "forward": ["motor_forward", "walk_p9"],
            "escape": ["escape_gf", "escape_mdn"],
            "feeding": ["feeding_mn9"],
            "threat": ["threat_lc4_left", "threat_lc4_right"],
            "reward": ["reward_sugar_left", "reward_sugar_right"],
        }
        out = []
        for name in self.readout_names:
            parts = []
            for g in mapping[name]:
                if g in self.groups and self.groups[g].numel() > 0:
                    parts.append(self.groups[g])
            if parts:
                out.append(torch.unique(torch.cat(parts)))
            else:
                out.append(torch.zeros(0, dtype=torch.long, device=self.col_gain.device))
        return out

    def reset_state(self):
        self.elig.zero_()
        self.col_gain.fill_(1.0)
        self.mean_gain = 1.0

    def inject_stim(self, stim: torch.Tensor) -> torch.Tensor:
        """
        stim: (B, STIM_DIM) Hz-like [0,1] or [0,300] — 归一化到活动幅度。
        返回 (B, n) 初始活动。
        """
        if stim.dim() == 1:
            stim = stim.unsqueeze(0)
        b = stim.shape[0]
        x = torch.zeros(b, self.n, device=stim.device, dtype=torch.float32)
        # normalize Hz-like to ~[0,1]
        s = stim.clamp(min=0.0) / 300.0
        for i, name in enumerate(self.stim_names):
            idx = self.groups.get(name)
            if idx is None or idx.numel() == 0:
                continue
            x[:, idx] = s[:, i].unsqueeze(1)
        return x

    def _maybe_move_W(self, device):
        """稀疏 W 不是 buffer，需手动迁移。"""
        if self.W is not None and self.W.device != device:
            self.W = self.W.to(device)

    def _spmm(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, n) -> (B, n) = W @ (x * col_gain)。"""
        v = x * self.col_gain.unsqueeze(0)
        if self._use_dense:
            # (B,n) @ (n,n)^T if W[post,pre]; we stored W as scipy (post, pre), so y = v @ W.T = W @ v
            y = torch.matmul(v, self.W_dense.T)
        else:
            self._maybe_move_W(x.device)
            outs = []
            for b in range(v.shape[0]):
                outs.append(torch.sparse.mm(self.W, v[b].unsqueeze(1)).squeeze(1))
            y = torch.stack(outs, dim=0)
        return y * W_SCALE

    def forward_rates(self, stim: torch.Tensor) -> torch.Tensor:
        """返回 (B, READOUT_DIM) group mean rates in ~[0,1+]."""
        x = self.inject_stim(stim)
        h = x
        for _ in range(self.steps):
            h = F.softplus(self._spmm(h) + 0.1 * h)
        # eligibility: recent activity
        with torch.no_grad():
            act = h.detach().mean(dim=0)
            self.elig.mul_(ELIG_DECAY)
            self.elig.add_(act.clamp(0, 1))

        rates = []
        for idx in self._readout_index_lists:
            if idx.numel() == 0:
                rates.append(torch.zeros(h.shape[0], device=h.device))
            else:
                rates.append(h[:, idx].mean(dim=1))
        return torch.stack(rates, dim=1)

    def apply_reward(self, pulses: float = 1.0):
        """Dopamine pulse: potentiate outgoing synapses of eligible plastic neurons."""
        if self.plastic_idx.numel() == 0:
            return
        elig = self.elig[self.plastic_idx]
        mask = elig > 0.05
        if not mask.any():
            return
        idx = self.plastic_idx[mask]
        boost = 1.0 + LEARN_RATE * pulses * self.elig[idx]
        newg = torch.minimum(self.col_gain[idx] * boost, torch.tensor(MAX_GAIN, device=idx.device))
        self.col_gain[idx] = newg
        self.mean_gain = float(self.col_gain[self.plastic_idx].mean().item())

    def forward(self, stim: torch.Tensor) -> torch.Tensor:
        return self.forward_rates(stim)

    def rates_numpy(self, stim_np: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            t = torch.as_tensor(stim_np, dtype=torch.float32, device=self.col_gain.device)
            if t.dim() == 1:
                t = t.unsqueeze(0)
            out = self.forward_rates(t)
            return out.squeeze(0).cpu().numpy()


def smoke_test(data_dir: Optional[Path] = None) -> None:
    from player.neuro.download_data import ensure_data

    ensure_data(allow_synthetic=True)
    brain = FlyBrainCore(data_dir=data_dir or DATA_DIR, device="cpu")
    stim = np.zeros(len(STIM_GROUPS), dtype=np.float32)
    stim[0] = 120.0
    stim[2] = 80.0
    rates = brain.rates_numpy(stim)
    print("stim", stim)
    print("rates", dict(zip(READOUT_ORDER, rates.tolist())))
    brain.apply_reward(1.0)
    print("mean_gain", brain.mean_gain)
    print("smoke ok, n=", brain.n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--hops", type=int, default=DEFAULT_HOPS)
    parser.add_argument("--max-neurons", type=int, default=DEFAULT_MAX_NEURONS)
    args = parser.parse_args()
    if args.rebuild:
        from player.neuro.download_data import ensure_data

        ensure_data(allow_synthetic=True)
        build_or_load_subgraph(hops=args.hops, max_neurons=args.max_neurons, force_rebuild=True)
    if args.smoke or not args.rebuild:
        smoke_test()


if __name__ == "__main__":
    main()
