"""下载 FlyWire 连接组与 groups.json（支持镜像；失败时可生成合成连接组）。"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

from player.neuro.constants import (
    COMPLETENESS_CSV,
    CONNECTIVITY_PARQUET,
    DATA_DIR,
    GROUPS_JSON,
)

GROUPS_URLS = [
    "https://ghproxy.net/https://raw.githubusercontent.com/SameerV7862/fly-brain-snake/main/groups.json",
    "https://cdn.jsdelivr.net/gh/SameerV7862/fly-brain-snake@main/groups.json",
    "https://raw.githubusercontent.com/SameerV7862/fly-brain-snake/main/groups.json",
]

CSV_URLS = [
    "https://ghproxy.net/https://github.com/eonsystemspbc/fly-brain/raw/main/data/2025_Completeness_783.csv",
    "https://github.com/eonsystemspbc/fly-brain/raw/main/data/2025_Completeness_783.csv",
]

PARQUET_URLS = [
    "https://ghproxy.net/https://github.com/eonsystemspbc/fly-brain/raw/main/data/2025_Connectivity_783.parquet",
    "https://github.com/eonsystemspbc/fly-brain/raw/main/data/2025_Connectivity_783.parquet",
]


def _download(urls: list[str], dest: Path, min_bytes: int = 100, timeout: int = 60) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size >= min_bytes:
        print(f"[skip] {dest} already exists ({dest.stat().st_size} bytes)")
        return True
    for url in urls:
        print(f"[try] {url} (timeout={timeout}s)")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "play.snake-neuro/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
            if len(data) < min_bytes:
                print(f"  too small: {len(data)} bytes")
                continue
            dest.write_bytes(data)
            print(f"[ok] {dest} ({len(data)} bytes)")
            return True
        except Exception as e:
            print(f"  fail: {e}")
    return False


def _write_synthetic_connectome(data_dir: Path) -> None:
    """无官方 parquet 时：仅用 groups.json 神经元构造稀疏合成 W，保证可训。"""
    import numpy as np
    import pandas as pd

    groups_path = data_dir / GROUPS_JSON
    groups = json.loads(groups_path.read_text())
    ids = sorted({int(x) for v in groups.values() for x in v})
    n = len(ids)
    print(f"[synthetic] building connectome over {n} group neurons")

    id_index = {fid: i for i, fid in enumerate(ids)}
    rng = np.random.default_rng(42)

    # Completeness CSV: index = flywire id
    comp = pd.DataFrame({"status": ["synthetic"] * n}, index=ids)
    comp.index.name = "root_id"
    comp.to_csv(data_dir / COMPLETENESS_CSV)

    # Structured edges: sensory -> taste_sez -> motor
    sensory = []
    for k in (
        "threat_lc4_left",
        "threat_lc4_right",
        "reward_sugar_left",
        "reward_sugar_right",
        "bitter",
    ):
        sensory.extend(groups.get(k, []))
    mid = groups.get("taste_sez", [])
    motor = []
    for k in (
        "steer_left",
        "steer_right",
        "motor_forward",
        "walk_p9",
        "escape_gf",
        "escape_mdn",
        "feeding_mn9",
    ):
        motor.extend(groups.get(k, []))

    def idxs(lst):
        return [id_index[i] for i in lst if i in id_index]

    pre, post, w = [], [], []
    for a in idxs(sensory):
        targets = rng.choice(idxs(mid) or idxs(motor), size=min(8, max(1, n // 20)), replace=False)
        for b in targets:
            pre.append(a)
            post.append(b)
            w.append(float(rng.uniform(0.5, 3.0)))
    for a in idxs(mid):
        targets = rng.choice(idxs(motor) or [a], size=min(4, max(1, len(idxs(motor)) or 1)), replace=True)
        for b in targets:
            pre.append(a)
            post.append(int(b))
            w.append(float(rng.uniform(0.5, 2.5)))
    # light recurrence within mid
    mid_i = idxs(mid)
    if len(mid_i) > 2:
        for _ in range(min(2000, len(mid_i) * 3)):
            a, b = rng.choice(mid_i, size=2, replace=False)
            pre.append(int(a))
            post.append(int(b))
            w.append(float(rng.uniform(0.1, 1.0)))

    conn = pd.DataFrame(
        {
            "Presynaptic_Index": pre,
            "Postsynaptic_Index": post,
            "Excitatory x Connectivity": w,
        }
    )
    conn.to_parquet(data_dir / CONNECTIVITY_PARQUET, index=False)
    print(f"[synthetic] wrote {len(conn)} synapses -> {data_dir / CONNECTIVITY_PARQUET}")


def ensure_data(allow_synthetic: bool = True) -> Path:
    data_dir = Path(DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)

    ok_groups = _download(GROUPS_URLS, data_dir / GROUPS_JSON, min_bytes=1000, timeout=30)
    if not ok_groups and not (data_dir / GROUPS_JSON).exists():
        raise RuntimeError("Failed to download groups.json")

    csv_p = data_dir / COMPLETENESS_CSV
    pq_p = data_dir / CONNECTIVITY_PARQUET
    if csv_p.exists() and pq_p.exists() and pq_p.stat().st_size > 1000:
        print(f"[skip] connectome already present ({pq_p.stat().st_size} bytes)")
        return data_dir

    # 大文件短超时；失败则合成。可用 FLY_FORCE_SYNTHETIC=1 跳过官方下载。
    force_synth = os.environ.get("FLY_FORCE_SYNTHETIC", "").strip() in ("1", "true", "True")
    if force_synth:
        ok_csv = ok_pq = False
        print("[info] FLY_FORCE_SYNTHETIC=1 -> skip official connectome download")
    else:
        ok_csv = _download(CSV_URLS, csv_p, min_bytes=10_000, timeout=90)
        ok_pq = _download(PARQUET_URLS, pq_p, min_bytes=50_000_000, timeout=120)

    if not (ok_csv and ok_pq):
        if not allow_synthetic:
            raise RuntimeError("FlyWire data download failed and synthetic fallback disabled")
        print("[warn] official connectome unavailable; building synthetic connectome from groups.json")
        _write_synthetic_connectome(data_dir)

    return data_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description="Download FlyWire data for neuro players")
    parser.add_argument("--no-synthetic", action="store_true", help="fail if download missing")
    args = parser.parse_args(argv)
    path = ensure_data(allow_synthetic=not args.no_synthetic)
    print(f"Data ready under {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
