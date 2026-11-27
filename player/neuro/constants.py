"""果蝇脑 player 共享常量。"""
import os
from pathlib import Path

NEURO_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("FLY_DATA_DIR", NEURO_DIR / "data"))

STIM_GROUPS = [
    "threat_lc4_left",
    "threat_lc4_right",
    "reward_sugar_left",
    "reward_sugar_right",
    "bitter",
]
STIM_DIM = len(STIM_GROUPS)

READOUT_ORDER = [
    "steer_left",
    "steer_right",
    "forward",
    "escape",
    "feeding",
    "threat",
    "reward",
]
READOUT_DIM = len(READOUT_ORDER)

# Shiu / fly-brain-snake
W_SCALE = 0.275
LEARN_RATE = 0.08
MAX_GAIN = 3.0
ELIG_DECAY = 0.9  # per decision step (rate-coded approx)

# 子图
DEFAULT_HOPS = 2
DEFAULT_MAX_NEURONS = 5000

# 训练预算（可用环境变量覆盖）
NUM_GAMES_DEFAULT = int(os.environ.get("FLY_NUM_GAMES", "100000"))
N_ENVS_DEFAULT = int(os.environ.get("FLY_N_ENVS", "8"))
# 训练时单局截断（更短 → 同 timestep 下更多局）
STEP_LIMIT_DEFAULT = int(os.environ.get("FLY_STEP_LIMIT", "200"))
# 默认 1e7 steps ≈ 计划中的 1e5 局×100 step；可用 FLY_TOTAL_TIMESTEPS 覆盖
TOTAL_TIMESTEPS_DEFAULT = int(os.environ.get("FLY_TOTAL_TIMESTEPS", "10000000"))

MODEL_PATHS = {
    "fixed": "mdl/neuro/fixed/fly_fixed.zip",
    "plastic": "mdl/neuro/plastic/fly_plastic.zip",
    "enc": "mdl/neuro/enc/fly_enc.zip",
    "snake": "mdl/neuro/snake/fly_snake.zip",
}

COMPLETENESS_CSV = "2025_Completeness_783.csv"
CONNECTIVITY_PARQUET = "2025_Connectivity_783.parquet"
GROUPS_JSON = "groups.json"
SUBGRAPH_CACHE = "subgraph_cache.npz"
