"""Explicit filesystem boundary for locally supplied research assets."""
import os
from pathlib import Path

MODELS = {"A": "4083080210", "B": "195031232", "C": "3378116080", "D": "2127140479"}
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def model_id(name: str) -> str:
    value = MODELS.get(name.upper(), name)
    if value not in MODELS.values():
        raise ValueError(f"未知修复模型：{name}")
    return value


def asset_root(path: Path | None = None) -> Path:
    value = path or os.environ.get("STUBBLE_MODEL_DIR") or PROJECT_ROOT / "models/local"
    return Path(value).expanduser().resolve()


def require_assets(root: Path, restoration_id: str) -> None:
    required = [
        "dunnite_networks.json", "beard_alignment_template.json",
        f"decoded_models/{restoration_id}.decoded.bin",
        f"recovered_weights/{restoration_id}.float.npz",
        "parsing_models/dunnite_networks.json",
        "parsing_models/2102828093.decoded.bin",
    ]
    missing = [str(root / item) for item in required if not (root / item).is_file()]
    if missing:
        raise FileNotFoundError(
            "缺少本地研究资产：\n" + "\n".join(missing)
            + "\n参见 models/README.md；程序不会自动下载模型资产。"
        )
