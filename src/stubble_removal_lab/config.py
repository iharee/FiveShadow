"""Portable manual geometry profiles; all coordinates use oriented RGB pixels."""
from dataclasses import dataclass
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class Profile:
    points: list[list[float]]
    parsing: dict
    mask_strategy: str = "model_resolution"
    review_crop: list[int] | None = None
    source_sha256: str | None = None
    source_size_wh: list[int] | None = None
    schema_version: int = 1
    origin: str = "manual"
    detection: dict | None = None

    @classmethod
    def load(cls, path: Path) -> "Profile":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    @classmethod
    def from_dict(cls, data: dict) -> "Profile":
        if not isinstance(data, dict):
            raise ValueError("配置必须为JSON对象")
        if data.get("schema_version") != 1:
            raise ValueError("配置 schema_version 必须为 1")
        points = data.get("points")
        if (not isinstance(points, list) or len(points) != 3
                or any(not isinstance(p, list) or len(p) != 2 for p in points)):
            raise ValueError("points 需要三个二维坐标：图像左侧嘴角、右侧嘴角、下巴底点")
        if any(not isinstance(v, (int, float)) or not math.isfinite(v)
               for p in points for v in p):
            raise ValueError("points 必须为有限数值")
        parsing = data.get("parsing", {})
        if not isinstance(parsing, dict):
            raise ValueError("parsing 必须为JSON对象")
        crop = parsing.get("crop_xyxy")
        affine = parsing.get("output_to_source_affine")
        if (crop is None) == (affine is None):
            raise ValueError("parsing 需且仅需 crop_xyxy 或 output_to_source_affine")
        if crop is not None:
            if (len(crop) != 4 or any(not isinstance(v, int) for v in crop)
                    or crop[2] <= crop[0] or crop[3] <= crop[1]):
                raise ValueError("parsing.crop_xyxy 必须为有效整数矩形")
        if affine is not None:
            if (len(affine) != 2 or any(len(row) != 3 for row in affine)
                    or any(not isinstance(v, (int, float)) or not math.isfinite(v)
                           for row in affine for v in row)):
                raise ValueError("parsing.output_to_source_affine 必须为有限的2×3矩阵")
        strategy = data.get("mask_strategy", "model_resolution")
        if strategy not in ("model_resolution", "low_resolution_then_resize"):
            raise ValueError("未知 mask_strategy")
        review = data.get("review_crop")
        if review is not None and (len(review) != 4
                or any(not isinstance(v, int) for v in review)
                or review[2] <= review[0] or review[3] <= review[1]):
            raise ValueError("review_crop 必须为有效整数矩形")
        size = data.get("source_size_wh")
        if size is not None and (len(size) != 2 or any(not isinstance(v, int) or v < 1 for v in size)):
            raise ValueError("source_size_wh 必须为正整数宽高")
        return cls(points=points, parsing=parsing, mask_strategy=strategy,
                   review_crop=review, source_sha256=data.get("source_sha256"),
                   source_size_wh=size,origin=data.get("origin","manual"),detection=data.get("detection"))
