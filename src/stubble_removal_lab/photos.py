"""Consistent EXIF orientation and RGB handling for CLI, detection and GUI."""
from pathlib import Path
from PIL import Image, ImageOps


def open_photo(path: Path) -> Image.Image:
    with Image.open(path) as opened:
        if opened.mode not in ("RGB", "RGBA", "L"):
            raise ValueError("请使用8位RGB照片（JPG/PNG）；当前不支持RAW、16位或CMYK")
        photo = ImageOps.exif_transpose(opened).convert("RGB")
        if opened.info.get("icc_profile"):
            photo.info["icc_profile"] = opened.info["icc_profile"]
        return photo
