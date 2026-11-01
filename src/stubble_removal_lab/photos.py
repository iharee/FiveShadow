"""Shared still-image decoding, oriented RGB pixels, and result encoding."""
from io import BytesIO
import hashlib
import os
from pathlib import Path
import tempfile

from PIL import Image, ImageCms, ImageOps

from .metadata import capture_metadata, encoding_metadata, normalized_exif, restore_file_attributes


FORMATS = {"png": "PNG", "jpg": "JPEG", "tiff": "TIFF", "webp": "WEBP",
           "bmp": "BMP", "gif": "GIF"}
ALIASES = {"jpeg": "jpg", "tif": "tiff"}
FORMAT_CHOICES = ("same", "png", "jpg", "jpeg", "tiff", "tif", "webp", "bmp", "gif")
PHOTO_EXTENSIONS = frozenset("." + name for name in (*FORMATS, *ALIASES))
PHOTO_FILTERS = {
    "png": "PNG (*.png)", "jpg": "JPEG (*.jpg *.jpeg)",
    "tiff": "TIFF (*.tif *.tiff)", "webp": "WebP (*.webp)",
    "bmp": "BMP (*.bmp)", "gif": "GIF (*.gif)",
}


def open_photo(path: Path) -> Image.Image:
    """Return oriented RGB; retain alpha and supported metadata for export."""
    path = Path(path).expanduser()
    file_stat=path.stat()
    with Image.open(path) as opened:
        if opened.format not in FORMATS.values():
            raise ValueError("支持 JPG/JPEG、PNG、TIFF、WebP、BMP 和静态 GIF")
        if getattr(opened, "n_frames", 1) != 1:
            raise ValueError("暂不支持动画或多页图片，请先导出为单张图片")
        bits = opened.tag_v2.get(258, (8,)) if opened.format == "TIFF" else (8,)
        if isinstance(bits, int):
            bits = (bits,)
        if opened.format == "PNG":
            with path.open("rb") as stream:
                bits = (stream.read(25)[24],)
        if (max(bits) > 8 or opened.mode in ("I", "F")
                or opened.mode.startswith("I;16")):
            raise ValueError("暂不支持16位或浮点图片；请先转换为8位图片")
        source_format = opened.format
        metadata=capture_metadata(opened,path,file_stat)
        photo = ImageOps.exif_transpose(opened)
        alpha = photo.convert("RGBA").getchannel("A") if (
            "A" in photo.getbands() or "transparency" in photo.info) else None
        icc = opened.info.get("icc_profile")
        color_changed=photo.mode in ("CMYK", "L", "LA") and bool(icc)
        if color_changed:
            color = photo.convert("L") if photo.mode == "LA" else photo
            srgb = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))
            photo_rgb = ImageCms.profileToProfile(
                color, ImageCms.ImageCmsProfile(BytesIO(icc)), srgb, outputMode="RGB")
            icc = srgb.tobytes()
        else:
            photo_rgb = photo.convert("RGB")
        exif=normalized_exif(metadata,photo_rgb.size,source_format,color_changed)
        photo_rgb.info = {"source_format": source_format,
                          "source_suffix": path.suffix,"source_name":path.name,
                          "source_metadata":metadata,"color_changed":color_changed}
        if color_changed:
            metadata["adjusted"].append("颜色已转换为sRGB，ICC配置相应更新")
        if alpha is not None:
            photo_rgb.info["source_alpha"] = alpha
        if icc:
            photo_rgb.info["icc_profile"] = icc
        if exif:
            photo_rgb.info["exif"] = exif
        if opened.info.get("dpi"):
            photo_rgb.info["dpi"] = opened.info["dpi"]
            if metadata["orientation"] in (5,6,7,8):
                photo_rgb.info["dpi"]=tuple(reversed(photo_rgb.info["dpi"]))
        return photo_rgb


def output_encoding(photo: Image.Image, output_format="same") -> tuple[str, str]:
    """Resolve a supported Pillow encoder and result suffix before inference."""
    name = ALIASES.get(output_format.lower(), output_format.lower())
    if name == "same":
        encoding = photo.info.get("source_format", "PNG")
        name = next((key for key, value in FORMATS.items() if value == encoding), None)
    if name not in FORMATS:
        raise ValueError(f"不支持的输出格式：{output_format}")
    Image.init()
    encoding = FORMATS[name]
    if encoding not in Image.SAVE:
        raise ValueError(f"当前 Pillow 未安装 {encoding} 编码支持")
    suffix = "." + name
    original_suffix = photo.info.get("source_suffix", "")
    if output_format.lower() == "same" and ALIASES.get(
            original_suffix.lower().lstrip("."), original_suffix.lower().lstrip(".")) == name:
        suffix = original_suffix
    return encoding, suffix


def photo_filename(source: Image.Image, output_format="same") -> str:
    _,suffix=output_encoding(source,output_format)
    return Path(source.info.get("source_name","result")).stem+suffix


def save_photo(photo: Image.Image, source: Image.Image, destination: Path, *,
               output_format="same", quality=95):
    """Save a result and return its metadata/attribute preservation report."""
    destination=Path(destination).expanduser()
    original=source.info.get("source_metadata",{}).get("path")
    if original and (destination.resolve()==Path(original)
                     or (destination.exists() and os.path.samefile(destination,original))):
        raise ValueError("请选择其他输出目录或文件名，保留原图")
    if not 1 <= quality <= 100:
        raise ValueError("quality 必须在1到100之间")
    if photo.size != source.size:
        raise ValueError("结果尺寸与原图不一致")
    encoding, _ = output_encoding(source, output_format)
    options,report=encoding_metadata(source,encoding)
    if destination.name==source.info.get("source_name"):
        report["preserved"].append("原文件名")
    elif source.info.get("source_name"):
        report["adjusted"].append("文件名或扩展名按输出选择调整")
    result = photo.convert("RGB")
    if encoding in ("BMP", "GIF") and source.info.get("icc_profile"):
        result = ImageCms.profileToProfile(result,
            ImageCms.ImageCmsProfile(BytesIO(source.info["icc_profile"])),
            ImageCms.createProfile("sRGB"),outputMode="RGB")
        report["adjusted"].append("目标格式无法嵌入ICC，像素转换为sRGB")
    alpha = source.info.get("source_alpha")
    if alpha is not None:
        result.putalpha(alpha)
        if encoding in ("JPEG", "BMP"):
            background = Image.new("RGB", result.size, "white")
            background.paste(result, mask=alpha)
            result = background
            report["adjusted"].append("透明度已合成到白色背景（目标格式不支持）")
        elif encoding == "GIF":
            result = result.convert("RGB").quantize(colors=255)
            result.paste(255, mask=alpha.point(lambda value: 255 if value < 128 else 0))
            result.info["transparency"] = 255
            report["adjusted"].append("GIF透明度按128阈值转为二值，颜色量化为调色板")
    if encoding in ("JPEG", "WEBP"):
        options["quality"] = quality
    if encoding == "JPEG":
        options["subsampling"] = 0
    if encoding == "WEBP":
        options["exact"] = True
    if encoding == "TIFF":
        options["compression"] = "tiff_deflate"
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent,suffix=destination.suffix,delete=False) as stream:
            temporary=Path(stream.name)
        result.save(temporary,format=encoding,**options)
        # Hash before restoring timestamps; reading the exported file can change atime.
        with temporary.open("rb") as stream:
            report["sha256"]=hashlib.file_digest(stream,"sha256").hexdigest()
        temporary.replace(destination)
        restore_file_attributes(source,destination,report,encoding)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    for key in ("preserved","adjusted","omitted","warnings"):
        report[key]=list(dict.fromkeys(report[key]))
    return report
