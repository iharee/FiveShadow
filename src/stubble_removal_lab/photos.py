"""Shared still-image decoding, oriented RGB pixels, and result encoding."""
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageCms, ImageOps


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
        photo = ImageOps.exif_transpose(opened)
        alpha = photo.convert("RGBA").getchannel("A") if (
            "A" in photo.getbands() or "transparency" in photo.info) else None
        icc = opened.info.get("icc_profile")
        if photo.mode in ("CMYK", "L", "LA") and icc:
            color = photo.convert("L") if photo.mode == "LA" else photo
            srgb = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))
            photo_rgb = ImageCms.profileToProfile(
                color, ImageCms.ImageCmsProfile(BytesIO(icc)), srgb, outputMode="RGB")
            icc = srgb.tobytes()
        else:
            photo_rgb = photo.convert("RGB")
        exif = photo.getexif()
        # TIFF storage tags describe the old file, not the newly encoded pixels.
        for tag in (256, 257, 258, 259, 262, 273, 274, 277, 278, 279, 284, 317,
                    320, 322, 323, 324, 325, 338, 339, 34675):
            if tag in exif:
                del exif[tag]
        photo_rgb.info = {"source_format": source_format,
                          "source_suffix": path.suffix.lower()}
        if alpha is not None:
            photo_rgb.info["source_alpha"] = alpha
        if icc:
            photo_rgb.info["icc_profile"] = icc
        if exif:
            photo_rgb.info["exif"] = exif.tobytes()
        if opened.info.get("dpi"):
            photo_rgb.info["dpi"] = opened.info["dpi"]
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
            original_suffix.lstrip("."), original_suffix.lstrip(".")) == name:
        suffix = original_suffix
    return encoding, suffix


def save_photo(photo: Image.Image, source: Image.Image, destination: Path, *,
               output_format="same", quality=95):
    """Encode RGB results with source alpha, ICC, EXIF, and DPI where supported."""
    if not 1 <= quality <= 100:
        raise ValueError("quality 必须在1到100之间")
    if photo.size != source.size:
        raise ValueError("结果尺寸与原图不一致")
    encoding, _ = output_encoding(source, output_format)
    result = photo.convert("RGB")
    if encoding in ("BMP", "GIF") and source.info.get("icc_profile"):
        result = ImageCms.profileToProfile(result,
            ImageCms.ImageCmsProfile(BytesIO(source.info["icc_profile"])),
            ImageCms.createProfile("sRGB"),outputMode="RGB")
    alpha = source.info.get("source_alpha")
    if alpha is not None:
        result.putalpha(alpha)
        if encoding in ("JPEG", "BMP"):
            background = Image.new("RGB", result.size, "white")
            background.paste(result, mask=alpha)
            result = background
        elif encoding == "GIF":
            result = result.convert("RGB").quantize(colors=255)
            result.paste(255, mask=alpha.point(lambda value: 255 if value < 128 else 0))
            result.info["transparency"] = 255
    options = {}
    if encoding in ("JPEG", "PNG", "TIFF", "WEBP"):
        for key in ("icc_profile", "exif", "dpi"):
            if source.info.get(key):
                options[key] = source.info[key]
    if encoding in ("JPEG", "WEBP"):
        options["quality"] = quality
    if encoding == "JPEG":
        options["subsampling"] = 0
    if encoding == "WEBP":
        options["exact"] = True
    if encoding == "TIFF":
        options["compression"] = "tiff_deflate"
    result.save(destination, format=encoding, **options)
