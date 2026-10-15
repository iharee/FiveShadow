"""Equal-coordinate visual comparisons without brightness normalization."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps


def make_review(panels: list[tuple[str,Path]], crop, destination: Path, scale=2):
    if destination.exists():
        raise FileExistsError(destination)
    if crop is None:
        raise ValueError("对照图需要配置 review_crop；完整结果不受此限制")
    x0,y0,x1,y1 = crop
    width,height = (x1-x0)*scale,(y1-y0)*scale
    rows = (len(panels)+1)//2
    canvas = Image.new("RGB",(width*2,(height+32)*rows),"white")
    draw = ImageDraw.Draw(canvas)
    source_size = None
    for index,(label,path) in enumerate(panels):
        with Image.open(path) as opened:
            photo = ImageOps.exif_transpose(opened).convert("RGB")
        if source_size is None:
            source_size = photo.size
        if photo.size != source_size:
            raise ValueError(f"对照照片尺寸不同：{path}")
        if x0<0 or y0<0 or x1>photo.width or y1>photo.height:
            raise ValueError("review_crop 超出照片范围")
        region = photo.crop(tuple(crop))
        if scale != 1:
            region = region.resize((width,height),Image.Resampling.LANCZOS)
        x,y = index%2*width,index//2*(height+32)
        draw.text((x+6,y+10),label,fill="black")
        canvas.paste(region,(x,y+32))
    canvas.save(destination)
