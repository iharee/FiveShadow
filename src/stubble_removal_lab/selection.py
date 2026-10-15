"""Shared manual/automatic face selections in original oriented pixel units."""
import hashlib
import math
from pathlib import Path

from .config import Profile
from .photos import open_photo


def source_hash(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):
            digest.update(chunk)
    return digest.hexdigest()


def square_box(box, size, padding=1.0):
    x0,y0,x1,y1 = box
    if any(not math.isfinite(v) for v in box) or x1<=x0 or y1<=y0:
        raise ValueError("请选择一个有效的人脸区域")
    width,height = size
    side = min(max(x1-x0,y1-y0)*padding,width,height)
    side = max(2,int(math.ceil(side)))
    x = min(max(0,int(round((x0+x1-side)/2))),width-side)
    y = min(max(0,int(round((y0+y1-side)/2))),height-side)
    return [x,y,x+side,y+side]


def make_profile(source: Path, box, points, *, origin="manual", detection=None):
    photo = open_photo(source)
    if len(points) != 3:
        raise ValueError("请依次标记画面左嘴角、右嘴角、下颌轮廓最低处")
    for x,y in points:
        if not (math.isfinite(x) and math.isfinite(y) and 0<=x<photo.width and 0<=y<photo.height):
            raise ValueError("定位点需在照片范围内")
    a,b,c=points
    area=(b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    if abs(area)<1:
        raise ValueError("三个点太接近或共线，请重新定位")
    crop=square_box(box,photo.size)
    if any(not (crop[0]<=x<crop[2] and crop[1]<=y<crop[3]) for x,y in points):
        raise ValueError("三个定位点需位于所框脸部区域内，请扩大或调整脸框")
    x0=min(p[0] for p in points);x1=max(p[0] for p in points)
    y0=min(p[1] for p in points);y1=max(p[1] for p in points)
    margin=max(x1-x0,y1-y0)*.55
    review=[max(0,int(x0-margin)),max(0,int(y0-margin)),
            min(photo.width,int(x1+margin+1)),min(photo.height,int(y1+margin+1))]
    return Profile.from_dict({"schema_version":1,"points":[list(p) for p in points],
        "parsing":{"crop_xyxy":crop},"mask_strategy":"model_resolution", "review_crop":review,
        "source_sha256":source_hash(source),"source_size_wh":list(photo.size),
        "origin":origin,"detection":detection})
