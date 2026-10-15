"""Pinned official detection assets; only model bytes are downloaded."""
import hashlib
import os
from pathlib import Path
import tempfile
from urllib.request import Request, urlopen

from .assets import PROJECT_ROOT

ASSETS = {
    "face_detection_yunet_2023mar.onnx": (
        "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"),
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"),
    "YuNet-LICENSE.txt": (
        "https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/face_detection_yunet/LICENSE",
        "c83b8120c50ccbd4c4f96edf53141bdd566ebb8f8e9227e415326aa1b1aba958"),
}


def ensure_models(directory: Path | None = None, progress=None):
    root = Path(directory or os.environ.get("STUBBLE_DETECTION_DIR")
                or PROJECT_ROOT/"models/detection").expanduser().resolve()
    root.mkdir(parents=True,exist_ok=True)
    notify = progress or (lambda message:None)
    for name,(url,expected) in ASSETS.items():
        destination = root/name
        if destination.exists():
            if hashlib.sha256(destination.read_bytes()).hexdigest()!=expected:
                raise ValueError(f"定位模型校验不一致：{destination}；请移走该文件后重试")
            continue
        notify(f"首次准备定位模型：下载 {name}")
        with tempfile.NamedTemporaryFile(dir=root,suffix=".download",delete=False) as stream:
            temporary = Path(stream.name)
            try:
                request = Request(url,headers={"User-Agent":"stubble-removal-local/0.2"})
                with urlopen(request,timeout=45) as response:
                    while chunk:=response.read(1024*1024):
                        stream.write(chunk)
                stream.flush()
                if hashlib.sha256(temporary.read_bytes()).hexdigest()!=expected:
                    raise ValueError(f"下载模型校验失败：{name}")
                stream.close()
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
    return root
