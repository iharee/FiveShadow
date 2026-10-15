"""Reusable photo -> parsing -> mask -> restoration -> full-size correction."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image
import torch

from . import __version__
from .assets import asset_root, model_id, require_assets
from .config import Profile
from .engines.parsing import ParsingGraph
from .engines.restoration import RecoveredGraph
from .geometry import add_correction, align_image, parsing_image, restoration_transform
from .masking import semantic_mask
from .photos import open_photo

PALETTE = np.array([[0,0,0],[170,60,160],[240,180,140],[160,100,50],
    [30,150,250],[200,250,250],[60,80,80],[120,50,40],[240,70,130],
    [80,10,30],[255,255,255],[190,140,80],[0,200,160],[220,150,80],
    [200,190,100],[100,70,220],[0,180,255]],np.uint8)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Pipeline:
    """Single-face CPU pipeline shared by automatic/manual CLI and GUI."""

    def __init__(self, model="C", assets: Path | None = None, threads=6):
        if threads < 1:
            raise ValueError("threads 必须为正整数")
        self.model_id = model_id(model)
        self.root = asset_root(assets)
        require_assets(self.root, self.model_id)
        torch.set_num_threads(threads)
        self.threads = threads
        self.parser = ParsingGraph(self.root)
        self.restoration = RecoveredGraph(self.model_id, self.root)
        template = json.loads((self.root/"beard_alignment_template.json").read_text())
        self.template = np.asarray(template["points"], np.float64)
        if self.template.shape != (142,2) or not np.isfinite(self.template).all():
            raise ValueError("无效的142点模板")

    def process(self, source: Path, profile: Profile, output: Path, *, strength=1.0,
                save_intermediates=False, save_diagnostics=False, progress=None) -> Path:
        if not np.isfinite(strength) or not 0 <= strength <= 1:
            raise ValueError("strength 必须在0到1之间；它不等同于原软件UI强度")
        source = source.expanduser().resolve(strict=True)
        output = output.expanduser().resolve()
        if output.exists():
            raise FileExistsError(f"输出目录已经存在，请换一个目录：{output}")
        started = time.monotonic()
        source_hash = sha256(source)
        if profile.source_sha256 and profile.source_sha256 != source_hash:
            raise ValueError("源文件与配置的SHA-256不一致；请为这张照片单独建立配置")
        photo = open_photo(source)
        icc = photo.info.get("icc_profile")
        if profile.source_size_wh and list(photo.size) != profile.source_size_wh:
            raise ValueError("源图尺寸与配置不一致（坐标基于EXIF转正后的图像）")
        for x,y in profile.points:
            if not (0 <= x < photo.width and 0 <= y < photo.height):
                raise ValueError("定位点超出照片范围")
        original = np.asarray(photo)
        output.mkdir(parents=True)
        notify = progress or (lambda message: None)
        report = {"version":__version__, "status":"running", "source":str(source),
                  "source_sha256":source_hash, "model":self.model_id,
                  "strength":strength, "threads":self.threads, "profile":asdict(profile),
                  "limitations":["三点仿射对齐；自动关键点或手工标注，不等于原生142点几何。",
                    "固定选模，不运行原生预筛选。", "CV遮罩与最终合成为研究近似。",
                    "未证明原生逐像素等价，也未证明任意照片上的质量。"]}
        try:
            notify("计算人脸语义分割")
            parse_photo, parse_to_source = parsing_image(photo, profile)
            if save_diagnostics or save_intermediates:
                parse_photo.save(output/"parsing_input.png")
            normalized = np.asarray(parse_photo,np.float32).transpose(2,0,1)[None]
            normalized = ((normalized-np.array([123.675,116.28,103.53],np.float32)[None,:,None,None])
                          / np.array([58.395,57.12,57.375],np.float32)[None,:,None,None])
            probabilities, _ = self.parser.parse(normalized)
            if not np.isfinite(probabilities).all():
                raise ValueError("分割输出含非有限值")
            labels = probabilities[0].argmax(axis=0)
            if save_diagnostics or save_intermediates:
                Image.fromarray((np.asarray(parse_photo)*.5 + PALETTE[labels]*.5).astype(np.uint8)).save(
                    output/"parsing_overlay.png")
            _,_,height,width = self.restoration.graph["nodes"][0]["dims"]
            transform = restoration_transform(profile.points,self.template,width,height)
            notify("构造皮肤、胡须与鼻嘴保护遮罩")
            if profile.mask_strategy == "low_resolution_then_resize":
                low_transform = restoration_transform(profile.points,self.template,512,384)
                low_mask = semantic_mask(probabilities[0],low_transform@parse_to_source,
                                         self.template,512,384)
                mask = cv2.resize(low_mask,(width,height),interpolation=cv2.INTER_LINEAR)
            else:
                mask = semantic_mask(probabilities[0],transform@parse_to_source,
                                     self.template,width,height)
            if save_diagnostics or save_intermediates:
                Image.fromarray(np.rint(mask*255).astype(np.uint8)).save(output/"mask.png")
            aligned = align_image(photo,transform,width,height)
            if save_diagnostics or save_intermediates:
                aligned.save(output/"input_roi.png")
            image = np.asarray(aligned,np.float32).transpose(2,0,1)[None]/127.5-1
            notify(f"运行修复模型 {self.model_id}")
            result, trace = self.restoration.run(image,mask[None,None],capture_alpha2=save_intermediates)
            delta = ((result["2977284499"][0]-image[0])*127.5).transpose(1,2,0)
            if not np.isfinite(delta).all():
                raise ValueError("修复输出含非有限值")
            notify("将局部RGB修正合成回完整照片")
            full = add_correction(original,delta,transform,strength)
            destination = output/"result.png"
            options = {"icc_profile":icc} if icc else {}
            Image.fromarray(full).save(destination,**options)
            (output/"profile.json").write_text(json.dumps(asdict(profile),ensure_ascii=False,indent=2),encoding="utf-8")
            if save_intermediates:
                np.savez_compressed(output/"intermediates.npz",image=image,mask=mask,
                                    delta=delta,probabilities=probabilities,**result)
                (output/"trace.json").write_text(json.dumps(trace,indent=2),encoding="utf-8")
            if sha256(source) != source_hash:
                raise RuntimeError("运行期间输入照片发生变化")
            asset_files = ["dunnite_networks.json", "beard_alignment_template.json",
                f"decoded_models/{self.model_id}.decoded.bin",
                f"recovered_weights/{self.model_id}.float.npz",
                "parsing_models/dunnite_networks.json", "parsing_models/2102828093.decoded.bin"]
            report.update(status="complete",seconds=time.monotonic()-started,
                output=str(destination),output_sha256=sha256(destination),size_wh=list(photo.size),
                source_to_roi=transform.tolist(),parse_to_source=parse_to_source.tolist(),
                mask_mean=float(mask.mean()),delta_abs_mean=float(np.abs(delta).mean()),
                asset_sha256={name:sha256(self.root/name) for name in asset_files})
            return destination
        except Exception as error:
            report.update(status="failed",error=str(error),seconds=time.monotonic()-started)
            raise
        finally:
            (output/"manifest.json").write_text(
                json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")
