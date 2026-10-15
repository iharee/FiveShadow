"""Shared sequential batch processing for the CLI and desktop worker."""
import json
from pathlib import Path

from .paths import unique_path
from .photos import PHOTO_EXTENSIONS


def collect_photos(inputs, *, recursive=False):
    sources=[]
    seen=set()
    for value in inputs:
        path=Path(value).expanduser().resolve()
        if path.is_dir():
            paths=sorted(path.rglob("*") if recursive else path.iterdir())
            paths=[item for item in paths if item.is_file()
                   and item.suffix.lower() in PHOTO_EXTENSIONS]
        else:
            paths=[path]
        for item in paths:
            item=item.resolve()
            if item not in seen:
                seen.add(item)
                sources.append(item)
    if not sources:
        raise ValueError("没有找到支持的图片：JPG/JPEG、PNG、TIFF、WebP、BMP、静态GIF")
    return sources


def process_batch(inputs, output: Path, *, recursive=False, model="C", assets=None,
                  threads=6, strength=1.0, output_format="same", quality=95,
                  face_index=0, debug=False, progress=None):
    from .detection import detect_profile
    from .pipeline import Pipeline
    if not 0<=strength<=1:
        raise ValueError("strength需在0到1之间")
    if not 1<=quality<=100:
        raise ValueError("quality必须在1到100之间")
    if face_index<0:
        raise ValueError("face-index必须从0开始")
    output=unique_path(Path(output).expanduser().resolve())
    sources=collect_photos(inputs,recursive=recursive)
    notify=progress or (lambda message:None)
    notify("准备修复模型")
    pipeline=Pipeline(model,assets,threads)
    output.mkdir(parents=True)
    report={"status":"running", "total":len(sources), "succeeded":0, "failed":0,
            "output":str(output), "model":model, "strength":strength,
            "output_format":output_format, "quality":quality, "items":[]}
    summary=output/"batch.json"

    def checkpoint():
        temporary=output/".batch.json.tmp"
        temporary.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
        temporary.replace(summary)

    checkpoint()
    try:
        for index,source in enumerate(sources,1):
            directory=output/f"{index:04d}-{source.stem[:48]}{source.suffix.lower()}"
            item={"source":str(source), "directory":str(directory), "status":"running"}
            report["items"].append(item)
            checkpoint()
            label=f"[{index}/{len(sources)}] {source.name}"
            notify(label)
            try:
                profile=detect_profile(source,face_index=face_index,
                    progress=lambda message:notify(f"{label} · {message}"))
                result=pipeline.process(source,profile,directory,strength=strength,
                    output_format=output_format,quality=quality,
                    save_intermediates=debug,save_diagnostics=debug,
                    progress=lambda message:notify(f"{label} · {message}"))
                item.update(status="complete",output=str(result))
                report["succeeded"]+=1
                notify(f"{label} · 已完成")
            except Exception as error:
                item.update(status="failed",error=str(error))
                report["failed"]+=1
                notify(f"{label} · 失败：{error}")
            checkpoint()
        report["status"]="complete" if not report["failed"] else "partial"
    except KeyboardInterrupt:
        report["status"]="cancelled"
        if report["items"] and report["items"][-1]["status"]=="running":
            report["items"][-1]["status"]="cancelled"
        raise
    finally:
        checkpoint()
    return report
