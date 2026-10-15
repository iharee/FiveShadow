"""Isolated GUI worker. One JSON request in, JSON-line progress/results out."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import signal
import sys
import traceback


def emit(kind,**payload):
    print(json.dumps({"type":kind,**payload},ensure_ascii=False),flush=True)


def interrupt(signum,frame):
    raise KeyboardInterrupt


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--request",type=Path,required=True)
    args=parser.parse_args()
    signal.signal(signal.SIGTERM,interrupt)
    try:
        request=json.loads(args.request.read_text(encoding="utf-8"))
        progress=lambda message:emit("progress",message=message)
        if request["task"]=="detect":
            from .detection import detect_faces
            faces=detect_faces(Path(request["source"]),box=request.get("box"),progress=progress)
            emit("result",faces=[asdict(face) for face in faces])
        elif request["task"]=="process":
            from .config import Profile
            from .pipeline import Pipeline
            profile=Profile.from_dict(request["profile"])
            assets=Path(request["assets"]) if request.get("assets") else None
            progress("准备修复模型")
            pipeline=Pipeline(request.get("model","C"),assets,request.get("threads",6))
            result=pipeline.process(Path(request["source"]),profile,Path(request["output"]),
                strength=1,output_format="png",progress=progress)
            emit("result",path=str(result))
        elif request["task"]=="batch":
            from .batch import process_batch
            assets=Path(request["assets"]) if request.get("assets") else None
            report=process_batch(request["inputs"],Path(request["output"]),
                recursive=request.get("recursive",False),model=request.get("model","C"),
                assets=assets,threads=request.get("threads",6),strength=request.get("strength",1),
                output_format=request.get("output_format","same"),quality=request.get("quality",95),
                face_index=request.get("face_index",0),progress=progress)
            emit("result",report=report)
        else:
            raise ValueError("未知后台任务")
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as error:
        emit("error",message=str(error))
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__=="__main__":
    raise SystemExit(main())
