"""Desktop launcher and automatic/manual local processing commands."""
import argparse
from dataclasses import asdict
from datetime import datetime
import gc
import json
from pathlib import Path
import sys
from uuid import uuid4

from . import __version__


def build_parser():
    parser=argparse.ArgumentParser(description="本地照片祛胡青：自动定位、手动选区、GUI与命令行")
    parser.add_argument("--version",action="version",version=__version__)
    commands=parser.add_subparsers(dest="command")
    gui=commands.add_parser("gui",help="打开可视化界面")
    gui.add_argument("input",type=Path,nargs="?")
    gui.add_argument("--assets",type=Path)
    models=commands.add_parser("models",help="准备并校验自动定位模型")
    models.add_argument("--directory",type=Path)
    for name,help_text in [("process","处理照片：默认自动定位，也支持手工配置"),
                           ("compare","分别生成B/C和局部对照"),
                           ("detect","仅定位人脸并导出可复用配置")]:
        command=commands.add_parser(name,help=help_text)
        command.add_argument("--input",type=Path,required=True)
        command.add_argument("--output",type=Path,help="新输出目录（detect时是新JSON文件）")
        command.add_argument("--box",type=int,nargs=4,metavar=("X0","Y0","X1","Y1"),help="框住整张脸")
        command.add_argument("--face-index",type=int,default=0,help="从0开始，按脸面积降序")
        if name=="detect":
            continue
        command.add_argument("--profile",type=Path,help="加载此前保存的配置，跳过自动定位")
        command.add_argument("--points",type=float,nargs=6,metavar=("LX","LY","RX","RY","CX","CY"),
                             help="左嘴角、右嘴角、画面中下颌轮廓最低处；需搭配--box")
        command.add_argument("--assets",type=Path)
        command.add_argument("--threads",type=int,default=6)
        command.add_argument("--strength",type=float,default=1.0)
        command.add_argument("--debug",action="store_true",help="保存ROI、遮罩、分割与中间数组")
        if name=="process":
            command.add_argument("--model",choices=["C","B","A","D"],default="C")
        else:
            command.add_argument("--reference",type=Path)
    return parser


def pick_profile(args):
    from .config import Profile
    from .selection import make_profile
    if getattr(args,"profile",None):
        if args.box or getattr(args,"points",None):
            raise ValueError("--profile 与 --box/--points 不能同时使用")
        return Profile.load(args.profile)
    if getattr(args,"points",None):
        if args.box is None:
            raise ValueError("--points 需搭配 --box 指定整张脸的区域")
        points=[args.points[i:i+2] for i in (0,2,4)]
        return make_profile(args.input,args.box,points)
    from .detection import detect_faces
    candidates=detect_faces(args.input,box=args.box,progress=lambda m:print(m,file=sys.stderr,flush=True))
    if not 0<=args.face_index<len(candidates):
        raise ValueError(f"检测到{len(candidates)}张脸，face-index应为0..{len(candidates)-1}")
    chosen=candidates[args.face_index]
    if chosen.profile is None:
        raise ValueError(chosen.message)
    return Profile.from_dict(chosen.profile)


def run(args):
    if args.command in (None,"gui"):
        from .gui import main as gui_main
        return gui_main(input_path=getattr(args,"input",None),assets=getattr(args,"assets",None))
    if args.command=="models":
        from .detection_models import ensure_models
        print(ensure_models(args.directory,lambda m:print(m,flush=True)))
        return 0
    if args.command=="detect":
        from .detection import detect_faces
        result=detect_faces(args.input,box=args.box,progress=lambda m:print(m,file=sys.stderr,flush=True))
        if args.output:
            if not 0<=args.face_index<len(result) or result[args.face_index].profile is None:
                raise ValueError("选中的人脸没有可用关键点，请在界面里手动选区")
            if args.output.exists():
                raise FileExistsError(args.output)
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps(result[args.face_index].profile,ensure_ascii=False,indent=2),encoding="utf-8")
        print(json.dumps([asdict(face) for face in result],ensure_ascii=False,indent=2))
        return 0
    from .pipeline import Pipeline
    from .review import make_review
    if not 0<=args.strength<=1:
        raise ValueError("strength需在0到1之间")
    output=args.output or Path("runs")/f"{args.input.stem}-{datetime.now():%Y%m%d-%H%M%S}-{uuid4().hex[:6]}"
    if output.exists():
        raise FileExistsError(f"输出目录已存在：{output}")
    profile=pick_profile(args)
    if args.command=="process":
        pipeline=Pipeline(args.model,args.assets,args.threads)
        result=pipeline.process(args.input,profile,output,strength=args.strength,
            save_intermediates=args.debug,save_diagnostics=args.debug,
            progress=lambda m:print(m,file=sys.stderr,flush=True))
        print(result)
        return 0
    output.mkdir(parents=True)
    panels=[("Source",args.input)]
    if args.reference:
        panels.append(("Reference",args.reference))
    for model in ("B","C"):
        pipeline=Pipeline(model,args.assets,args.threads)
        result=pipeline.process(args.input,profile,output/model,strength=args.strength,
            save_intermediates=args.debug,save_diagnostics=args.debug,
            progress=lambda m:print(f"[{model}] {m}",file=sys.stderr,flush=True))
        panels.append((f"{model}",result))
        del pipeline
        gc.collect()
    make_review(panels,profile.review_crop,output/"review.png",scale=2)
    print(output/"review.png")
    return 0


def main(argv=None):
    args=build_parser().parse_args(argv)
    try:
        return run(args)
    except KeyboardInterrupt:
        print("已取消",file=sys.stderr)
        return 130
    except Exception as error:
        print(f"错误：{error}",file=sys.stderr)
        return 1


if __name__=="__main__":
    raise SystemExit(main())
