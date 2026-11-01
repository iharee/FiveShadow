"""Desktop launcher and automatic/manual local processing commands."""
import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import sys

from . import __version__
from .paths import unique_path
from .photos import FORMAT_CHOICES


class HelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    pass


def build_parser():
    parser=argparse.ArgumentParser(description="本地照片祛胡青：自动定位、手动选区、GUI与命令行",
        formatter_class=HelpFormatter,epilog="不带子命令会打开GUI。非交互调用请指定process、batch、compare或detect。")
    parser.add_argument("--version",action="version",version=__version__)
    commands=parser.add_subparsers(dest="command")
    gui=commands.add_parser("gui",help="打开可视化界面")
    gui.add_argument("input",type=Path,nargs="?")
    gui.add_argument("--assets",type=Path)
    models=commands.add_parser("models",help="准备并校验自动定位模型")
    models.add_argument("--directory",type=Path)
    for name,help_text in [("process","处理照片：默认自动定位，也支持手工配置"),
                           ("compare","分别生成B/C和局部对照"),
                           ("detect","仅定位人脸并导出可复用配置"),
                           ("batch","批量处理目录或多个文件，逐图自动定位，单张失败后继续")]:
        description="支持8位JPG/JPEG、PNG、TIFF、WebP、BMP和静态GIF；不支持动画、多页或高位深图片。"
        epilog="进度与错误写stderr；成功退出0，执行失败1，参数错误2，取消130。"
        if name=="batch":
            epilog+=("\nstdout为JSON汇总，输出目录内保存batch.json和逐图结果目录；任一图片失败退出1。"
                     "\n示例：stubble batch --input photos/ a.jpg --recursive --output runs/batch --format same")
        else:
            description+="\n坐标使用EXIF转正后原图像素，左上角为原点，X向右、Y向下；左右按画面方向。"
            if name=="detect":
                epilog+="\nstdout为所有候选脸的JSON数组；--output保存选中脸的单个profile对象，不能用数组代替profile。"
            else:
                epilog+=("\n全自动：不传定位参数；框内自动：只传--box；手动：--box加--points。"
                         "\n--profile与--box/--points互斥；手动或profile模式忽略--face-index。"
                         "\nstdout为结果路径；输出目录保留profile.json和manifest.json。")
        if name!="detect":
            epilog+="\n默认保留原文件名（转换格式时调整扩展名）；尽量保留元数据与文件属性。调整/丢失信息记录在manifest.json及batch.json。"
        command=commands.add_parser(name,help=help_text,description=description,
                                   epilog=epilog,formatter_class=HelpFormatter)
        if name=="batch":
            command.add_argument("--input",type=Path,nargs="+",required=True,help="一个或多个图片文件/目录，按路径去重")
            command.add_argument("--recursive",action="store_true",help="递归搜索目录内的图片")
        else:
            command.add_argument("--input",type=Path,required=True,help="输入图片，不覆盖原文件")
            command.add_argument("--box",type=int,nargs=4,metavar=("X0","Y0","X1","Y1"),help="整张脸的左、上、右、下边界（整数）")
        command.add_argument("--output",type=Path,help="JSON配置文件；已存在则覆盖" if name=="detect" else "输出目录；已存在则自动改用带_(1)编号的目录；省略则用runs下的名称")
        command.add_argument("--face-index",type=int,default=0,help="从0开始，按脸面积降序；仅用于自动定位")
        if name=="detect":
            continue
        if name!="batch":
            command.add_argument("--profile",type=Path,help="加载同一原图的配置，跳过自动定位")
            command.add_argument("--points",type=float,nargs=6,metavar=("LX","LY","RX","RY","CX","CY"),
                                 help="画面左嘴角、右嘴角、下颌轮廓最低处；允许小数，需搭配--box")
        command.add_argument("--assets",type=Path,help="修复模型目录，也可通过STUBBLE_MODEL_DIR配置")
        command.add_argument("--threads",type=int,default=6,help="CPU线程数，必须为正整数")
        command.add_argument("--strength",type=float,default=1.0,help="修复强度，范围0到1")
        command.add_argument("--format",type=str.lower,choices=FORMAT_CHOICES,default="same",
                             help="结果图片格式；same保持输入格式。JPEG/BMP不支持透明度，转为白底；GIF限制为256色")
        command.add_argument("--quality",type=int,default=95,help="JPEG/WebP编码质量，范围1到100；JPEG会重新压缩")
        command.add_argument("--debug",action="store_true",help="保存ROI、遮罩、分割与中间数组")
        if name in ("process","batch"):
            command.add_argument("--model",choices=["C","B","A","D"],default="C",help="修复模型")
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
    from .detection import detect_profile
    return detect_profile(args.input,box=args.box,face_index=args.face_index,
                          progress=lambda m:print(m,file=sys.stderr,flush=True))


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
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps(result[args.face_index].profile,ensure_ascii=False,indent=2),encoding="utf-8")
        print(json.dumps([asdict(face) for face in result],ensure_ascii=False,indent=2))
        return 0
    if args.command=="batch":
        from .batch import process_batch
        output=args.output or Path("runs")/"batch"
        report=process_batch(args.input,output,recursive=args.recursive,model=args.model,
            assets=args.assets,threads=args.threads,strength=args.strength,
            output_format=args.format,quality=args.quality,face_index=args.face_index,
            debug=args.debug,progress=lambda m:print(m,file=sys.stderr,flush=True))
        print(json.dumps(report,ensure_ascii=False,indent=2))
        return 1 if report["failed"] else 0
    from .pipeline import Pipeline
    from .review import make_review
    if not 0<=args.strength<=1:
        raise ValueError("strength需在0到1之间")
    if not 1<=args.quality<=100:
        raise ValueError("quality必须在1到100之间")
    output=unique_path(args.output or Path("runs")/args.input.stem)
    profile=pick_profile(args)
    if args.command=="process":
        pipeline=Pipeline(args.model,args.assets,args.threads)
        result=pipeline.process(args.input,profile,output,strength=args.strength,
            output_format=args.format,quality=args.quality,
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
            output_format=args.format,quality=args.quality,
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
