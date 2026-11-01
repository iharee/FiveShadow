"""Best-effort photo metadata transfer, with explicit preservation reports."""
import ctypes
import os
import stat
import struct
import sys
from xml.etree import ElementTree

from PIL import ExifTags, Image, PngImagePlugin


STORAGE_TAGS={258,259,262,273,277,278,279,284,317,320,322,323,324,325,338,339,
              513,514,700,33723,34377,34675}
XMP_KEY="XML:com.adobe.xmp"


def capture_metadata(opened, path, file_stat):
    metadata={"stat":file_stat,"path":str(path.resolve()),"orientation":opened.getexif().get(274,1),
              "adjusted":[],"omitted":[],"warnings":[],"xattrs":{},"extra":[],
              "text":{},"chunks":[],"xmp":opened.info.get("xmp"),
              "comment":opened.info.get("comment"),"iptc":None,"photoshop":None}
    if hasattr(os,"listxattr"):
        try:
            for name in os.listxattr(path):
                try:
                    metadata["xattrs"][name]=os.getxattr(path,name)
                except OSError as error:
                    metadata["warnings"].append(f"无法读取扩展属性 {name}：{error}")
        except OSError as error:
            metadata["warnings"].append(f"无法读取文件扩展属性：{error}")
    raw=opened.info.get("exif")
    if raw is None and opened.getexif():
        raw=opened.getexif().tobytes()
    metadata["raw_exif"]=raw
    metadata["thumbnail"]=False
    if raw:
        probe=Image.Exif()
        probe.load(raw)
        metadata["thumbnail"]=bool(probe.get_ifd(ExifTags.IFD.IFD1))
    if opened.format=="PNG":
        opened.load()
        metadata["text"]=dict(opened.text)
        metadata["xmp"]=opened.info.get("xmp") or metadata["text"].get(XMP_KEY)
        metadata["chunks"]=list(opened.private_chunks)
        metadata["png_color"]={key:opened.info[key] for key in ("gamma","chromaticity","srgb") if key in opened.info}
    if opened.format=="TIFF":
        metadata["iptc"]=opened.tag_v2.get(33723)
        metadata["photoshop"]=opened.tag_v2.get(34377)
        if metadata["photoshop"]:
            metadata["photoshop"],iptc=_photoshop(metadata["photoshop"],metadata)
            metadata["iptc"]=metadata["iptc"] or iptc
    if opened.format=="GIF" and opened.info.get("extension"):
        metadata["omitted"].append("GIF应用扩展（未验证其与新图像的兼容性）")
    for marker,data in getattr(opened,"applist",[]):
        if marker=="COM":
            metadata["extra"].append((marker,data))
        elif marker=="APP13" and data.startswith(b"Photoshop 3.0\0"):
            cleaned,iptc=_photoshop(data[14:],metadata)
            metadata["photoshop"]=cleaned
            metadata["iptc"]=iptc
            metadata["extra"].append((marker,b"Photoshop 3.0\0"+cleaned))
        elif marker=="APP0":
            if data.startswith(b"JFXX\0") or (data.startswith(b"JFIF\0") and len(data)>14):
                metadata["adjusted"].append("JPEG内嵌预览已移除，避免显示处理前的图片")
        elif marker=="APP14":
            metadata["adjusted"].append("JPEG色彩编码标记按输出重建")
        elif marker=="APP2" and data.startswith(b"MPF\0"):
            metadata["omitted"].append("MPF/HDR辅助图索引（原索引不适用于修复后的图像）")
        elif marker=="APP1" and data.startswith((b"Exif\0",b"http://ns.adobe.com/xap/1.0/\0")):
            continue
        elif marker=="APP1" and data.startswith(b"http://ns.adobe.com/xmp/extension/\0"):
            metadata["omitted"].append("扩展XMP分片未迁移（不能验证其中的预览及跨分片引用）")
        elif marker=="APP2" and data.startswith(b"ICC_PROFILE\0"):
            continue
        else:
            metadata["extra"].append((marker,data))
            metadata["warnings"].append(f"JPEG私有段 {marker} 按原字节保留，厂商含义及内部引用未验证")
    return metadata


def _photoshop(data, metadata):
    """Keep Photoshop/IPTC resources, excluding stale embedded thumbnails."""
    data=bytes(data)
    position=0
    chunks=[]
    iptc=None
    while position<len(data):
        start=position
        if data[position:position+4]!=b"8BIM" or position+7>len(data):
            metadata["omitted"].append("无法安全解析的Photoshop资源")
            break
        resource=int.from_bytes(data[position+4:position+6],"big")
        name_length=data[position+6]+1
        position+=6+name_length+(name_length%2)
        if position+4>len(data):
            metadata["omitted"].append("不完整的Photoshop资源")
            break
        length=int.from_bytes(data[position:position+4],"big")
        position+=4
        payload=data[position:position+length]
        position+=length+(length%2)
        if len(payload)!=length:
            metadata["omitted"].append("不完整的Photoshop资源")
            break
        if resource in (0x0409,0x040C):
            metadata["adjusted"].append("Photoshop内嵌预览已移除")
        elif resource==0x0424:
            metadata["xmp"]=metadata["xmp"] or payload
        else:
            chunks.append(data[start:position])
            if resource==0x0404:
                iptc=payload
    return b"".join(chunks),iptc


def normalized_exif(metadata, size, source_format, color_changed):
    raw=metadata.get("raw_exif")
    if not raw:
        return None
    exif=Image.Exif()
    exif.load(raw)
    changed=metadata["thumbnail"] or metadata["orientation"]!=1 or source_format=="TIFF" or color_changed
    if not changed:
        return raw
    if metadata["thumbnail"]:
        metadata["adjusted"].append("EXIF内嵌缩略图已移除，避免显示处理前的图片")
    if metadata["orientation"]!=1:
        exif.pop(274,None)
        metadata["adjusted"].append("EXIF方向已按转正后的像素更新")
        if metadata["orientation"] in (5,6,7,8) and 282 in exif and 283 in exif:
            exif[282],exif[283]=exif[283],exif[282]
    if source_format=="TIFF":
        for tag in STORAGE_TAGS:
            exif.pop(tag,None)
        metadata["adjusted"].append("TIFF存储布局标签由输出编码器重建")
    for tag,value in ((256,size[0]),(257,size[1])):
        if tag in exif:
            exif[tag]=value
    if ExifTags.IFD.Exif in exif:
        nested=exif.get_ifd(ExifTags.IFD.Exif)
        for tag,value in ((40962,size[0]),(40963,size[1])):
            if tag in nested:
                nested[tag]=value
        if color_changed and 40961 in nested:
            nested[40961]=1
        if ExifTags.IFD.MakerNote in nested:
            metadata["warnings"].append("MakerNote已保留，但重写EXIF后其厂商私有偏移未能验证")
    return exif.tobytes()


def _normalized_xmp(raw, source, report):
    if not raw:
        return None
    if isinstance(raw,str):
        raw=raw.encode("utf-8")
    try:
        root=ElementTree.fromstring(raw)
    except ElementTree.ParseError:
        if source.info.get("source_metadata",{}).get("orientation",1)!=1 or source.info.get("color_changed"):
            report["omitted"].append("无法更新方向的非标准XMP")
            return None
        report["warnings"].append("XMP按原字节保留，未能解析和验证其内容")
        return raw
    changed=False
    risky=("hdrgm","hdr-gain-map","GCamera","Container/","GainMap","MotionPhoto","MicroVideo","HasExtendedXMP")
    for parent in list(root.iter()):
        for key in list(parent.attrib):
            if any(value.lower() in key.lower() for value in risky):
                del parent.attrib[key]
                changed=True
        for child in list(parent):
            if any(value.lower() in child.tag.lower() for value in (*risky,"Thumbnails")):
                parent.remove(child)
                changed=True
    if changed:
        report["adjusted"].append("XMP中旧缩略图、辅助图及未迁移扩展分片的引用已移除")
    updates={"Orientation":"1","ImageWidth":str(source.width),"ImageLength":str(source.height),
             "PixelXDimension":str(source.width),"PixelYDimension":str(source.height)}
    if source.info.get("color_changed"):
        updates["ColorSpace"]="1"
    for element in root.iter():
        for key,value in list(element.attrib.items()):
            name=key.rsplit("}",1)[-1]
            if name in updates and value!=updates[name]:
                element.set(key,updates[name])
                changed=True
        name=element.tag.rsplit("}",1)[-1]
        if name in updates and element.text!=updates[name]:
            element.text=updates[name]
            changed=True
    if changed:
        report["adjusted"].append("XMP图像属性已与输出同步")
        return ElementTree.tostring(root,encoding="utf-8")
    return raw


def encoding_metadata(source, encoding):
    metadata=source.info.get("source_metadata",{})
    report={key:list(metadata.get(key,[])) for key in ("preserved","adjusted","omitted","warnings")}
    report["limitations"]=[
        "文件创建时间和扩展属性的复制能力依赖平台及文件系统。",
        "文件标识、系统更改时间、所有权及ACL不主动迁移。",
        "无法识别的私有结构不能保证完整迁移；报告仅覆盖已识别的信息。",
    ]
    options={}
    for key,label,formats in (
        ("exif","EXIF（拍摄时间、相机参数、GPS等原有字段）",("JPEG","PNG","TIFF","WEBP")),
        ("icc_profile","ICC色彩配置",("JPEG","PNG","TIFF","WEBP")),
        ("dpi","分辨率/DPI",("JPEG","PNG","TIFF","BMP")),
    ):
        if source.info.get(key):
            if encoding in formats:
                value=source.info[key]
                if key=="exif" and (encoding=="TIFF" or encoding!=source.info.get("source_format")):
                    exif=Image.Exif()
                    exif.load(value)
                    removed=[tag for tag in STORAGE_TAGS if tag in exif]
                    for tag in removed:
                        exif.pop(tag,None)
                    if removed:
                        value=exif.tobytes()
                        report["adjusted"].append("EXIF中的原格式存储标签已移除，由目标编码器重建")
                if key=="exif" and encoding=="JPEG" and len(value)>65533:
                    report["omitted"].append("EXIF超过JPEG单段容量，未写入")
                    continue
                options[key]=value
                report["preserved"].append(label)
            else:
                report["omitted"].append(f"{label}：{encoding}编码器不支持")
    xmp=_normalized_xmp(metadata.get("xmp"),source,report)
    if xmp:
        if encoding in ("JPEG","WEBP"):
            if encoding=="JPEG" and len(xmp)>65504:
                report["omitted"].append("XMP超过JPEG单段容量，未写入")
            else:
                if encoding=="JPEG":
                    payload=b"http://ns.adobe.com/xap/1.0/\0"+xmp
                    options["extra"]=b"\xff\xe1"+(len(payload)+2).to_bytes(2,"big")+payload
                else:
                    options["xmp"]=xmp
                report["preserved"].append("XMP（说明、评级等原有字段）")
        elif encoding=="TIFF":
            options.setdefault("tiffinfo",{})[700]=xmp
            report["preserved"].append("XMP（说明、评级等原有字段）")
        elif encoding!="PNG":
            report["omitted"].append(f"XMP：{encoding}编码器不支持")
    if encoding=="PNG":
        info=PngImagePlugin.PngInfo()
        for key,value in metadata.get("text",{}).items():
            if key==XMP_KEY:
                continue
            if isinstance(value,PngImagePlugin.iTXt):
                info.add_itxt(key,str(value),lang=value.lang,tkey=value.tkey)
            else:
                info.add_text(key,value)
        if metadata.get("text"):
            report["preserved"].append("PNG文本字段")
        if xmp:
            try:
                info.add_itxt(XMP_KEY,xmp.decode("utf-8"))
                report["preserved"].append("XMP（说明、评级等原有字段）")
            except UnicodeDecodeError:
                report["omitted"].append("非UTF-8的XMP无法保存为PNG国际文本")
        for chunk in metadata.get("chunks",[]):
            name,data,*after=chunk
            if name[3]&32:
                info.add(name,data,after_idat=bool(after and after[0]))
                report["preserved"].append(f"PNG私有块 {name.decode('ascii')}")
            else:
                report["omitted"].append(f"PNG私有块 {name.decode('ascii')} 未声明可在像素修改后复制")
        color=metadata.get("png_color",{})
        if not source.info.get("color_changed"):
            for key,name in (("gamma",b"gAMA"),("chromaticity",b"cHRM"),("srgb",b"sRGB")):
                if key in color:
                    value=color[key]
                    data=bytes([value]) if key=="srgb" else struct.pack(
                        ">"+"I"*(8 if key=="chromaticity" else 1),
                        *[round(v*100000) for v in (value if key=="chromaticity" else (value,))])
                    info.add(name,data)
                    report["preserved"].append(f"PNG {key}")
        options["pnginfo"]=info
    elif metadata.get("text") or metadata.get("chunks") or metadata.get("png_color"):
        report["omitted"].append("PNG专属文本/辅助块未迁移到其他格式（XMP单独处理）")
    extras=metadata.get("extra",[])
    if encoding=="JPEG":
        extras=list(extras)
        if metadata.get("photoshop") and not any(marker=="APP13" for marker,_ in extras):
            extras.append(("APP13",b"Photoshop 3.0\0"+metadata["photoshop"]))
        elif metadata.get("iptc") and not metadata.get("photoshop"):
            try:
                data=bytes(metadata["iptc"])
            except (ValueError,TypeError):
                report["omitted"].append("TIFF数值型IPTC无法无歧义转换为JPEG字节段")
            else:
                resource=b"8BIM\x04\x04\0\0"+len(data).to_bytes(4,"big")+data+b"\0"*(len(data)%2)
                extras.append(("APP13",b"Photoshop 3.0\0"+resource))
        allowed=[]
        for marker,data in extras:
            if len(data)>65533:
                report["omitted"].append(f"{marker}元数据超过JPEG容量")
            else:
                allowed.append((marker,data))
        options["extra"]=options.get("extra",b"")+b"".join(bytes([255,254 if marker=="COM" else 224+int(marker[3:])])+
            (len(data)+2).to_bytes(2,"big")+data for marker,data in allowed)
        if allowed:
            report["preserved"].append("JPEG注释、IPTC/Photoshop及其他原有APP元数据段")
    elif encoding=="TIFF":
        for tag,key in ((33723,"iptc"),(34377,"photoshop")):
            if metadata.get(key):
                options.setdefault("tiffinfo",{})[tag]=metadata[key]
                report["preserved"].append("IPTC" if tag==33723 else "Photoshop资源")
        if extras and any(marker not in ("APP13","COM") for marker,_ in extras):
            report["omitted"].append("JPEG专属APP/注释段未迁移到TIFF")
    elif any(marker!="COM" for marker,_ in extras) or metadata.get("iptc") or metadata.get("photoshop"):
        report["omitted"].append(f"JPEG私有段/IPTC/Photoshop资源未迁移到{encoding}")
    if metadata.get("comment") and encoding=="GIF":
        options["comment"]=metadata["comment"]
        report["preserved"].append("图片注释")
    elif metadata.get("comment") and encoding!="JPEG":
        if encoding=="PNG":
            options["pnginfo"].add_text("Comment",metadata["comment"].decode("latin1"))
            report["adjusted"].append("图片注释以原字节映射为PNG文本Comment")
        else:
            report["omitted"].append(f"图片注释：{encoding}编码器不支持")
    elif metadata.get("comment") and not extras:
        if encoding=="JPEG":
            options["comment"]=metadata["comment"]
            report["preserved"].append("图片注释")
    if encoding=="TIFF" and "tiffinfo" in options and "exif" in options:
        # Pillow gives tiffinfo precedence over exif: merge them to retain both.
        exif=Image.Exif()
        exif.load(options.pop("exif"))
        for tag,value in options["tiffinfo"].items():
            exif[tag]=value
        options["tiffinfo"]=exif
    return options,report


def restore_file_attributes(source, destination, report, encoding):
    metadata=source.info.get("source_metadata",{})
    original=metadata.get("stat")
    if original is None:
        return
    for name,value in metadata.get("xattrs",{}).items():
        if name in ("com.apple.ResourceFork","com.apple.decmpfs"):
            report["omitted"].append(f"扩展属性 {name} 可能含旧预览或压缩数据，未复制")
            continue
        if name=="com.apple.FinderInfo" and len(value)>=10:
            value=bytearray(value)
            if source.info.get("source_format")!=encoding:
                value[:4]=b"\0"*4
                report["adjusted"].append("Finder文件类型交由系统按新格式识别")
            if "com.apple.ResourceFork" in metadata.get("xattrs",{}):
                flags=int.from_bytes(value[8:10],"big")&~0x0400
                value[8:10]=flags.to_bytes(2,"big")
            value=bytes(value)
        try:
            os.setxattr(destination,name,value)
            report["preserved"].append(f"扩展属性 {name}")
        except (OSError,AttributeError) as error:
            report["warnings"].append(f"无法保留扩展属性 {name}：{error}")
    try:
        os.chmod(destination,stat.S_IMODE(original.st_mode))
        report["preserved"].append("文件权限")
    except OSError as error:
        report["warnings"].append(f"无法保留文件权限：{error}")
    if hasattr(original,"st_birthtime"):
        if sys.platform=="darwin":
            try:
                _set_birthtime(destination,getattr(original,"st_birthtime_ns",round(original.st_birthtime*1e9)))
                report["preserved"].append("文件创建时间")
            except OSError as error:
                report["warnings"].append(f"无法保留文件创建时间：{error}")
        else:
            report["omitted"].append("当前平台未实现文件创建时间复制")
    try:
        os.utime(destination,ns=(original.st_atime_ns,original.st_mtime_ns))
        report["preserved"].append("文件访问时间和修改时间")
    except OSError as error:
        report["warnings"].append(f"无法保留文件时间：{error}")
    if getattr(original,"st_flags",0):
        report["omitted"].append("文件锁定等文件系统标志未复制")


def _set_birthtime(path, nanoseconds):
    class AttrList(ctypes.Structure):
        _fields_=[("count",ctypes.c_uint16),("reserved",ctypes.c_uint16),
                  ("common",ctypes.c_uint32),("volume",ctypes.c_uint32),
                  ("directory",ctypes.c_uint32),("file",ctypes.c_uint32),("fork",ctypes.c_uint32)]
    class Timespec(ctypes.Structure):
        _fields_=[("seconds",ctypes.c_long),("nanoseconds",ctypes.c_long)]
    attributes=AttrList(count=5,common=0x00000200)
    timestamp=Timespec(*divmod(nanoseconds,1_000_000_000))
    library=ctypes.CDLL(None,use_errno=True)
    if library.setattrlist(os.fsencode(path),ctypes.byref(attributes),ctypes.byref(timestamp),ctypes.sizeof(timestamp),0):
        number=ctypes.get_errno()
        raise OSError(number,os.strerror(number))


def report_details(report):
    labels={"preserved":"已保留","adjusted":"已调整","omitted":"未保留","warnings":"未能确认/复制","limitations":"系统边界"}
    return "\n\n".join(label+"：\n"+"\n".join(dict.fromkeys(report.get(key,[])))
                       for key,label in labels.items() if report.get(key))
