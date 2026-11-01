from io import BytesIO
import os
from pathlib import Path
import stat
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch
from xml.etree import ElementTree

from PIL import ExifTags, Image, PngImagePlugin

from stubble_removal_lab.photos import open_photo, photo_filename, save_photo


XMP=b'''<x:xmpmeta xmlns:x="adobe:ns:meta/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:xmp="http://ns.adobe.com/xap/1.0/" xmlns:dc="http://purl.org/dc/elements/1.1/"><rdf:RDF><rdf:Description xmp:Rating="4"><dc:description>Original description</dc:description></rdf:Description></rdf:RDF></x:xmpmeta>'''


def jpeg_segment(marker,data):
    return bytes((255,marker))+(len(data)+2).to_bytes(2,"big")+data


def with_thumbnail(raw):
    image=BytesIO()
    Image.new("RGB",(4,3),"red").save(image,format="JPEG")
    thumbnail=image.getvalue()
    tiff=bytearray(raw[6:])
    order="<" if tiff[:2]==b"II" else ">"
    offset=struct.unpack_from(order+"I",tiff,4)[0]
    count=struct.unpack_from(order+"H",tiff,offset)[0]
    struct.pack_into(order+"I",tiff,offset+2+12*count,len(tiff))
    thumbnail_offset=len(tiff)+30
    tiff+=struct.pack(order+"H",2)
    tiff+=struct.pack(order+"HHII",513,4,1,thumbnail_offset)
    tiff+=struct.pack(order+"HHII",514,4,1,len(thumbnail))
    tiff+=struct.pack(order+"I",0)+thumbnail
    return b"Exif\0\0"+bytes(tiff)


class MetadataTest(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name).resolve()

    def test_jpeg_keeps_xmp_iptc_comments_and_camera_fields_but_drops_old_preview(self):
        source=self.root/"相片.JPEG"
        exif=Image.Exif()
        exif[274]=1
        exif[315]="Photographer"
        exif[ExifTags.IFD.Exif]={36867:"2026:10:31 14:06:08",33434:(1,100)}
        exif[ExifTags.IFD.GPSInfo]={1:"N",2:(30,0,0),3:"E",4:(120,0,0)}
        iptc=b"\x1c\x02\x78\0\x04test"
        resource=b"8BIM\x04\x04\0\0"+len(iptc).to_bytes(4,"big")+iptc+b"\0"*(len(iptc)%2)
        extra=jpeg_segment(0xED,b"Photoshop 3.0\0"+resource)+jpeg_segment(0xFE,b"First comment")+jpeg_segment(0xFE,b"Second comment")
        Image.new("RGB",(32,24),(40,80,120)).save(source,exif=with_thumbnail(exif.tobytes()),xmp=XMP,extra=extra)
        photo=open_photo(source)
        self.assertEqual(photo_filename(photo),source.name)
        destination=self.root/"output"/source.name
        destination.parent.mkdir()
        report=save_photo(photo,photo,destination)
        with Image.open(destination) as saved:
            self.assertEqual(saved.info["xmp"],XMP)
            self.assertEqual(saved.getexif()[315],"Photographer")
            self.assertEqual(saved.getexif().get_ifd(ExifTags.IFD.Exif)[36867],"2026:10:31 14:06:08")
            self.assertEqual(saved.getexif().get_ifd(ExifTags.IFD.GPSInfo)[1],"N")
            self.assertFalse(saved.getexif().get_ifd(ExifTags.IFD.IFD1))
            self.assertEqual(saved.info["photoshop"][0x0404],iptc)
            self.assertEqual([value for key,value in saved.applist if key=="COM"],[b"First comment",b"Second comment"])
        self.assertIn("原文件名",report["preserved"])
        self.assertTrue(any("缩略图" in message for message in report["adjusted"]))

    def test_xmp_migrates_to_png_webp_and_tiff_and_reports_unsupported_formats(self):
        source=self.root/"original.jpg"
        Image.new("RGB",(32,24)).save(source,xmp=XMP)
        photo=open_photo(source)
        for encoding in ("png","webp","tiff","bmp","gif"):
            with self.subTest(format=encoding):
                destination=self.root/("converted."+encoding)
                report=save_photo(photo,photo,destination,output_format=encoding)
                with Image.open(destination) as saved:
                    if encoding in ("png","webp","tiff"):
                        self.assertEqual(saved.info["xmp"],XMP)
                    else:
                        self.assertTrue(any("XMP" in value for value in report["omitted"]))

    def test_png_retains_unicode_text_language_and_safe_private_chunks(self):
        source=self.root/"text.png"
        info=PngImagePlugin.PngInfo()
        info.add_itxt("Description","原始说明",lang="zh-CN",tkey="说明")
        info.add_text("Author","Photographer")
        info.add(b"vpAg",b"private metadata")
        info.add(b"vpAG",b"unsafe after pixel edit")
        Image.new("RGB",(32,24)).save(source,pnginfo=info)
        photo=open_photo(source)
        destination=self.root/"saved.png"
        report=save_photo(photo,photo,destination)
        with Image.open(destination) as saved:
            saved.load()
            self.assertEqual(saved.text["Description"],"原始说明")
            self.assertEqual(saved.text["Description"].lang,"zh-CN")
            self.assertEqual(saved.text["Description"].tkey,"说明")
            self.assertEqual(saved.text["Author"],"Photographer")
            self.assertIn((b"vpAg",b"private metadata"),saved.private_chunks)
            self.assertNotIn(b"vpAG",[chunk[0] for chunk in saved.private_chunks])
        self.assertTrue(any("vpAG" in value for value in report["omitted"]))
        converted=self.root/"saved.jpg"
        report=save_photo(photo,photo,converted,output_format="jpg")
        self.assertTrue(any("PNG专属" in value for value in report["omitted"]))

    def test_xmp_orientation_is_updated_without_removing_capture_date(self):
        source=self.root/"rotated.jpg"
        exif=Image.Exif()
        exif[274]=6
        xmp=b'''<x:xmpmeta xmlns:x="adobe:ns:meta/" xmlns:tiff="http://ns.adobe.com/tiff/1.0/" xmlns:exif="http://ns.adobe.com/exif/1.0/" tiff:Orientation="6" tiff:ImageWidth="32" tiff:ImageLength="24" exif:DateTimeOriginal="2026:10:31 14:06:08"/>'''
        Image.new("RGB",(32,24)).save(source,exif=exif,xmp=xmp)
        photo=open_photo(source)
        destination=self.root/"upright.jpg"
        save_photo(photo,photo,destination)
        with Image.open(destination) as saved:
            root=ElementTree.fromstring(saved.info["xmp"])
            self.assertEqual(saved.size,(24,32))
            self.assertEqual(root.attrib["{http://ns.adobe.com/tiff/1.0/}Orientation"],"1")
            self.assertEqual(root.attrib["{http://ns.adobe.com/tiff/1.0/}ImageWidth"],"24")
            self.assertEqual(root.attrib["{http://ns.adobe.com/exif/1.0/}DateTimeOriginal"],"2026:10:31 14:06:08")

    def test_file_times_permissions_and_supported_extended_attributes_are_copied(self):
        source=self.root/"original.png"
        Image.new("RGB",(32,24)).save(source)
        os.chmod(source,0o640)
        os.utime(source,ns=(1_600_000_000_123456789,1_600_000_001_123456789))
        attribute="com.apple.metadata:_kMDItemUserTags" if sys.platform=="darwin" else "user.stubble_test"
        attribute_copied=False
        if hasattr(os,"setxattr"):
            try:
                os.setxattr(source,attribute,b"test metadata")
                attribute_copied=True
            except OSError:
                pass
        photo=open_photo(source)
        snapshot=photo.info["source_metadata"]["stat"]
        destination=self.root/"saved.png"
        report=save_photo(photo,photo,destination)
        exported=destination.stat()
        self.assertEqual(exported.st_mtime_ns,snapshot.st_mtime_ns)
        self.assertEqual(exported.st_atime_ns,snapshot.st_atime_ns)
        self.assertEqual(stat.S_IMODE(exported.st_mode),0o640)
        if sys.platform=="darwin":
            self.assertAlmostEqual(exported.st_birthtime,snapshot.st_birthtime,places=5)
        if attribute_copied:
            self.assertEqual(os.getxattr(destination,attribute),b"test metadata")
        self.assertFalse(report["warnings"],report)

    def test_attribute_failure_is_reported_without_losing_image(self):
        source=self.root/"original.png"
        Image.new("RGB",(32,24)).save(source)
        photo=open_photo(source)
        photo.info["source_metadata"]["xattrs"]={"test.attribute":b"value"}
        destination=self.root/"saved.png"
        with patch("stubble_removal_lab.metadata.os.setxattr",side_effect=OSError("not supported"),create=True):
            report=save_photo(photo,photo,destination)
        self.assertTrue(destination.is_file())
        self.assertTrue(any("test.attribute" in value for value in report["warnings"]))

    def test_original_and_hardlinks_are_protected(self):
        source=self.root/"original.jpg"
        Image.new("RGB",(32,24)).save(source)
        original=source.read_bytes()
        photo=open_photo(source)
        link=self.root/"hardlink.jpg"
        os.link(source,link)
        for destination in (source,link):
            with self.subTest(path=destination),self.assertRaisesRegex(ValueError,"保留原图"):
                save_photo(photo,photo,destination)
        self.assertEqual(source.read_bytes(),original)

    def test_original_filename_and_case_are_kept_until_format_changes(self):
        source=self.root/"原名.JPEG"
        Image.new("RGB",(32,24)).save(source,format="JPEG")
        photo=open_photo(source)
        self.assertEqual(photo_filename(photo),"原名.JPEG")
        self.assertEqual(photo_filename(photo,"png"),"原名.png")

    def test_tiff_metadata_survives_same_format_and_jpeg_conversion(self):
        source=self.root/"original.tiff"
        iptc=b"\x1c\x02\x78\0\x04test"
        resource=b"8BIM\x04\x04\0\0"+len(iptc).to_bytes(4,"big")+iptc+b"\0"*(len(iptc)%2)
        Image.new("RGB",(32,24)).save(source,tiffinfo={700:XMP,33723:iptc,34377:resource,
            270:"Original description",315:"Photographer",65000:b"custom metadata"})
        photo=open_photo(source)
        for encoding in ("tiff","jpg"):
            with self.subTest(format=encoding):
                destination=self.root/("saved."+encoding)
                save_photo(photo,photo,destination,output_format=encoding)
                with Image.open(destination) as saved:
                    self.assertEqual(saved.info["xmp"],XMP)
                    self.assertEqual(saved.getexif()[270],"Original description")
                    self.assertEqual(saved.getexif()[315],"Photographer")
                    self.assertEqual(saved.getexif()[65000],b"custom metadata")
                    if encoding=="jpg":
                        self.assertEqual(saved.info["photoshop"][0x0404],iptc)
