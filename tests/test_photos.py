import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageCms

from stubble_removal_lab.photos import FORMATS, open_photo, output_encoding, save_photo


class PhotoFormatsTest(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name)

    def test_all_formats_can_be_read_and_exported_with_original_encoding(self):
        for suffix,encoding in {**FORMATS,"jpeg":"JPEG","tif":"TIFF"}.items():
            with self.subTest(suffix=suffix):
                source=self.root/f"source.{suffix}"
                Image.new("RGB",(32,24),(60,120,180)).save(source,format=encoding)
                photo=open_photo(source)
                actual,saved_suffix=output_encoding(photo)
                self.assertEqual((actual,saved_suffix),(encoding,"."+suffix))
                destination=self.root/f"saved{saved_suffix}"
                save_photo(photo,photo,destination)
                with Image.open(destination) as saved:
                    self.assertEqual((saved.format,saved.size),(encoding,(32,24)))

    def test_explicit_format_uses_real_encoder_not_filename(self):
        source=self.root/"source.bmp"
        Image.new("RGB",(32,24)).save(source)
        photo=open_photo(source)
        for name,encoding in FORMATS.items():
            with self.subTest(format=name):
                destination=self.root/f"converted-{name}.data"
                save_photo(photo,photo,destination,output_format=name)
                with Image.open(destination) as saved:
                    self.assertEqual(saved.format,encoding)

    def test_alpha_is_preserved_in_png_tiff_and_webp_and_matted_for_jpeg_bmp(self):
        source=self.root/"alpha.png"
        Image.new("RGBA",(32,24),(20,60,100,80)).save(source)
        photo=open_photo(source)
        for name in ("png","tiff","webp","jpg","bmp"):
            with self.subTest(format=name):
                destination=self.root/f"export-alpha.{name}"
                save_photo(photo,photo,destination,output_format=name)
                with Image.open(destination) as saved:
                    pixel=saved.convert("RGBA").getpixel((0,0))
                    if name in ("png","tiff","webp"):
                        self.assertEqual(pixel[3],80)
                    else:
                        self.assertEqual(pixel[3],255)
                        self.assertGreater(pixel[0],170)

    def test_palette_transparency_and_static_gif_roundtrip(self):
        source=self.root/"palette.gif"
        image=Image.new("P",(2,1))
        image.putpalette([255,0,0,0,255,0]+[0]*762)
        image.putdata([0,1])
        image.save(source,transparency=0)
        photo=open_photo(source)
        destination=self.root/"saved.gif"
        save_photo(photo,photo,destination)
        with Image.open(destination) as result:
            rgba=result.convert("RGBA")
            self.assertEqual(rgba.getpixel((0,0))[3],0)
            self.assertEqual(rgba.getpixel((1,0)),(0,255,0,255))

    def test_exif_orientation_metadata_and_alpha_stay_aligned(self):
        source=self.root/"rotated.png"
        image=Image.new("RGBA",(3,2),(40,80,120,255))
        image.putpixel((0,0),(10,20,30,0))
        exif=Image.Exif()
        exif[274]=6
        exif[315]="Test photographer"
        icc=ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
        image.save(source,exif=exif,icc_profile=icc,dpi=(144,144))
        photo=open_photo(source)
        self.assertEqual(photo.size,(2,3))
        self.assertEqual(photo.info["source_alpha"].getpixel((1,0)),0)
        for name in ("png","jpg","tiff","webp"):
            with self.subTest(format=name):
                destination=self.root/f"metadata.{name}"
                save_photo(photo,photo,destination,output_format=name)
                with Image.open(destination) as result:
                    self.assertEqual(result.size,(2,3))
                    self.assertNotIn(274,result.getexif())
                    self.assertEqual(result.getexif()[315],"Test photographer")
                    self.assertEqual(result.info["icc_profile"],icc)
                    if name!="webp":
                        self.assertAlmostEqual(result.info["dpi"][0],144,delta=1)

    def test_high_bit_depth_is_rejected_including_rgb_tiff(self):
        for suffix,shape in [("png",(24,32)),("tiff",(24,32)),("tiff",(24,32,3)),("png",(24,32,3))]:
            with self.subTest(suffix=suffix,shape=shape):
                path=self.root/("highbit."+suffix)
                self.assertTrue(cv2.imwrite(str(path),np.full(shape,32768,dtype=np.uint16)))
                with self.assertRaisesRegex(ValueError,"16位"):
                    open_photo(path)

    def test_animations_and_multipage_images_are_rejected(self):
        frames=[Image.new("RGB",(32,24),color) for color in ("red","green")]
        for suffix in ("gif","png","tiff","webp"):
            with self.subTest(suffix=suffix):
                path=self.root/f"multiple.{suffix}"
                frames[0].save(path,save_all=True,append_images=frames[1:],duration=100)
                with self.assertRaisesRegex(ValueError,"动画或多页"):
                    open_photo(path)

    def test_cmyk_and_grayscale_inputs_become_rgb(self):
        for mode,suffix in [("CMYK","jpg"),("L","png"),("LA","png"),("1","bmp")]:
            with self.subTest(mode=mode):
                path=self.root/f"mode.{suffix}"
                Image.new(mode,(32,24)).save(path)
                self.assertEqual(open_photo(path).mode,"RGB")

    def test_invalid_quality_or_format_does_not_create_output(self):
        source=Image.new("RGB",(32,24))
        destination=self.root/"invalid.png"
        for options in ({"quality":0},{"quality":101},{"output_format":"unknown"}):
            with self.subTest(options=options),self.assertRaises(ValueError):
                save_photo(source,source,destination,**options)
        self.assertFalse(destination.exists())
