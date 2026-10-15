import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from stubble_removal_lab.batch import collect_photos, process_batch
from stubble_removal_lab.cli import main
from stubble_removal_lab.photos import open_photo, output_encoding, save_photo


class EncodingPipeline:
    """Stands in for the real Pipeline so batch tests never load models."""

    def process(self,source,profile,output,**options):
        photo=open_photo(source)
        _,suffix=output_encoding(photo,options["output_format"])
        output.mkdir()
        result=output/f"result{suffix}"
        save_photo(photo,photo,result,output_format=options["output_format"])
        return result


class BatchTest(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name).resolve()
        self.inputs=self.root/"inputs"
        self.inputs.mkdir()

    def photo(self,name):
        path=self.inputs/name
        path.parent.mkdir(parents=True,exist_ok=True)
        Image.new("RGB",(32,24),(50,100,150)).save(path)
        return path

    def test_directory_discovery_recursion_deduplication_and_case(self):
        first=self.photo("first.JPG")
        second=self.photo("nested/second.png")
        (self.inputs/"other.txt").write_text("not a photo")
        self.assertEqual(collect_photos([self.inputs,first]),[first])
        self.assertEqual(collect_photos([self.inputs,second],recursive=True),[first,second])

    def test_batch_reuses_model_continues_errors_and_keeps_duplicate_names(self):
        first=self.photo("a/same.jpg")
        corrupt=self.inputs/"broken.png"
        corrupt.write_text("not an image")
        last=self.photo("b/same.jpg")

        with patch("stubble_removal_lab.pipeline.Pipeline",return_value=EncodingPipeline()) as factory, \
             patch("stubble_removal_lab.detection.detect_profile",side_effect=lambda source,**kw:open_photo(source)):
            report=process_batch([first,corrupt,last],self.root/"output",output_format="webp")
        factory.assert_called_once()
        self.assertEqual((report["status"],report["succeeded"],report["failed"]),("partial",2,1))
        self.assertEqual([item["status"] for item in report["items"]],["complete","failed","complete"])
        self.assertNotEqual(report["items"][0]["output"],report["items"][2]["output"])
        for item in (report["items"][0],report["items"][2]):
            with Image.open(item["output"]) as result:
                self.assertEqual(result.format,"WEBP")
        self.assertEqual(json.loads((self.root/"output/batch.json").read_text()),report)

    def test_existing_output_gets_numbered_and_empty_directories_are_not_modified(self):
        photo=self.photo("one.png")
        existing=self.root/"existing"
        existing.mkdir()
        (existing/"keep.txt").write_text("mine")
        with patch("stubble_removal_lab.pipeline.Pipeline",return_value=EncodingPipeline()), \
             patch("stubble_removal_lab.detection.detect_profile",side_effect=lambda source,**kw:open_photo(source)):
            first=process_batch([photo],existing)
            second=process_batch([photo],existing)
        self.assertEqual(first["output"],str(self.root/"existing_(1)"))
        self.assertEqual(second["output"],str(self.root/"existing_(2)"))
        self.assertTrue(Path(first["output"],"batch.json").is_file())
        self.assertTrue((existing/"keep.txt").is_file())
        self.assertFalse((existing/"batch.json").exists())
        empty=self.root/"empty-input"
        empty.mkdir()
        with self.assertRaisesRegex(ValueError,"没有找到"):
            process_batch([empty],self.root/"empty-output")
        self.assertFalse((self.root/"empty-output").exists())

    def test_interrupt_keeps_cancelled_summary(self):
        first=self.photo("first.png")
        with patch("stubble_removal_lab.pipeline.Pipeline"), \
             patch("stubble_removal_lab.detection.detect_profile",side_effect=KeyboardInterrupt), \
             self.assertRaises(KeyboardInterrupt):
            process_batch([first],self.root/"cancelled")
        report=json.loads((self.root/"cancelled/batch.json").read_text())
        self.assertEqual(report["status"],"cancelled")
        self.assertEqual(report["items"][0]["status"],"cancelled")

    def test_cli_batch_returns_nonzero_for_partial_failure(self):
        with patch("stubble_removal_lab.batch.process_batch",return_value={"failed":1}), \
             patch("builtins.print"):
            self.assertEqual(main(["batch","--input","one.jpg","two.png","--format","jpeg"]),1)
