import os
os.environ.setdefault("QT_QPA_PLATFORM","offscreen")

import tempfile
import json
import unittest
from pathlib import Path
from unittest.mock import Mock,patch

from PIL import Image
from PySide6.QtWidgets import QApplication

from stubble_removal_lab.gui import Window
from stubble_removal_lab.gui_batch import BatchDialog
from stubble_removal_lab.photos import FORMATS, PHOTO_FILTERS, open_photo


class DesktopPhotosTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app=QApplication.instance() or QApplication([])

    def test_single_export_uses_same_formats_as_cli_and_retains_transparency(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve()
            source=root/"source.png"
            Image.new("RGBA",(32,24),(10,30,50,80)).save(source)
            window=Window()
            try:
                window.source=source
                window.original=open_photo(source)
                window.full_result=window.original.copy()
                for name,encoding in FORMATS.items():
                    with self.subTest(format=name):
                        destination=root/f"export.{name}"
                        window.export.format.setCurrentIndex(window.export.format.findData(name))
                        with patch("stubble_removal_lab.gui.QFileDialog.getSaveFileName",
                                   return_value=(str(destination),PHOTO_FILTERS[name])), \
                             patch.object(window,"error",side_effect=AssertionError):
                            window.save_result()
                        with Image.open(destination) as saved:
                            self.assertEqual(saved.format,encoding)
                            if name in ("png","tiff","webp"):
                                self.assertEqual(saved.convert("RGBA").getpixel((0,0))[3],80)
            finally:
                window.close()

    def test_batch_dialog_passes_human_choices_to_shared_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve()
            dialog=BatchDialog(model="B",strength=70)
            try:
                dialog.add_paths([root,root])
                dialog.recursive.setChecked(True)
                dialog.output.setText(str(root/"results"))
                dialog.export.format.setCurrentIndex(dialog.export.format.findData("jpg"))
                dialog.export.quality.setValue(90)
                dialog.face.setValue(2)
                request=dialog.request()
                self.assertEqual(request,{"task":"batch","inputs":[str(root)],
                    "output":str(root/"results"),"recursive":True,"model":"B", "strength":.7,
                    "face_index":1,"output_format":"jpg","quality":90})
            finally:
                dialog.close()

    def test_worker_json_retains_unicode_paths_across_read_boundaries(self):
        window=Window()
        payload={"type":"result","path":"/照片/修复结果.jpg"}
        data=(json.dumps(payload,ensure_ascii=False)+"\n").encode("utf-8")
        boundary=data.index("照".encode("utf-8"))+1
        window.job=Mock()
        window.job.readAllStandardOutput.side_effect=[data[:boundary],data[boundary:]]
        try:
            window.read_stdout()
            window.read_stdout()
            self.assertEqual(window.pending_result,payload)
        finally:
            window.job=None
            window.close()
