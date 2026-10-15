import tempfile
import unittest
from pathlib import Path

from stubble_removal_lab.paths import unique_path


class UniquePathTest(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name).resolve()

    def test_free_path_is_returned_unchanged(self):
        self.assertEqual(unique_path(self.root/"photo"),self.root/"photo")
        self.assertEqual(unique_path(self.root/"selection.json",suffix=True),
                         self.root/"selection.json")

    def test_existing_directories_get_numbered(self):
        (self.root/"photo").mkdir()
        self.assertEqual(unique_path(self.root/"photo"),self.root/"photo_(1)")
        (self.root/"photo_(1)").mkdir()
        self.assertEqual(unique_path(self.root/"photo"),self.root/"photo_(2)")

    def test_file_counter_goes_before_the_extension(self):
        (self.root/"selection.json").write_text("{}")
        self.assertEqual(unique_path(self.root/"selection.json",suffix=True),
                         self.root/"selection_(1).json")
        (self.root/"selection_(1).json").write_text("{}")
        self.assertEqual(unique_path(self.root/"selection.json",suffix=True),
                         self.root/"selection_(2).json")

    def test_dotted_directory_name_is_kept_intact(self):
        (self.root/"my.photo").mkdir()
        self.assertEqual(unique_path(self.root/"my.photo"),self.root/"my.photo_(1)")
