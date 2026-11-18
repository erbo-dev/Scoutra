import tempfile
import unittest
from pathlib import Path

from common import Blocked
from two_clip_pilot import NAMES, selected_paths


class TwoClipSelection(unittest.TestCase):
    def test_explicit_directory_needs_only_the_two_clips(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / name for name in NAMES]
            for path in paths:
                path.touch()
            (root / "unrelated.mp4").touch()
            self.assertEqual(selected_paths(str(root)), paths)

    def test_auto_selection_rejects_a_split_or_ambiguous_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for parent in ("a", "b"):
                (root / parent).mkdir()
            for parent, name in zip(("a", "b"), NAMES, strict=True):
                (root / parent / name).touch()
            with self.assertRaises(Blocked):
                selected_paths("auto", root)
            for parent in ("a", "b"):
                for name in NAMES:
                    (root / parent / name).touch()
            with self.assertRaises(Blocked):
                selected_paths("auto", root)

    def test_auto_selection_prefers_the_known_mount(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "datasets/erbosynshaimardan/zhanto-highlights"
            path.mkdir(parents=True)
            for name in NAMES:
                (path / name).touch()
                (root / name).touch()
            self.assertEqual(selected_paths("auto", root), [path / name for name in NAMES])
