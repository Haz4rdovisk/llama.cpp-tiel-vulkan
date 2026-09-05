"""No model or GPU required: checkpoint integrity failure tests."""
import json
from pathlib import Path
import tempfile
import unittest
from checkpoint import digest, verify


class CheckpointTests(unittest.TestCase):
    def test_integrity_and_corruption(self):
        with tempfile.TemporaryDirectory(prefix='tiel-checkpoint-test-') as directory:
            root = Path(directory)
            artifact = root / 'library'
            artifact.write_bytes(b'original')
            (root / 'manifest.json').write_text(json.dumps({
                'sha256': {'library': digest(artifact)}}))
            verify(root)
            artifact.write_bytes(b'changed')
            with self.assertRaises(RuntimeError):
                verify(root)

    def test_missing_file(self):
        with tempfile.TemporaryDirectory(prefix='tiel-checkpoint-test-') as directory:
            root = Path(directory)
            (root / 'manifest.json').write_text(json.dumps({'sha256': {'missing': '0'}}))
            with self.assertRaises(FileNotFoundError):
                verify(root)

    def test_escape_path(self):
        with tempfile.TemporaryDirectory(prefix='tiel-checkpoint-test-') as directory:
            root = Path(directory)
            (root / 'manifest.json').write_text(json.dumps({'sha256': {'../outside': '0'}}))
            with self.assertRaises(RuntimeError):
                verify(root)


if __name__ == '__main__':
    unittest.main()
