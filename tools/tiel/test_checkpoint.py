"""No model or GPU required: checkpoint integrity failure tests."""
import json
from pathlib import Path
import tempfile
import unittest
from checkpoint import digest, verify, runtime_environment


class CheckpointTests(unittest.TestCase):
    def test_inherited_experiments_removed(self):
        inherited = dict(PATH='/usr/bin', LLAMA_TIEL_EXTRA_BANK='1',
                         LLAMA_TIEL_PHASE_ARENA='1', GGML_VK_MEMORY_LOGGER='1',
                         LLAMA_MOE_CACHE_ADAPTIVE='1', LLAMA_SERVER_TOOL_CHECKPOINTS='1',
                         LD_PRELOAD='wrong.so', LD_LIBRARY_PATH='/wrong', RADV_PERFTEST='test')
        env = runtime_environment(inherited, {}, Path('/checkpoint'))
        self.assertEqual(env, {'PATH':'/usr/bin', 'LD_LIBRARY_PATH':'/checkpoint/bin'})
        self.assertEqual(inherited['LLAMA_TIEL_EXTRA_BANK'], '1')

    def test_explicit_profile_preserved(self):
        env = runtime_environment({'LLAMA_TIEL_EXTRA_BANK':'1'},
                                  {'LLAMA_TIEL_PHASE_ARENA':'1', 'LD_LIBRARY_PATH':'/wrong'}, Path('/saved'))
        self.assertEqual(env, {'LLAMA_TIEL_PHASE_ARENA':'1', 'LD_LIBRARY_PATH':'/saved/bin'})

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
