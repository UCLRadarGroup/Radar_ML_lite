import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'hackathon'))
import tempfile
import unittest
import zipfile
from pathlib import Path
import numpy as np
from compact_dataset import build, expand
from make_dataset_hackathon import process_file


class CompactTests(unittest.TestCase):
    def test_zip_roundtrip_matches_direct_processing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / '100us_cw_0Mhz_270Mhz.npy'
            raw = np.zeros((3, 2, 33600), dtype=np.int16)
            raw[1, :, 6000:24000] = 200
            np.save(source, raw)
            archive = root / 'raw.zip'
            with zipfile.ZipFile(archive, 'w') as z:
                z.write(source, 'nested/' + source.name)
            self.assertEqual(build(archive, root / 'compact', 'ADC2', 2), 1)
            compact = next((root / 'compact').glob('*_recorded.npy'))
            np.testing.assert_array_equal(np.load(compact), raw[1:2])
            expand(root / 'compact', root / 'expanded', [30, 0, -30])
            expected = process_file(source, root / 'direct', 'ADC2', 2, [30, 0, -30])
            np.testing.assert_array_equal(np.load(expected), np.load(root / 'expanded' / expected.name))
            with self.assertRaises(FileExistsError):
                build(archive, root / 'compact', 'ADC2', 2)
            with compact.open('ab') as stream:
                stream.write(b'corruption')
            with self.assertRaisesRegex(ValueError, 'Checksum mismatch'):
                expand(root / 'compact', root / 'corrupt')
