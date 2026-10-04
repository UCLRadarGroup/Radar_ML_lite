import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'hackathon'))
import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np

spec = importlib.util.spec_from_file_location('make_dataset', Path(__file__).resolve().parents[1] / 'hackathon' / 'make_dataset_hackathon.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class DatasetTests(unittest.TestCase):
    def test_channel_shape_and_repeatability(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / '100us_cw_0Mhz_270Mhz.npy'
            raw = np.zeros((3, 2, 33600), dtype=np.int16)
            for c in range(3):
                raw[c, :, 6000:24000] = 100 * (c + 1)
            np.save(source, raw)
            first = m.process_file(source, root / 'a', channel='ADC2', pulses=2, snrs=[0])
            second = m.process_file(source, root / 'b', channel='ADC2', pulses=2, snrs=[0])
            a, b = np.load(first), np.load(second)
            self.assertEqual(a.shape, (1, 1, 2, 33600))
            self.assertEqual(a.dtype, np.int16)
            np.testing.assert_array_equal(a, b)
            expected, _, _ = m.degrade(raw[1, 0], 0, np.random.default_rng(int(__import__('hashlib').md5(f'42_{source.name}_0_0'.encode()).hexdigest()[:8], 16)))
            np.testing.assert_array_equal(a[0, 0, 0], expected)
            with self.assertRaises(FileExistsError):
                m.process_file(source, root / 'a', pulses=2, snrs=[0])
            with self.assertRaises(ValueError):
                m.process_file(source, root / 'c', pulses=3, snrs=[0])

    def test_high_snr_guard_and_clipping(self):
        row = np.ones(33600, dtype=np.int16) * 100
        row[6000:24000] = 200
        out, unattainable, _ = m.degrade(row, 100, np.random.default_rng(1))
        self.assertTrue(unattainable)
        np.testing.assert_array_equal(out, row)
        _, _, clipped = m.degrade(row, -60, np.random.default_rng(1))
        self.assertGreater(clipped, 0)

    def test_names(self):
        self.assertEqual(m.output_name('100us_cw_0Mhz_270Mhz.npy'), '100us_cw_0Mhz_-30Mhz_single_channel.npy')
        self.assertEqual(m.output_name('100us_fmcw_20Mhz_230Mhz-250Mhz.npy'), '100us_fmcw_20Mhz_10Mhz--10Mhz_single_channel.npy')


if __name__ == '__main__':
    unittest.main()
