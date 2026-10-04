"""Create a fixed per-recording 80/20 split for the compact release."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


def create_manifest(data, seed=42):
    data = Path(data)
    target = data / 'split_manifest.csv'
    report = data / 'split_manifest.json'
    if target.exists() or report.exists():
        raise FileExistsError('A split manifest already exists; do not replace a published split.')
    files = sorted(data.glob('*_recorded.npy'))
    if len(files) != 660 or (data / 'INCOMPLETE.txt').exists():
        raise ValueError('Expected a complete 660-file compact release')
    rows = []
    for path in files:
        meta = json.loads(path.with_suffix('.json').read_text())
        array = np.load(path, mmap_mode='r', allow_pickle=False)
        if (array.shape != (1, 100, 33600) or array.dtype != np.int16 or
                meta['source_pulse_indices'] != list(range(100))):
            raise ValueError(f'Unexpected pulse layout: {path}')
        del array
        # Hash ranking gives a fixed pseudorandom permutation independently for
        # each recording, unaffected by file ordering or NumPy RNG versions.
        ranked = sorted(range(100), key=lambda pulse: (
            hashlib.sha256(f'{seed}:{meta["source_file"]}:{pulse}'.encode()).digest(), pulse))
        training = set(ranked[:80])
        rows.extend(dict(file=path.name, pulse_index=pulse,
                         split='train' if pulse in training else 'validation')
                    for pulse in range(100))
    temporary = target.with_suffix('.partial.csv')
    with temporary.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['file', 'pulse_index', 'split'])
        writer.writeheader()
        writer.writerows(rows)
    # Verify the serialized artifact, including exact coverage and disjointness.
    with temporary.open(newline='', encoding='utf-8') as stream:
        saved = list(csv.DictReader(stream))
    assert len(saved) == 66000
    identities = {(row['file'], int(row['pulse_index'])) for row in saved}
    assert identities == {(path.name, pulse) for path in files for pulse in range(100)}
    counts = Counter((row['file'], row['split']) for row in saved)
    for path in files:
        assert counts[path.name, 'train'] == 80
        assert counts[path.name, 'validation'] == 20
    temporary.replace(target)
    metadata = dict(
        format_version=1, seed=seed,
        algorithm='Rank pulses by SHA256 of UTF-8 seed:source_file:pulse_index; lowest 80 train, remaining 20 validation',
        index_definition='Zero-based pulse axis index; matches original source index 0..99',
        compact_layout='array[0, pulse_index, :]',
        expanded_layout='array[0, snr_index, pulse_index, :]',
        filename_mapping='Replace _recorded.npy with _single_channel.npy for expanded files',
        snr_rule='All SNR variants inherit the original pulse split',
        files=660, train=52800, validation=13200, final_test_included=False,
        manifest_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        verification='CSV reloaded: complete coverage, unique identities, exactly 80/20 per file')
    report.write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(metadata, indent=2))
    return target


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    create_manifest(args.data, args.seed)
