"""Keep recorded pulses unchanged, or expand a compact RadarML dataset locally.

Adapted preprocessing utilities: GPL-3.0; see make_dataset_hackathon.py.
"""
import argparse
import json
from pathlib import Path
import shutil
import zipfile
import subprocess
from contextlib import contextmanager

import numpy as np
from make_dataset_hackathon import (
    CHANNEL_NAMES, FS, TARGET_SNR_DBS, file_hash, output_name,
    process_rows, read_zip_rows,
)


def save_compact(source, rows, shape, output, channel):
    target = output / output_name(source).replace('_single_channel', '_recorded')
    if target.exists() or target.with_suffix('.json').exists():
        raise FileExistsError(target)
    temporary = target.with_suffix('.partial.npy')
    with temporary.open('wb') as stream:
        np.save(stream, rows[np.newaxis], allow_pickle=False)
    temporary.replace(target)
    metadata = dict(format='radarml-recorded-v1', source_file=source,
                    source_shape=list(shape), channel=channel,
                    source_pulse_indices=list(range(len(rows))),
                    sample_rate_hz=FS, output_shape=[1, *rows.shape],
                    dtype='int16', snr_modified=False,
                    output_sha256=file_hash(target, 'sha256'))
    target.with_suffix('.json').write_text(json.dumps(metadata, indent=2))
    return target


class NativeZipReader:
    """Use optional 7-Zip for fast encrypted prefix reads; no raw extraction."""
    def __init__(self, executable, path):
        self.executable, self.path = str(executable), str(path.resolve())

    @contextmanager
    def open(self, member, pwd=None):
        process = subprocess.Popen(
            [self.executable, 'x', '-so', '-pUCLRESM', self.path, member.filename],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        try:
            yield process.stdout
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.terminate()
            process.wait()


def build(input_path, output, channel='ADC0', pulses=100, seven_zip=None):
    input_path, output = Path(input_path), Path(output)
    if pulses < 1:
        raise ValueError('Pulse count must be positive')
    if output.exists():
        raise FileExistsError('Choose a new output directory')
    output.mkdir(parents=True)
    marker = output / 'INCOMPLETE.txt'
    marker.write_text('Do not distribute until the build completes.')
    count = 0
    if input_path.is_file():
        with zipfile.ZipFile(input_path) as archive:
            reader = NativeZipReader(seven_zip, input_path) if seven_zip else archive
            for member in sorted(archive.infolist(), key=lambda m: m.filename):
                if not member.filename.endswith('.npy'):
                    continue
                rows, shape = read_zip_rows(reader, member, channel, pulses)
                save_compact(Path(member.filename).name, rows, shape, output, channel)
                count += 1
                print(f'{count}: {member.filename}', flush=True)
    else:
        for source in sorted(input_path.glob('*.npy')):
            raw = np.load(source, mmap_mode='r', allow_pickle=False)
            if raw.ndim != 3 or raw.shape[0] != 3 or raw.shape[2] != 33600 or raw.dtype != np.int16:
                raise ValueError(f'{source}: expected original int16 (3, N, 33600)')
            if pulses > raw.shape[1]:
                raise ValueError(f'{source}: insufficient pulses')
            save_compact(source.name, raw[CHANNEL_NAMES.index(channel), :pulses], raw.shape, output, channel)
            count += 1
    if not count:
        raise ValueError('No original NPY recordings found')
    for name in ('compact_dataset.py', 'make_dataset_hackathon.py', 'LICENSE'):
        shutil.copyfile(Path(__file__).with_name(name), output / name)
    (output / 'README.md').write_text(
        '# Recorded RadarML pulses\n\n'
        'Each *_recorded.npy contains (1, pulses, 33600) interleaved int16 I/Q values '
        'at 120 MS/s. Original recorded noise remains; no synthetic noise has been added. '
        'Keep the JSON sidecars with the arrays.\n\n'
        'Install Python and NumPy (`python -m pip install numpy`), then run from this folder:\n\n'
        '```\npython compact_dataset.py expand --input . --output ../expanded\n```\n\n'
        'Default expansion uses 13 requested SNRs (+30 to -30 dB), seed 42, and about '
        '13 times the compact array storage. Add `--snrs 20 10 0 -10 -20` for a smaller sweep. '
        'Outputs match the organizer preprocessing for identical settings and NumPy environment. '
        'All variants of a source pulse must stay in the same train/validation/test split. '
        'This source-named dataset exposes labels; it is intended for an open workshop, '
        'not an anonymized blind competition.\n\n'
        'Source: Ritchie, White and Hosford (2025), RadarML, '
        'https://doi.org/10.5522/04/30752767.v1. '
        'The included LICENSE covers the preprocessing code (GPL-3.0); '
        'dataset licensing is separate; see the source record.\n')
    marker.unlink()
    return count


def expand(input_path, output, snrs=TARGET_SNR_DBS, seed=42):
    root, output = Path(input_path), Path(output)
    files = sorted(root.glob('*_recorded.npy'))
    if not files or (root / 'INCOMPLETE.txt').exists():
        raise ValueError('Missing or incomplete compact dataset')
    if output.exists():
        raise FileExistsError('Choose a new output directory')
    output.mkdir(parents=True)
    for source in files:
        meta = json.loads(source.with_suffix('.json').read_text())
        rows = np.load(source, mmap_mode='r', allow_pickle=False)
        if (meta.get('format') != 'radarml-recorded-v1' or
                rows.shape != (1, len(meta['source_pulse_indices']), 33600) or
                rows.dtype != np.int16 or meta['source_pulse_indices'] != list(range(rows.shape[1]))):
            raise ValueError(f'Invalid compact file: {source}')
        if file_hash(source, 'sha256') != meta['output_sha256']:
            raise ValueError(f'Checksum mismatch: {source}')
        process_rows(meta['source_file'], rows[0], meta['source_shape'], output,
                     meta['channel'], snrs, seed,
                     provenance=dict(compact_sha256=meta['output_sha256']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('build', 'expand'):
        p = sub.add_parser(command)
        p.add_argument('--input', type=Path, required=True)
        p.add_argument('--output', type=Path, required=True)
        if command == 'build':
            p.add_argument('--channel', choices=CHANNEL_NAMES, default='ADC0')
            p.add_argument('--pulses', type=int, default=100)
            p.add_argument('--seven-zip', type=Path, help='Optional 7z executable for faster encrypted ZIP reading')
        else:
            p.add_argument('--snrs', type=int, nargs='+', default=TARGET_SNR_DBS)
            p.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.command == 'build':
        build(args.input, args.output, args.channel, args.pulses, args.seven_zip)
    else:
        expand(args.input, args.output, args.snrs, args.seed)
