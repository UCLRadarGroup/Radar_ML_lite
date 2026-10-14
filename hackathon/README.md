# RadarML Lite hackathon guide

[Back to the dataset overview](../README.md)

## Your challenge

Predict a radar pulse's modulation family and measure how classification changes
as noise increases. Use the exact labels `barker`, `cw`, `fmcw`, `fsk`, `hyp`,
`nlfm`, `quad`. All Barker-code subtypes belong to the single `barker` class.

| Track | Inputs and processing |
| --- | --- |
| A: time domain | Ordered I/Q; magnitude, phase differences, normalization, time-domain statistics and learned 1D convolutions are allowed. No explicit FFT, STFT, wavelet or spectrogram representations. |
| B: time-frequency | Representations computed from the same I/Q, such as STFT spectrograms, with your chosen model. Include representation computation in inference timing. |

Compare tracks using identical split assignments and evaluation settings.

## Participant workflow

Follow the [seven-day timetable and event rules](EVENT_RULES.md). Day 0 is the
opening day; Day 6 is the final day. All times are Europe/London local time.
Final predictions and reports are due **Saturday 17 October 2026, 12:00 noon BST (11:00 UTC)**.
Upload prediction CSVs to the [Kaggle competition](https://www.kaggle.com/t/8c86ae869036416784b3f5521d5f6717).

| Milestone | Deadline |
| --- | --- |
| Kick-off and training begins | Day 0, 09:00 |
| Model and preprocessing freeze | Day 4, 17:00 |
| Blind-test release | Day 5, 09:00 |
| Final predictions and report due | Saturday 17 October 2026, 12:00 noon BST (11:00 UTC) |
| Scoring and wrap-up | After the final submission deadline |

Train from scratch on supplied training pulses only; use validation for tuning.
External training data and pretrained weights are excluded. Each team may submit
one final entry per track. Do not use assessment feedback to tune the frozen pipeline.
See the event rules for replacement submissions, permitted tools and tie-breaking.

| Stage | What you receive or use | What you do |
| --- | --- | --- |
| Start | `recorded_dataset.zip`, `split_manifest.csv` and `split_manifest.json` | Extract the compact data, place the manifest files beside the recordings, set up Python and inspect a pulse. |
| Development | Published `train` and `validation` assignments | Choose a track, train using the 80 training pulses per configuration, and tune using the 20 validation pulses. Keep every augmented version in its source pulse's split. |
| Model freeze | Your selected model and preprocessing | Save the model, configuration, random seeds and code revision. Record validation results before receiving the blind test. |
| Blind evaluation | [`blind_test_participants.zip` from the UCL blind-test release](https://rdr.ucl.ac.uk/articles/dataset/RadarML_-_Lite_-_Blind_Test_Release/34065705?file=69638289) | Extract into a separate folder, verify the checksum, load the anonymous test signals and run the frozen pipeline. Do not retrain or tune on test signals. |
| Submission | Completed template and method materials | Submit predictions and the supporting items below. The organizer scores predictions using the private answer key. |

The main dataset's 80/20 manifest contains **no test split**. The blind test is a
separate release with different source pulses. Do not create a new test split
from the supplied training/validation rows or run the older 60/20/20 workflow.

### Final submission checklist

- Save `predictions.csv` with exactly `sample_id,label` as the header,
  using every ID in the blind-test template once and only the seven permitted
  labels. Keep a separate prediction file for each track entered.
- Include the saved model, preprocessing settings, code revision or code archive,
  dependency versions and a command that reproduces inference.
- State the track, training subset, training SNR settings and random seeds.
- Include validation balanced accuracy by requested SNR and validation confusion
  matrices. The organizer supplies final blind-test scores after scoring.
- Report training time, inference time per pulse, hardware and model size.
  Include signal preprocessing and time-frequency conversion in inference timing.
- Give a brief account of the method and its main limitations. Disclose any
  changes made after the model freeze; do not present those as the frozen result.

There is no participant answer key for the blind evaluation. Do not attempt to
recover labels from the original public archive. Deadlines, submission limits
and ranking rules are in [EVENT_RULES.md](EVENT_RULES.md). Prediction CSVs are submitted to the [Kaggle competition](https://www.kaggle.com/t/8c86ae869036416784b3f5521d5f6717);
supporting model/report materials go to the organizers through their designated channel.

## Release and event status

The compact ZIP includes recorded arrays, JSON metadata, expansion scripts,
a verification report and code license. The shared split is supplied separately
as `split_manifest.csv` and `split_manifest.json`. Download both from the
[UCL dataset page](https://rdr.ucl.ac.uk/articles/dataset/Radar_ML_-_Lite/33977674)
or use this repository's [CSV](split_manifest.csv) and [JSON](split_manifest.json)
copies. Place both alongside the extracted recordings; the ZIP does not need
to be rebuilt. On GitHub, choose **Download raw file** rather than saving the
file-preview web page.

The fixed split uses **80 training and 20 validation pulses per recording**:
52,800 training and 13,200 validation source pulses across 660 recordings.
**This split contains training and validation only.** A separate blind test
has now been created and verified; see [final evaluation](#7-final-evaluation-and-submission)
for its contents and release sequence. Report results on the 80/20 split as
validation performance.

### Using the split CSV

| Column | Meaning |
| --- | --- |
| `file` | Compact `_recorded.npy` filename relative to the extracted data folder |
| `pulse_index` | Zero-based pulse index, 0–99 |
| `split` | `train` or `validation` |

Run from the extracted data folder with NumPy installed:

```python
import csv
import numpy as np

with open("split_manifest.csv", newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream))
train_rows = [r for r in rows if r["split"] == "train"]
validation_rows = [r for r in rows if r["split"] == "validation"]
r = train_rows[0]
array = np.load(r["file"], mmap_mode="r", allow_pickle=False)
raw = array[0, int(r["pulse_index"]), :]
iq = raw[0::2].astype(np.float32) + 1j * raw[1::2].astype(np.float32)
print(iq.shape)  # (16800,)
```

For expanded arrays, replace `_recorded.npy` with `_single_channel.npy` and
select `array[0, snr_index, pulse_index, :]`. Read SNR ordering from the expanded
JSON sidecar. Every SNR variant or augmentation inherits its source pulse's
assignment. Expansion does not apply the CSV automatically: training and
evaluation code must enforce it. All 13 SNRs give 686,400 training and 171,600
validation examples.

The JSON records seed 42, the CSV checksum and the algorithm: rank pulses by
SHA256 of UTF-8 `42:source_file:pulse_index`, using the original source filename
from each recording's JSON; the lowest 80 hashes define training. Use the
published assignments rather than creating another random split.

Review on 23 September 2026 confirmed all 66,000 pulse identities were unique,
all recordings had complete 0–99 coverage and exactly 80/20 assignments,
array layouts matched, assignments reproduced and the CSV checksum matched.
Both splits share waveform configurations, so validation measures new pulses
within known configurations, not generalization to unseen configurations or
hardware. Keep filenames and metadata out of model inputs.

## 1. Download and set up

Download only the development files at the start:

1. Get `recorded_dataset.zip` from the
   [public UCL Radar ML - Lite page](https://rdr.ucl.ac.uk/articles/dataset/Radar_ML_-_Lite/33977674).
2. Get `split_manifest.csv` and `split_manifest.json` separately from that page.
   If they are not visible, download the [CSV](split_manifest.csv) and
   [JSON](split_manifest.json) from this repository using **Download raw file**.
3. Extract the ZIP. Move both split files into the extracted `recorded_dataset`
   folder, not beside the ZIP or into a separate split folder.
4. Check the layout before running any training:

```text
recorded_dataset/
  compact_dataset.py
  make_dataset_hackathon.py
  split_manifest.csv
  split_manifest.json
  ..._recorded.npy
  ..._recorded.json
```

Do not download, inspect or use `blind_test_participants.zip` during development,
even if it appears on the same page. Wait for the organizer's final-evaluation
announcement after freezing your model and preprocessing. If you have already
accessed test signals, tell the organizers before submitting final results.

Open a terminal inside `recorded_dataset`, where `compact_dataset.py` is visible.
Keep each `*_recorded.npy` beside its matching JSON. Use Python 3.9 or newer.

On Windows, create an environment without needing PowerShell activation:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install numpy
```

On macOS/Linux:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install numpy
```

Below, `python` means that environment's interpreter: use
`.\.venv\Scripts\python.exe` on Windows or `.venv/bin/python` on macOS/Linux.
Add libraries for your classifier and plots separately.

Allow about 4.44 GB for compact arrays and another 57.66 GB for full expansion.
Keeping the ZIP too uses approximately 64.63 GB, before models and features.

## 2. Inspect one recorded pulse

Each compact array has shape `(1, 100, 33600)`: receiver, pulse, interleaved I/Q.
A pulse contains 16,800 complex samples at 120 MS/s, spanning 140 µs including
surrounding noise. The active pulse may be shorter.

Save this as `inspect_pulse.py` in the extracted folder and run
`python inspect_pulse.py`:

```python
from pathlib import Path
import numpy as np

path = sorted(Path('.').glob('*_recorded.npy'))[0]
recording = np.load(path, mmap_mode='r', allow_pickle=False)
row = recording[0, 0]
iq = row[0::2].astype(np.float32) + 1j * row[1::2].astype(np.float32)
time_us = np.arange(iq.size) / 120_000_000 * 1e6
print('Stored array:', recording.shape, recording.dtype)
print('Complex pulse:', iq.shape, iq.dtype)
```

Expect `(1, 100, 33600) int16` and `(16800,) complex64`. I and Q are components
of one signal, not receiver channels. Amplitudes are ADC values, not calibrated
RF power. Recorded arrays have no requested SNR axis or synthetic noise.

## 3. Try a small SNR expansion

Create a `pilot` folder inside the extracted dataset folder. Copy one
`*_recorded.npy` and its matching JSON into it, then run:

```bash
python compact_dataset.py expand --input pilot --output ../pilot_expanded --snrs 30 0 -30
```

Expect one `(1, 3, 100, 33600)` array, approximately 20.16 MB, and metadata.
This tests the workflow; one configuration cannot assess seven-class recognition.

When ready for full expansion:

```bash
python compact_dataset.py expand --input . --output ../expanded
```

The default uses seed 42 and requested SNRs
`30, 25, 20, 15, 10, 5, 0, -5, -10, -15, -20, -25, -30` dB. Each
`*_single_channel.npy` has shape `(1, 13, 100, 33600)`. A smaller sweep can use
`--snrs 20 10 0 -10 -20`, but still processes every input recording.

Checksums are verified before processing each recording. Choose a new output
directory for each run. Expansion cannot currently resume: after interruption,
restart in a new directory and manage incomplete output separately. Do not treat
a partly generated folder as a complete release.

Use the scripts bundled with the compact ZIP together. The original RadarML
repository's preprocessing scripts expect multi-channel inputs and are not
replacements for this expansion command.

## 4. Load an expanded pulse

From the extracted compact folder:

```python
import json
from pathlib import Path
import numpy as np

path = sorted(Path('../expanded').glob('*_single_channel.npy'))[0]
meta = json.loads(path.with_suffix('.json').read_text())
data = np.load(path, mmap_mode='r', allow_pickle=False)
snr_index = meta['snr_db'].index(0)
row = data[0, snr_index, 0]
iq = row[0::2].astype(np.float32) + 1j * row[1::2].astype(np.float32)
print(iq.shape, meta['snr_db'][snr_index])
```

For the pilot, replace `../expanded` with `../pilot_expanded`. Read the SNR order
from metadata. Memory-map arrays and load batches, not the entire collection.

`participant_loader.py` in this repository supports a different format: mixed
`(N, 33600)` shards with `index.csv` files. It does not directly load these
per-configuration compact or expanded arrays.

## 5. Keep training and validation separate

Use the published [split manifest](split_manifest.csv). The fixed 80/20 split
contains 52,800 training and 13,200 validation source pulses. At all 13 SNRs,
these become 686,400 and 171,600 examples respectively.

- Assign splits by source recording and original pulse identity before adding
  noise. Every SNR version of a pulse stays in the same split.
- Fit learned preprocessing only on training data. Use validation to choose
  settings, not to fit normalization, features or model weights.
- Use filenames and metadata for managing examples and labels only. Never use
  filenames, pulse IDs, labels, SNR settings or storage locations as model inputs.
- Crop using a fixed rule or signal-derived detector, not filename pulse widths.
  Keep validation captures out of training augmentation.

This evaluates new pulses from known configurations, not unseen configurations
or recording sessions. Source filenames expose labels, so this compact release
is an open development dataset, not a blind test set.

## 6. Develop and evaluate

Start with a manageable training subset covering all seven classes; record its
selection rule. Build a complete training-to-prediction pipeline and evaluate
on validation before scaling. Record seeds, SNRs, settings, dataset version,
runtime and hardware. Freeze preprocessing and model choices before final testing.

Training may use a subset of SNRs. The blind test evaluates across all
13 settings. At each SNR, average recall across the seven classes. The main
score averages those balanced accuracies across all 13 SNRs. Also report per-SNR
scores and confusion matrices. A reduced class or SNR subset is not a full score.

Report **requested SNR (dB)**. Preprocessing estimates noise over 0–20 µs and
signal power over 21–119 µs after subtracting noise power. That signal window
includes padding for shorter pulses. Adding noise cannot improve source SNR,
so unattainable higher targets receive no additional noise. Values are clipped
to int16 limits; metadata records clipping and unattainable-target counts.
Requested SNR therefore need not equal the achieved active-pulse SNR.

## 7. Final evaluation and submission

### Blind test contents

The blind test is now available from the [UCL RadarML Lite Blind Test Release](https://rdr.ucl.ac.uk/articles/dataset/RadarML_-_Lite_-_Blind_Test_Release/34065705?file=69638289).
Download `blind_test_participants.zip` from that record after freezing your model
and preprocessing. The large signal files are not stored in this Git repository.

| Property | Released blind test |
| --- | --- |
| Waveform configurations | 660 |
| Held-out source pulses per configuration | 20 |
| Total held-out source pulses | 13,200 |
| Requested SNR settings per pulse | 13 |
| Total test examples | 171,600 |
| Modulation classes | `barker`, `cw`, `fmcw`, `fsk`, `hyp`, `nlfm`, `quad` |
| Requested SNRs (dB) | +30, +25, +20, +15, +10, +5, 0, −5, −10, −15, −20, −25, −30 |
| Receiver and sample format | ADC0; 16,800 complex samples at 120 MS/s; interleaved int16 I/Q |
| Participant ZIP size | 11,546,404,564 bytes: approximately 11.55 GB / 10.75 GiB |
| Disk space for ZIP plus extracted data | Approximately 23.1 GB |

Every class appears at every requested SNR. Class counts are unequal. Test
pulses have no source-identity or exact-signal overlap with the published
training/validation data. They come from the same waveform configurations;
this is not an unseen-configuration or unseen-hardware test.

The participant package contains randomly ordered signal shards, anonymous IDs,
a compatible loader, a blank submission template, checksums and aggregate
preprocessing diagnostics. Per-example class labels, SNRs, original source
identities and the answer key remain private with the organizers.

The UCL record reports archive MD5 `8307e8afebbf00ce24464672187e571d`.
Check the downloaded ZIP against this value and verify extracted files using
the bundled `checksums.json`.

### When participants receive it

The release sequence follows the [event timetable](EVENT_RULES.md):

1. Start with the compact training/validation dataset and shared 80/20 manifest.
2. Train and tune using training and validation data only.
3. Freeze the model, preprocessing and settings; record or submit a fixed code
   revision and model artifact as directed by the organizers.
4. Receive the blind test, run inference without further training or tuning,
   and submit predictions for organizer scoring.

Freeze models by **Day 4, 17:00**. The blind test is released on **Day 5, 09:00**;
final submissions close on **Saturday 17 October 2026, 12:00 noon BST (11:00 UTC)**. Providing test signals at the start
would allow inspection to influence development and weaken the independence of
final evaluation. Labels remain private; upload predictions to Kaggle for assessment.
Do not use assessment feedback to change the frozen pipeline.

### Organizer publication checklist

1. Verify the [released blind-test archive](https://rdr.ucl.ac.uk/articles/dataset/RadarML_-_Lite_-_Blind_Test_Release/34065705?file=69638289) is publicly downloadable.
2. Keep answer keys, original source mappings and private build files out of public downloads and this repository.
3. Configure the [Kaggle competition](https://www.kaggle.com/t/8c86ae869036416784b3f5521d5f6717) to accept `sample_id,label` and use the answer key corresponding to this archive.
4. Set the Kaggle deadline to **Saturday 17 October 2026, 12:00 noon BST (11:00 UTC)**; verify the assessment metric and submission settings.
5. Ask teams to disclose early test access and changes after model freeze.

Do not recover held-out labels by matching signals to the public original archive.

### Matching preprocessing and clipping

The blind test uses the same SNR levels, power-estimation windows, noise-generation
method and int16 clipping rules as the main dataset, with independent noise seeds.
An int16 component is limited to −32,768 through +32,767. At low requested SNR,
added noise can exceed these limits and is capped, changing the noise distribution.

Use the released archive's `preprocessing_diagnostics.json` for clipping and
unattainable-target counts. Report **requested SNR**, not guaranteed achieved
active-pulse SNR.

### Load and submit

From the extracted `blind_test` folder, with NumPy installed:

```python
from participant_loader import Samples

samples = Samples("test")
iq, info = samples[0]
print(iq.shape, info["sample_id"])  # (16800,), anonymous ID
```

This loader supports the test shards; continue using the compact-data loading
instructions above for training/validation. Fill `submission_template.csv`,
preserve every `sample_id`, and predict
exactly one permitted label for each test example, using columns:

```csv
sample_id,label
```

### Upload predictions to Kaggle

1. Sign in to the [Radar ML Hackathon competition](https://www.kaggle.com/t/8c86ae869036416784b3f5521d5f6717), join it and accept its rules if prompted.
2. Run your frozen pipeline on every blind-test sample and save `predictions.csv` with exactly `sample_id,label` as the header. Use no spaces around the comma and omit any pandas index column.
3. If the archive template uses `predicted_label`, rename that column to `label` for the Kaggle upload. Preserve every sample ID and include each exactly once. Allowed labels are `barker`, `cw`, `fmcw`, `fsk`, `hyp`, `nlfm`, `quad`.
4. Use the competition's submission/upload control to upload `predictions.csv`. Identify your team and track in the description. Upload predictions, not the signal archive or answer key.
5. Check that Kaggle accepts the submission and completes assessment. Correct any format or ID errors before **Saturday 17 October 2026, 12:00 noon BST (11:00 UTC)**, using the same frozen pipeline.

Save the completed file as `predictions.csv` and submit it to the Kaggle site.
Missing, duplicate or extra IDs, empty predictions and unknown labels are rejected.
The challenge is scored using balanced accuracy at each SNR, averaged across
all 13 requested SNR settings. The answer key is not distributed to teams.
Follow the [submission limits and frozen-pipeline requirements](EVENT_RULES.md#submission-limits-and-required-files).

Provide reproduction instructions, code/settings, model, track, training subset
and SNRs, score curves, confusion matrices and a short account of what worked.
Report training time, inference time per pulse, hardware and model size. Include
signal preprocessing in inference timing.

## Attribution

Ritchie, White and Hosford (2025),
[RadarML, version 1](https://doi.org/10.5522/04/30752767.v1).
See the [overview](../README.md#attribution-and-licensing) for licensing references
and [contact details](../README.md#contact) for questions.
