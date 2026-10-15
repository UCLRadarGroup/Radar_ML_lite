# RadarML Lite

RadarML contains radar waveforms recorded over the air using UCL's ARESTOR
RFSoC platform. This repository supports a reduced, single-receiver dataset for
learning and comparing radar modulation classifiers. Configurations vary in
modulation, pulse duration, bandwidth and centre frequency.

**Taking part in the hackathon? Start with the [Hackathon guide](hackathon/README.md)**
for setup, signal loading, SNR expansion, evaluation and submission guidance.

## The reduced dataset

The compact archive preserves recorded samples without adding synthetic noise.
Participants can generate different signal-to-noise ratio (SNR) conditions locally,
keeping the download smaller than a dataset containing every noise variant.
Original receiver noise remains; these are not noise-free signals.

| Property | RadarML Lite compact release |
| --- | --- |
| Waveform configurations | 660 |
| Receiver | ADC0, one of the original three channels |
| Pulses per configuration | First 100 recorded pulses, source indices 0–99 |
| Total pulses | 66,000 |
| Classes | `barker`, `cw`, `fmcw`, `fsk`, `hyp`, `nlfm`, `quad` |
| Samples per pulse | 16,800 complex samples at 120 MS/s |
| Array format | int16, `(1, 100, 33600)`, interleaved I/Q |
| Compact ZIP | Approximately 2.54 GB |
| Extracted signal arrays | Approximately 4.44 GB |

The original archive has three receivers, ADC0, ADC2 and ADC4, with 1,000 pulses
per receiver for 648 configurations and 8,000 for the remaining 12. RadarML Lite
selects one receiver and 100 pulses from each configuration.

Expanding to 13 requested SNR settings (+30 to −30 dB in 5 dB steps) creates
858,000 examples and approximately 57.66 GB of arrays. The compact arrays use
13 times less storage, a reduction of 92.3%. Noise variants are not additional
independent captures.

## Download and quick start

At the start of the hackathon:

1. Download **`recorded_dataset.zip`** from the
   [UCL Radar ML - Lite dataset page](https://rdr.ucl.ac.uk/articles/dataset/Radar_ML_-_Lite/33977674).
2. Download `split_manifest.csv` and `split_manifest.json` separately from that
   page. If they are not yet publicly visible, use the repository copies:
   [CSV](hackathon/split_manifest.csv) and [JSON](hackathon/split_manifest.json).
   On GitHub, use **Download raw file** to save the actual file, not an HTML page.
3. Extract the ZIP and put both manifest files inside the extracted
   `recorded_dataset` folder, alongside the `.npy` files. No re-zipping is needed;
   the original ZIP and its checksum remain unchanged.

**Do not download, inspect or use `blind_test_participants.zip` until the
organizers announce final evaluation after model freeze**, even if it is visible
on the same download page. Train and tune only on the supplied 80/20 split.
The original full RadarML archive is not needed.

Open a terminal inside the extracted `recorded_dataset` folder.
With Python 3.9 or newer installed:

```bash
python -m pip install numpy
python compact_dataset.py expand --input . --output ../expanded
```

The expansion scripts are **included in the compact ZIP**. Use those bundled
scripts together; the upstream original preprocessing scripts expect a
different input format. Full expansion requires **57.66 GB of additional space**.
See the [guide](hackathon/README.md) for environment setup, a small first check
and loading examples before expanding everything.

To clone this repository:

```bash
git clone https://github.com/UCLRadarGroup/Radar_ML_lite.git
```

## Blind final evaluation

The blind-test dataset is available from the [UCL RadarML Lite Blind Test Release](https://rdr.ucl.ac.uk/articles/dataset/RadarML_-_Lite_-_Blind_Test_Release/34065705?file=69638289). Download `blind_test_participants.zip` and allow approximately **23.1 GB** for the ZIP and extracted files.

The dataset contains **171,600 examples**, generated from **13,200 held-out source pulses across 660 waveform configurations**. Each source pulse appears at 13 requested SNR settings, from +30 to −30 dB in 5 dB steps. All seven modulation classes appear at every SNR. Evaluation measures unseen pulses from known configurations, rather than unseen configurations or hardware.

Freeze your model and preprocessing before accessing the blind-test signals. Run inference using the frozen pipeline; do not retrain, tune or use assessment feedback to change it. Labels, per-example SNRs, source mappings and the answer key remain private.

### Teams and track submissions

Submit predictions through the [Kaggle competition](https://www.kaggle.com/t/8c86ae869036416784b3f5521d5f6717), using separate team names for the two tracks:

| Kaggle team name | Track |
| --- | --- |
| `name_time` | Track A: time-domain-only classification |
| `name_freq` | Track B: time-frequency methods |

Replace `name` with the same group name for both teams. For example, `RadarFox_time` and `RadarFox_freq`. Upload each track’s results through its corresponding team. Identify both team names in your supporting report.

### Prediction format

Save predictions as `predictions.csv` with exactly this header:

```csv
sample_id,label
```

Include every blind-test sample ID exactly once and use only `barker`, `cw`, `fmcw`, `fsk`, `hyp`, `nlfm` or `quad`. If the supplied template uses `predicted_label`, rename that column to `label`. Do not include an extra index column.

Missing, duplicate or extra IDs, empty predictions and unknown labels are rejected.

Each group may submit **one final entry per track**. Check that Kaggle accepts and assesses the file before the deadline. Format corrections must use the same frozen pipeline. Provide supporting models, reports and reproduction materials through the organizer-designated channel.

### Deadline and scoring

**Final predictions and reports are due Saturday 17 October 2026 at 12 noon BST (11:00 UTC).**

The primary score is balanced accuracy at each requested SNR, averaged across all 13 settings. Tracks are ranked separately.

See the [hackathon guide](hackathon/README.md) for loading instructions and the [event rules](hackathon/EVENT_RULES.md) for submission and ranking requirements.


## Repository contents

| Path | Purpose |
| --- | --- |
| [hackathon/README.md](hackathon/README.md) | Participant walkthrough and event protocol |
| [hackathon/EVENT_RULES.md](hackathon/EVENT_RULES.md) | Relative-day timetable, data/model rules, submission limits and tie-breaking |
| [hackathon/split_manifest.csv](hackathon/split_manifest.csv) | Fixed training/validation pulse assignments |
| [hackathon/split_manifest.json](hackathon/split_manifest.json) | Split algorithm, indexing and CSV checksum |
| [hackathon/participant_loader.py](hackathon/participant_loader.py) | Loader for anonymous blind-test shards |
| [hackathon/score_submission.py](hackathon/score_submission.py) | Organizer scoring utility; no answer key is included |
| [docs/participant_notes.md](docs/participant_notes.md) | Updated slide-ready background notes, I/Q theory and current event workflow |
| [hackathon/compact_dataset.py](hackathon/compact_dataset.py) | Build compact recordings or expand them into requested SNR conditions |
| [hackathon/make_dataset_hackathon.py](hackathon/make_dataset_hackathon.py) | Shared noise preprocessing and original-array preparation |
| [hackathon/create_split_manifest.py](hackathon/create_split_manifest.py) | Reproducible split-generation utility; participants use the published manifest |
| [hackathon/plot_modulations.py](hackathon/plot_modulations.py) | Plot original or expanded recordings; compact inspection is covered in the guide |
| [tests/](tests/) | Compact conversion, preprocessing and scoring checks |

The [original RadarML repository](https://github.com/UCLRadarGroup/Radar_ML)
contains the original multi-channel scripts and example notebook. Their data
layout and split handling differ from this compact hackathon workflow. Use the
expansion scripts bundled in the compact ZIP. NumPy is sufficient for loading
and expansion; install libraries for your method as needed.

## Attribution and licensing

Source: Ritchie, White and Hosford (2025),
[RadarML, version 1](https://doi.org/10.5522/04/30752767.v1).
See [LICENSE](LICENSE) for repository code terms and the dataset record for
dataset licensing. Code licensing does not replace dataset terms.

## Contact

Open a GitHub issue or contact m.ritchie@ucl.ac.uk.
