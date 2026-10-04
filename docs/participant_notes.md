# Radar modulation recognition hackathon

Participant briefing and presenter notes. Each numbered section can be used as a slide topic.

See the [participant guide](../hackathon/README.md) and [event rules](../hackathon/EVENT_RULES.md) for the authoritative workflow. Day 0 is kick-off; freeze on Day 4 at 17:00, blind-test release on Day 5 at 09:00, and final submissions on Day 6 at 17:00. All times are Europe/London.

## 1. Your challenge

- Take a recording of one radar pulse from one receiver.
- Predict one of seven modulation labels.
- Measure how recognition changes as noise increases.
- Develop either a time-domain method or a time-frequency method.

The task is waveform classification: identify the modulation family from the recorded signal. You are not estimating target range, identifying an aircraft, or locating an emitter. A successful solution combines useful signal processing, a reproducible model, and honest evaluation across noise conditions.

## 2. Where the data comes from

RadarML contains real radar waveforms recorded over the air using UCL's ARESTOR platform. Waveform configurations vary in modulation, duration, bandwidth and centre frequency. The hackathon uses a reduced, single-receiver version of these recordings. Source attribution: Ritchie, White and Hosford, [RadarML Dataset, version 1 (2025)](https://doi.org/10.5522/04/30752767.v1).

The recordings already contain receiver noise and acquisition effects. Additional simulated noise is used to create the requested SNR conditions. This makes the task more realistic than classifying entirely ideal mathematical waveforms, but does not reproduce every possible receiver, propagation environment or interference source.

## 3. What is modulation?

A radar waveform can change its amplitude, phase or frequency over time. These changes create patterns that help distinguish waveform families.

- A tone has approximately constant frequency during its active interval.
- A chirp sweeps through frequency during the pulse.
- Frequency switching moves between discrete frequency values.
- Phase coding introduces controlled phase changes within the pulse.

These are general waveform concepts, not guaranteed visual templates for every noisy recording. Different parameter settings within a family can look quite different. Different families can also look similar over a short or noisy observation.

## 4. The seven output labels

Use these exact, lowercase strings:

| Label | How to treat it in this challenge |
| --- | --- |
| `barker` | One class combining all six supplied Barker-code subtypes |
| `cw` | A separate source-dataset class |
| `fmcw` | A separate source-dataset class |
| `fsk` | A separate source-dataset class |
| `hyp` | A separate source-dataset class |
| `nlfm` | A separate source-dataset class |
| `quad` | A separate source-dataset class |

The archive's labels define the classification problem. Do not merge the frequency-modulated classes or split Barker subtypes into extra classes. Learn the differences from labeled training examples. These notes retain the source labels without asserting unverified expansions or precise waveform equations for each class.

## 5. Understanding I and Q

The receiver records two components of one complex signal:

`x[n] = I[n] + j Q[n]`

- I is the in-phase component; Q is the quadrature component.
- Magnitude is `sqrt(I² + Q²)`.
- Phase is `atan2(Q, I)`.
- Changes in phase carry information about frequency.

I and Q are two components of the same receiver measurement, not two receiver channels. Preserving both lets you retain phase and frequency information that magnitude alone discards. Numerical amplitudes are ADC units; do not interpret them as calibrated volts or RF power without calibration information.

For adjacent samples, a useful phase-change quantity is `angle(x[n] * conj(x[n-1]))`. Multiplying this by `sample_rate / (2*pi)` gives a local frequency estimate under the usual sampling assumptions. It becomes unreliable where signal amplitude is small or noise dominates.

## 6. What one example contains

| Property | Value |
| --- | --- |
| Receiver channels | 1; ADC0 in the prepared dataset |
| Complex samples per example | 16,800 |
| Complex sample rate | 120,000,000 samples/second |
| Record duration | 140 microseconds |
| Stored values | 33,600 interleaved signed 16-bit integers |
| Stored order | `I0, Q0, I1, Q1, ...` |
| Loader output | A complex64 array of shape `(16800,)` |

The 140-microsecond record includes the pulse and surrounding noise. It is not necessarily a 140-microsecond active pulse. Time is `n / sample_rate`; the interval between complex samples is approximately 8.33 nanoseconds. The I/Q storage order does not double the complex sample rate.

## 7. Dataset size and repeated noise conditions

The complete processed collection contains 660 waveform configurations, with 100 source pulses per configuration and 13 requested SNR conditions per pulse:

`660 × 100 × 13 = 858,000 examples`

That is 66,000 distinct retained source pulses, each represented at 13 requested noise settings. The 858,000 examples are not independent physical captures.

Signal arrays occupy about 57.66 GB in decimal units, or 53.70 GiB. One receiver, fewer pulses and fewer SNR levels reduce storage substantially, but storing all noise variants still takes space. Read batches from disk rather than loading the complete collection into memory. The separate blind release adds 171,600 held-out examples; its `dataset.json` describes that package.

## 8. Signal-to-noise ratio: what the numbers mean

`SNR_dB = 10 log10(signal power / noise power)`

| SNR | Signal-to-noise power ratio |
| --- | --- |
| +20 dB | 100:1 |
| +10 dB | 10:1 |
| 0 dB | 1:1 |
| -10 dB | 1:10 |
| -20 dB | 1:100 |

The 13 requested settings are `+30, +25, +20, +15, +10, +5, 0, -5, -10, -15, -20, -25, -30 dB`.

Negative SNR means noise has more power than the signal under the measurement definition. It does not mean every useful pattern disappears: structure across many samples can still help. Expect the difficult noise conditions to expose differences between methods.

## 9. Requested SNR is not guaranteed achieved SNR

The blind test uses the same preprocessing rules as training, with independent noise seeds. Aggregate I/Q-value clipping is 5.38% at −30 dB and 0.239% at −25 dB. These are component-value rates, not percentages of pulses. Full diagnostics accompany the test download.

The preprocessing estimates noise and signal power using fixed windows and adds noise to target an SNR. Adding noise cannot improve the original recording. Where a requested setting is above the estimated source SNR, the recording is left unchanged; some nominal settings can therefore contain the same signal values.

The power-estimation windows are 0–20 microseconds for noise and 21–119 microseconds for the signal region. The signal window includes padding for shorter pulses. Consequently, a nominal SNR is not necessarily the SNR within the active pulse alone. Values are also clipped to the int16 range during preprocessing, which can affect noisy examples.

Label figures and results **requested SNR (dB)**. Do not present the horizontal axis as an independently calibrated measurement of achieved SNR.

## 10. Choose a track

| Track A: time domain | Track B: time-frequency |
| --- | --- |
| Work with ordered I/Q samples | Compute a representation from those same I/Q samples |
| May use magnitude, phase differences, normalization and time-domain statistics | May use STFT spectrograms or other time-frequency representations |
| Learned 1D convolution filters are allowed | Choose your own model and representation settings |
| No explicit FFT, STFT, wavelet or spectrogram representations | Include representation computation in inference timing |

Track B exposes frequency evolution explicitly and may be an accessible starting point. Track A explores learning from the waveform directly. Neither track is assumed to win. If you enter both, compare them using identical source pulses, training SNR subsets and evaluation conditions. The event uses separate track comparisons.

## 11. How to read a spectrogram

A short-time Fourier transform examines frequency content in successive, overlapping windows. A spectrogram displays its power, often on a logarithmic scale.

- Horizontal axis: time.
- Vertical axis: frequency relative to the receiver's reference.
- Colour: power on the chosen scale.
- A sustained tone can form a horizontal trace; a sweep can form a sloping or curved trace.

Longer windows improve frequency discrimination but smear rapid changes in time. Shorter windows resolve changes more precisely in time but separate nearby frequencies less well. Increasing the FFT size through zero-padding adds plotted bins without increasing the underlying resolving power. Select settings using training and validation data.

Complex I/Q signals can contain positive and negative frequency offsets. Use a two-sided frequency view when appropriate. Do not label the plot as absolute transmit frequency without the required acquisition reference. Independently rescaling each plot's colours can make signals of very different powers look deceptively similar.

## 12. Open the development package

Download `recorded_dataset.zip` from the [UCL Lite dataset record](https://rdr.ucl.ac.uk/articles/dataset/Radar_ML_-_Lite/33977674). Download `split_manifest.csv` and `split_manifest.json` separately and place them beside the extracted recordings. Repository copies are in `hackathon/`.

Compact arrays have shape `(1, 100, 33600)` and retain original recorded noise without synthetic augmentation. Expansion creates `(1, 13, 100, 33600)` arrays. Use the scripts bundled with the ZIP together. Start with the small expansion exercise in the guide before generating 57.66 GB of expanded arrays.

The blind test is a separate package, released after model freeze. Its anonymous shards have shape `(N, 33600)` and use `participant_loader.py`; that loader is not for the compact training arrays. Never create a new random split of expanded examples.

## 13. First exercise: load a training pulse

Run inside the extracted `recorded_dataset` folder with NumPy installed:

```python
import csv
import numpy as np

with open("split_manifest.csv", newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream))
example = next(r for r in rows if r["split"] == "train")
array = np.load(example["file"], mmap_mode="r", allow_pickle=False)
raw = array[0, int(example["pulse_index"]), :]
iq = raw[0::2].astype(np.float32) + 1j * raw[1::2].astype(np.float32)
time_us = np.arange(iq.size) / 120_000_000 * 1e6
print(iq.shape, iq.dtype)  # (16800,), complex64
```

Plot I, Q and magnitude against time. Training labels come from the waveform token in the filename: group Barker subtypes into `barker` and `fsk#0` through `fsk#3` into `fsk`. Use filenames only to manage labels, never as model inputs. Compact recordings do not have requested-SNR labels. For expanded data, use the JSON SNR ordering and index `array[0, snr_index, pulse_index, :]`.

## 14. Split correctly: prevent leakage

The published manifest assigns 80 training and 20 validation pulses per configuration. All noise variants of a source pulse retain its assignment. The blind test uses separate held-out source pulses; no test pulses are carved out of the 100 development pulses.

| Split | Source pulses | Examples at all 13 SNRs |
| --- | ---: | ---: |
| Training | 52,800 | 686,400 |
| Validation | 13,200 | 171,600 |
| Separate blind test | 13,200 | 171,600 |

Fit models and learned preprocessing using training pulses only. Use validation to choose settings; do not add validation pulses to a final training run. Splitting augmented examples independently leaks source pulses between splits.

The configurations are shared across splits, so the result measures new-pulse recognition within known configurations rather than generalization to new configurations, receivers or environments.

## 15. What your model may use

Train from scratch on supplied training pulses. External training data and pretrained weights are excluded. Standard libraries, papers, public code and AI coding assistants are allowed with attribution/disclosure; do not send test signals to external services. Follow the complete [event rules](../hackathon/EVENT_RULES.md).

- Signal values and fixed acquisition constants, such as sample rate.
- Training labels for supervised learning.
- Transformations permitted by your chosen track.
- Training-only augmentation that preserves the intended class.

Do not use sample IDs, shard names, row numbers, source filenames, metadata labels or requested SNR as model features. Requested SNR is for analysis and evaluation grouping. Do not use the original public dataset to recover held-out labels.

If cropping, use a fixed rule or a detector based on signal values. Never crop using a pulse width obtained from a source filename. Check that augmentation does not change the modulation family or introduce an artificial shortcut. Keep validation and test pulses out of training and augmentation.

## 16. A practical development workflow

1. Verify the loader, labels and array dimensions.
2. Start with a manageable training subset covering all seven classes and record the selection rule.
3. Build a complete training-to-prediction pipeline in your chosen track.
4. Evaluate on validation data, grouped by requested SNR and class.
5. Change one aspect at a time and record whether it helps.
6. Expand training once the pipeline is reliable.
7. Freeze preprocessing and model settings before final holdout evaluation.

Possible experiments include normalization, training SNR selection, class weighting and model capacity. Track B teams can also vary time-frequency resolution. These are open design choices, not a supplied reference method. Keep an experiment log with the data subset, random seed, settings, score and runtime.

The dataset has unequal class counts. Report those counts for your training subset. A small pilot containing only one class is useful for checking loading, but cannot measure seven-class recognition.

## 17. How scoring works

At each requested SNR, calculate recall for each class, then average over the seven classes:

`balanced_accuracy(SNR) = mean(correct predictions for class / number of examples of class)`

The main score is the mean of those balanced accuracies across all 13 requested SNR settings. Each class and each SNR setting receives equal weight in the score. Ordinary accuracy weights examples equally and can be dominated by more common classes.

For orientation, a classifier that always predicts one class scores approximately `1/7 = 14.3%` balanced accuracy when all seven classes are present. That is a reference level, not a measured model result.

Report the main score, a curve of balanced accuracy against requested SNR, and confusion matrices. In the supplied scorer, rows are true labels and columns are predicted labels; the JSON `classes` list gives their order. Do not claim a full challenge score from a reduced class set or a reduced evaluation subset.

## 18. Freeze, evaluate and submit

Freeze one pipeline per entered track by **Day 4, 17:00**. Submit the model, preprocessing/configuration, code revision or archive, dependency versions, inference command and artifact hashes. No retraining or test-driven adaptation is allowed after freeze.

The blind-test ZIP is released on **Day 5, 09:00**. It contains 171,600 examples and is about 11.55 GB. Allow about 23.1 GB for ZIP and extraction. Run from its extracted folder:

```python
from participant_loader import Samples
samples = Samples("test")
iq, info = samples[0]
print(iq.shape, info["sample_id"])
```

Fill `submission_template.csv`, preserving every ID, and save as `predictions.csv` with exactly these columns:

```csv
sample_id,predicted_label
```

Submit by **Day 6, 17:00**. One final entry per team per track is allowed. Packaging corrections before the deadline must use the same frozen pipeline. Labels remain private; organizers score predictions after the deadline without providing intermediate blind-score feedback.

Include a method summary, validation results, training subset and SNRs, seeds, attribution, hardware, training time, inference time including preprocessing, and model size. See [event rules](../hackathon/EVENT_RULES.md) for the full checklist and tie-break: mean balanced accuracy at −20, −25 and −30 dB, then shared placement if still tied within the stated tolerance.

## 19. Questions for your final presentation

- Which classes were easiest and hardest, and how did that change with SNR?
- What improved validation performance, and what did not?
- Did your method use waveform structure or appear to exploit incidental acquisition features?
- What accuracy, speed and memory trade-offs did you make?
- What additional experiment would test generalization to new configurations or receivers?

Show evidence with a score curve, selected confusion matrices and a few representative failure cases. Keep test-based analysis separate from the validation experiments used to select your model.

## 20. Before you finish

- Use the supplied splits and exact seven labels.
- Keep model inputs restricted to permitted signal information.
- Check every test ID has one prediction.
- State the dataset version, track and training subset.
- Report requested SNR, balanced accuracy and runtime clearly.
- Save code, settings and model so the result can be reproduced.

## Provenance of these notes

The source dataset and capture context are described in the [UCL dataset record](https://doi.org/10.5522/04/30752767.v1) and [UCLRadarGroup repository](https://github.com/UCLRadarGroup/Radar_ML). Source links are for attribution and background; they are not permission to recover test labels from the original data.

Hackathon-specific dimensions, rules, split handling, loading and submission details were checked against this project's preprocessing, packaging, participant loader and scorer. Full-collection counts describe the 660-file processed collection; use the published split manifest for development and the separate blind package’s `dataset.json` for final evaluation.
