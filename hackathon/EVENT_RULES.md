# RadarML Lite hackathon timetable and event rules

All times use **Europe/London** local time.

**Final predictions and reports are due Saturday 17 October 2026 at 12 noon BST (11:00 UTC).**

Day numbers refer to the original event timetable. The announced Day 0 mapping and **Day 4, 17:00 model-freeze deadline** remain in effect.

## Timetable

| Day and time | Activity | Required outcome |
| --- | --- | --- |
| Day 0, 09:00 | Kick-off and development-data access | Download the compact training dataset and split manifests, check loading and choose your tracks. |
| Day 1 | First complete pipeline | Train an initial model and produce validation predictions. |
| Day 2 | Signal processing and model experiments | Compare preprocessing and models using validation only. Record settings and seeds. |
| Day 3 | Robustness and efficiency | Evaluate across requested SNRs and measure runtime and memory requirements. |
| Day 4, before 17:00 | Final training and validation | Finish training, select the method and save validation results. |
| **Day 4, 17:00** | **Model freeze** | Freeze one pipeline per entered track and provide model artifacts, preprocessing, code/settings, dependencies, inference commands and artifact hashes. |
| Day 5, 09:00 | Blind-test release | Download and verify the blind-test archive, then run the frozen pipelines. |
| After blind release until the final deadline | Final inference and report preparation | Complete predictions, check the submission format and prepare supporting materials. No test-driven tuning. |
| **Saturday 17 October 2026, 12 noon BST (11:00 UTC)** | **Final submission deadline** | Upload predictions to Kaggle and provide supporting materials to the organizers. |
| After the final deadline | Final scoring and wrap-up | Apply the ranking rules and complete reproducibility checks. |

Allow approximately **23.1 GB** for the blind-test ZIP and extracted files.

If organizer-side download or submission infrastructure fails, any revised deadline will be announced to all teams. Individual extensions are not part of the standard rules.

## Tracks and Kaggle team names

Each participating group must create two separate Kaggle teams using the same group name:

| Kaggle team name | Track | Permitted approach |
| --- | --- | --- |
| `name_time` | Track A: time domain | Ordered I/Q, magnitude, phase differences, normalization, time-domain statistics and learned 1D convolutions. No explicit FFT, STFT, wavelet or spectrogram representations. |
| `name_freq` | Track B: time-frequency | Representations computed from the same I/Q, such as STFT spectrograms, with the chosen classifier. |

For example, a group called `RadarFox` must use `RadarFox_time` and `RadarFox_freq`.

Upload time-domain-only predictions through the `_time` team and time-frequency predictions through the `_freq` team. Do not mix the two tracks within a team entry. Identify both team names in the supporting report.

Each group may submit **one final entry per track**, using one frozen pipeline for each track entered. Tracks are ranked separately.

Include time-frequency representation generation in inference timing.

## Training data and permitted tools

- Use the supplied RadarML Lite recordings and published **80/20 training/validation split**.
- Fit models, learned normalization, feature transforms and calibration using **training pulses only**. Use validation for evaluation and selecting settings. Do not add validation pulses to a final training run.
- Every SNR variant and augmentation inherits its original source pulse’s split.
- Training on a subset of the supplied training pulses or SNR settings is allowed. Report the selection rule. Final evaluation covers every supplied blind-test example at all 13 requested SNR settings.
- Train models from scratch. External training datasets, additional recordings from the original RadarML archive, pretrained weights and external learned feature extractors are not permitted.
- Augmentation derived from supplied training pulses is allowed when it preserves the intended class. Additional independently generated labeled waveform datasets are excluded.
- Standard libraries, papers, tutorials, public code and AI coding assistants may be used. Cite reused methods and code, and disclose AI assistance in the report. These tools do not permit external learned weights or extra training data.
- Do not upload blind-test signals to external services for analysis or prediction.
- CPUs, GPUs and cloud compute are allowed. Declare hardware and training time. Self-reported runtime is not a ranking tie-break.
- Model inputs must be permitted signal representations and fixed acquisition constants. Do not use filenames, sample IDs, source metadata or requested-SNR labels as predictive inputs.
- Do not recover blind-test labels by matching signals to the original public archive.

## Model freeze and blind evaluation

By **Day 4 at 17:00**, provide one frozen pipeline per entered track, including all ensemble members where applicable.

Supply:

- Saved model weights.
- Preprocessing and configuration settings.
- A code archive or fixed Git commit.
- Dependency versions.
- A command that reproduces inference.
- SHA256 hashes of the model and configuration files.

A code commit alone is not a substitute for accessible model weights and settings.

After freeze, no retraining, threshold tuning, model selection, ensemble changes, test-time training or fitting preprocessing to test data is allowed.

Fixed per-pulse processing defined before freeze, such as normalizing an individual pulse by its own RMS, remains allowed. Run models in inference mode.

An execution-only repair, such as a dependency or file-path correction, requires organizer agreement and a recorded diff. It must not alter weights, preprocessing or prediction logic. Retain the original frozen artifacts.

The blind dataset is available from the [UCL RadarML Lite Blind Test Release](https://rdr.ucl.ac.uk/articles/dataset/RadarML_-_Lite_-_Blind_Test_Release/34065705?file=69638289). Use it only after model freeze and the scheduled release. Disclose any earlier test access to the organizers; they will determine eligibility for the blind ranking.

The answer key, per-example SNRs and source mappings remain private.

## Submission format and required materials

Upload predictions through the [Kaggle competition](https://www.kaggle.com/t/8c86ae869036416784b3f5521d5f6717), using the team corresponding to the track.

Save each prediction file as `predictions.csv` with exactly this header:

```csv
sample_id,label
```

Requirements:

- Include every blind-test sample ID exactly once.
- Preserve the supplied sample IDs.
- Use only `barker`, `cw`, `fmcw`, `fsk`, `hyp`, `nlfm` or `quad`.
- If the supplied template uses `predicted_label`, rename that column to `label`.
- Do not include an extra index column.
- Missing, duplicate or extra IDs, empty predictions and unknown labels make a submission invalid.

Identify the group and track in the Kaggle team name and submission description, not by adding prediction columns.

Provide the following through the organizer-designated channel:

1. Both Kaggle team names and the track associated with each submitted file.
2. Frozen pipeline identifiers and artifact hashes.
3. Any approved execution-only repair record.
4. A method summary describing training data/SNR subsets, random seeds, validation scores and confusion matrices.
5. Models, code/settings, dependency versions and reproduction instructions.
6. Code attribution and any AI-assistance disclosure.
7. Hardware, training time, inference time per pulse and model size. Include signal preprocessing and representation generation in inference timing.

Do not upload the signal archive or answer key as a prediction submission. Do not publish private organizer materials in the public repository.

## Submission corrections and assessment feedback

Before **Saturday 17 October 2026 at 12 noon BST**, a group may replace a submission to correct formatting, packaging or an incomplete inference run using the **same frozen pipeline**.

The latest valid submission for each group and track received before the deadline is the final entry. This does not permit testing multiple models.

Check that Kaggle accepts the file and completes assessment before the deadline. Do not use assessment feedback to tune the pipeline, select another model or change preprocessing.

No late replacements are accepted unless an event-wide infrastructure extension is announced.

## Scoring and tie-breaking

Rank Track A and Track B separately.

The primary score is the mean of the **seven-class balanced accuracies across all 13 requested SNR settings**, with equal weight per class and SNR.

Compare scores before display rounding. Differences of at most **0.000001** on the 0–1 score scale count as a tie.

For a primary-score tie, compare mean balanced accuracy over the three lowest requested SNRs: **−20, −25 and −30 dB**, using the same tolerance.

If those scores also tie, award a shared place. Submission time and self-reported runtime are not tie-breaks.

Final overall scores, per-SNR scores and confusion matrices are published after the deadline. Per-example answers are not published during the event.

Rankings are conditional on compliance with the data and track rules and reproducibility of the frozen pipeline.

## Organizer checks

- Configure the Kaggle deadline as **17 October 2026 at 11:00 UTC**, equivalent to noon BST.
- Verify that the competition accepts `sample_id,label` and uses the private answer key corresponding to the released archive.
- Verify that assessment implements the stated scoring metric.
- Confirm that the Kaggle team configuration supports the `_time` and `_freq` entries and communicate the membership procedure.
- Verify access to the development data, split manifests and released blind-test archive.
- Keep answer keys, source mappings and private build files out of public downloads and repositories.
- Confirm that frozen artifacts and supporting reports are accessible.
- Apply separate track rankings and the stated tie-breaking rules after submissions close.
