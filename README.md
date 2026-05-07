# gammabci-models

Trained model weights for replicating *Brain-machine interface control with artificial intelligence copilots* (Lee et al., 2024).

| Artifact | Status | Description |
|---|---|---|
| `lstm_copilot/` | Available | RecurrentPPO copilot, best of 4 seeds (0.98 success rate, 4-class center-out-back, normal_target softmax). Trained 1.2M timesteps on synthetic env. Patient-agnostic. |
| `cnn_kf_H1/` | Available | EEGNet + Kalman filter, trained on patient **H1** session `2024-02-13_H1_CL_1`. 5-fold CV, 68–73% validation accuracy. Released with the original paper. |
| `cnn_kf_replication/` | Coming soon | Replication run pooling all 4 OL sessions (H1, H2, H4, S2). Hoffman2 training in progress. |

## Companion resources

- **Dataset** (EEG/task/gaze for all sessions): https://zenodo.org/records/15165133
- **Code** (RASPy framework, training, inference): https://github.com/kaolab-research/bci_raspy

## Setup

```bash
git clone https://github.com/kaolab-research/bci_raspy.git
cd bci_raspy

# pip dependency chain pin (gym 0.21 needs old setuptools/wheel)
pip install "pip<24.1" "setuptools==65.5.0" "wheel==0.38.4"
pip install --use-deprecated=legacy-resolver -r requirements.txt
```

Place the dataset from Zenodo at `bci_raspy/data/raspy/<session_dir>/` (or symlink).

## Using the LSTM copilot

```bash
# from bci_raspy/
python -m SJtools.copilot.test /path/to/gammabci-models/lstm_copilot/best_model \
  -center_out_back -softmax_type=normal_target
```

`best_model.zip` is a Stable-Baselines3 RecurrentPPO checkpoint. The accompanying `model.yaml`, `reward.yaml`, `best_model.yaml` capture the training hyperparameters; `evaluations.npz` contains the eval-callback reward trace; `log.txt` records the full per-softmax test results.

## Using the CNN-KF (H1)

The decoder loads EEGNet weights + KF params at runtime via the `decoder` RASPy module. Point `decoder_folder` at the unpacked `cnn_kf_H1/` and pick a fold (`0`–`4`):

```yaml
# in your RASPy model YAML
modules:
  decoder:
    params:
      decoder_name: EEGNet
      decoder_folder: /path/to/gammabci-models/cnn_kf_H1
      fold: 0   # or any of 0..4
```

Per-fold validation accuracy (`results.csv`):

| Fold | Train Acc | Val Acc |
|---|---|---|
| 0 | 82.1% | 71.9% |
| 1 | 83.6% | 68.2% |
| 2 | 79.0% | 68.3% |
| 3 | 82.5% | 73.0% |
| 4 | 75.5% | 69.0% |

## File layout

```
lstm_copilot/
├── best_model.zip       # SB3 RecurrentPPO checkpoint (load with sb3_contrib.RecurrentPPO.load)
├── best_model.yaml      # eval metadata for best_model.zip
├── model.yaml           # full training config
├── reward.yaml          # baseLinDistDecay reward config used for training
├── evaluations.npz      # EvalCallback history (timesteps, results, ep_lengths)
└── log.txt              # full training + final-test stdout

cnn_kf_H1/
├── 0, 1, 2, 3, 4        # per-fold EEGNet state_dict (torch.load)
├── 0_kf.npz … 4_kf.npz  # per-fold Kalman filter params
├── EEGNet.py            # model architecture (snapshot — keep alongside weights)
├── instantiate.py       # constructs the model from EEGNet.py at load time
├── config.yaml          # full training config
├── results.csv          # per-fold accuracies
├── losses.pickle        # per-epoch train/val losses
├── labels.txt           # confusion-matrix raw labels
└── confusion_matrix/    # per-fold confusion matrix images
```

## Validation

`test/` contains two scripts that exercise the released weights end-to-end (each model is tested in isolation — see "Limitations" below). They expect a local `bci_raspy/` clone and a dataset checkout, both gitignored — symlink them in before running:

```bash
cd test/
ln -s /path/to/bci_raspy bci_raspy
ln -s /path/to/dataset_root data        # must contain raspy/2024-02-13_H1_CL_1/
```

Then:

```bash
# from gammabci-models/
python test/rollout_lstm.py     # 16 episodes in SJ4DirectionsEnv, headless
python test/infer_cnn.py        # CNN forward pass on 1s cue windows from H1_CL_1
```

Latest results (committed under `test/figures/`):

| Test | Result |
|---|---|
| LSTM rollout | 16/16 hits, mean episode length 337 (`lstm_cursor_paths.png`) |
| CNN cue-aligned 4-class | 84.2% on 133 windows from H1_CL_1, fold 0 (`cnn_predictions.png`, `cnn_confusion_matrix.png`) |

### Limitations

These scripts test the CNN-KF and LSTM copilot **separately**. The CNN test runs the decoder on recorded EEG; the LSTM test runs the copilot in its synthetic training env (`SJ4DirectionsEnv` with `softmax_type=normal_target`, i.e. a simulated decoder). Composing them — running the live RASPy pipeline with recorded EEG replayed into `UpdateEEG`, the CNN as the decoder, and the LSTM as the copilot — is the natural next validation and is not in this repo yet. The path: write a `stream_replay.py` (analog of `bci_raspy/stream/stream_fake.py`) that pumps `eeg.bin` over TCP at 1000 Hz, then run a RASPy YAML wiring `UpdateEEG → filterEEG → decoder (this CNN) → kf_clda → copilot (this LSTM) → task_module`.

## Citation

If you use these weights, please cite the original paper and the dataset:

- Lee et al., *Brain-machine interface control with artificial intelligence copilots* (2024).
- Dataset: https://zenodo.org/records/15165133
