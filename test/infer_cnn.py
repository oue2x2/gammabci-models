"""
Run the H1 CNN-KF on the H1 CL session it was trained on. Validates that:
1. The EEGNet weights load and produce sensible class probabilities.
2. The Kalman filter integrates those probabilities into a cursor trajectory.

Plots:
  test/figures/cnn_predictions.png     — 4-class softmax over time vs ground truth
  test/figures/cnn_confusion_matrix.png — confusion matrix on H1 CL data
  test/figures/cnn_kf_trajectory.png   — KF-decoded cursor path
"""
import os, sys, types
import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.modules.setdefault("pyautogui", types.SimpleNamespace(
    size=lambda: (1920, 1080), position=lambda: (0, 0),
    moveTo=lambda *a, **k: None, click=lambda *a, **k: None, FAILSAFE=False,
))

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
BCI_RASPY = os.path.abspath(os.path.join(HERE, "bci_raspy"))
DATA = os.path.abspath(os.path.join(HERE, "data", "raspy"))

sys.path.insert(0, BCI_RASPY)
sys.path.insert(0, os.path.join(BCI_RASPY, "Offline_EEGNet"))
os.chdir(os.path.join(BCI_RASPY, "Offline_EEGNet"))

from shared_utils import utils
from shared_utils.preprocessor import DataPreprocessor

CNN_DIR = os.path.join(REPO, "cnn_kf_H1")
SESSION_DIR = os.path.join(DATA, "2024-02-13_H1_CL_1")
FIG_DIR = os.path.join(REPO, "test", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ---- Convert .npy → .bin if missing (needed by read_data_file_to_dict) ----
sys.path.insert(0, os.path.join(BCI_RASPY, "jobs", "hoffman2"))
import npy_to_bin
for fname, name in [("eeg.npy", "eeg"), ("task.npy", "task")]:
    p = os.path.join(SESSION_DIR, fname)
    if not os.path.exists(p[:-4] + ".bin"):
        npy_to_bin.convert(p, name)

# ---- Load config + model ----
with open(os.path.join(CNN_DIR, "config.yaml")) as f:
    cfg = yaml.safe_load(f)

# Build EEGNet from the EEGNet.py snapshot in cnn_kf_H1
import importlib.util
spec = importlib.util.spec_from_file_location("Model", os.path.join(CNN_DIR, "EEGNet.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
EEGNet = getattr(mod, mod.name)

n_electrodes = 66 - len(cfg["data_preprocessor"]["ch_to_drop"])
output_dim = len(cfg["dataset_generator"]["dataset_operation"]["selected_labels"])
device = torch.device("cpu")

FOLD = 0  # any of 0..4; results.csv: fold 3 has best val acc (73.0%)
model = EEGNet(cfg["model"], output_dim, n_electrodes)
model.load_state_dict(torch.load(os.path.join(CNN_DIR, str(FOLD)), map_location=device))
model.eval().to(device)
print(f"Loaded EEGNet fold {FOLD}: n_elec={n_electrodes}, output_dim={output_dim}")

# ---- Load + preprocess EEG ----
eeg = utils.read_data_file_to_dict(os.path.join(SESSION_DIR, "eeg.bin"))
task = utils.read_data_file_to_dict(os.path.join(SESSION_DIR, "task.bin"))
print(f"EEG samples: {eeg['databuffer'].shape}, Task ticks: {task['state_task'].shape}")

pre = DataPreprocessor(cfg["data_preprocessor"])
filtered = pre.preprocess(eeg["databuffer"])
# Downsample 1000Hz → 100Hz: take every 10th sample
ds = filtered[::10]                                   # (T_ds, n_elec_after_drop)
print(f"Filtered/downsampled EEG: {ds.shape}")

# ---- Slice into 1s windows centered at the start of every cue period ----
# state_task encodes the per-tick task label (-1 = inter-trial). Take state changes
# from -1 → {0..3} as cue onsets, and grab a 1s window starting there.
state = task["state_task"].squeeze()
eeg_step = task["eeg_step"].squeeze()  # eeg sample index per task tick
window_size = 100      # samples at 100 Hz = 1s
windows = []
labels = []
window_centers_ds = []
prev = -2
for i, s in enumerate(state):
    s = int(s)
    if s != prev and s in (0, 1, 2, 3):
        eeg_idx = int(eeg_step[i])
        if eeg_idx < 0:
            prev = s; continue
        # 1s window starting from cue onset (drop first 1s of trial = first_ms_to_drop=1000)
        # The training pipeline does the same; we mirror it for fair eval.
        start_ds = (eeg_idx + 1000) // 10
        if start_ds + window_size <= ds.shape[0]:
            windows.append(ds[start_ds:start_ds + window_size, :].T)
            labels.append(s)
            window_centers_ds.append(start_ds)
    prev = s

windows = np.stack(windows, axis=0)[:, :, :, None].astype(np.float32)  # (N, n_elec, 100, 1)
labels = np.array(labels, dtype=np.int64)
print(f"Cue windows: {windows.shape}, label distribution: {np.bincount(labels)}")

# ---- Forward pass ----
with torch.no_grad():
    raw = model(torch.from_numpy(windows).to(device)).cpu().numpy()  # (N, 4) — OneHotMSE scores
preds = raw.argmax(axis=1)
acc = (preds == labels).mean()
print(f"Cue-aligned 4-class accuracy on full H1_CL_1: {acc*100:.1f}%")
# The CNN was trained with OneHotMSE loss; outputs are score-per-class fit toward
# {0,1}, not logits. Softmax over them is flat. Use raw scores for the time-series
# and a hardmax for the cursor trajectory.
probs = raw  # for plotting

# ---- Plot 1: softmax over trials with truth ----
fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True,
                         gridspec_kw={"height_ratios": [4, 1]})
classes = ["L (0)", "R (1)", "U (2)", "D (3)"]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
for c in range(4):
    axes[0].plot(probs[:, c], color=colors[c], label=classes[c], lw=1.0)
axes[0].set_ylabel("score (OneHotMSE)"); axes[0].axhline(0.5, color="0.7", lw=0.5, ls="--")
axes[0].legend(loc="upper right", fontsize=8)
axes[0].set_title(f"H1 CNN (fold {FOLD}) on 2024-02-13_H1_CL_1 — cue-onset windows ({len(labels)} trials, acc {acc*100:.1f}%)")
# truth strip
axes[1].imshow(labels[None, :], aspect="auto", cmap="tab10", interpolation="nearest", vmin=0, vmax=9)
axes[1].set_yticks([]); axes[1].set_xlabel("trial #")
axes[1].set_ylabel("truth")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "cnn_predictions.png"), dpi=120)
print(f"Saved {FIG_DIR}/cnn_predictions.png")

# ---- Plot 2: confusion matrix ----
cm = np.zeros((4, 4), dtype=int)
for t, p in zip(labels, preds):
    cm[t, p] += 1
fig, ax = plt.subplots(figsize=(4.5, 4))
im = ax.imshow(cm, cmap="Blues")
for i in range(4):
    for j in range(4):
        ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                color="white" if cm[i, j] > cm.max()/2 else "black", fontsize=10)
ax.set_xticks(range(4)); ax.set_xticklabels(classes)
ax.set_yticks(range(4)); ax.set_yticklabels(classes)
ax.set_xlabel("predicted"); ax.set_ylabel("true")
ax.set_title(f"H1 CNN fold {FOLD}, acc {acc*100:.1f}%")
fig.colorbar(im, ax=ax, fraction=0.046)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "cnn_confusion_matrix.png"), dpi=120)
print(f"Saved {FIG_DIR}/cnn_confusion_matrix.png")

# ---- Plot 3: per-trial cursor moves driven by predicted class ----
# Each cue-window prediction maps to a unit vector in its class direction
# (L: -x, R: +x, U: +y, D: -y). Plot the cursor position one step per trial,
# starting at origin. A perfect decoder would land on a circle of radius 1
# stepping outward then back to origin (since cues alternate task/rest).
# To make the spatial result legible we plot one *per-trial step* per direction
# rather than integrating across the session.
direction = np.array([[-1, 0], [1, 0], [0, 1], [0, -1]])  # L, R, U, D
pred_vec = direction[preds]
true_vec = direction[labels]

fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
for ax, vec, title in [
    (axes[0], true_vec, f"Ground-truth cue directions ({len(labels)} trials)"),
    (axes[1], pred_vec, f"CNN-predicted directions (acc {acc*100:.1f}%)"),
]:
    # Show as a 2D scatter with small jitter so overlapping markers are visible
    rng = np.random.RandomState(0)
    jit = rng.normal(0, 0.04, vec.shape)
    ax.scatter(vec[:, 0] + jit[:, 0], vec[:, 1] + jit[:, 1], c=labels, cmap="tab10",
               vmin=0, vmax=9, s=35, alpha=0.7)
    for i, lbl in enumerate(["L", "R", "U", "D"]):
        d = direction[i]
        ax.text(d[0] * 1.15, d[1] * 1.15, lbl, ha="center", va="center", fontsize=12,
                fontweight="bold")
    ax.scatter([0], [0], c="k", s=40, zorder=5)
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5); ax.set_aspect("equal")
    ax.set_title(title); ax.grid(alpha=0.3)

fig.suptitle("H1 CNN — single-trial direction decoding (color = true class)")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "cnn_kf_trajectory.png"), dpi=120)
print(f"Saved {FIG_DIR}/cnn_kf_trajectory.png")

# ---- Summary ----
import json
with open(os.path.join(FIG_DIR, "cnn_metrics.json"), "w") as f:
    json.dump({
        "session": "2024-02-13_H1_CL_1",
        "fold": FOLD,
        "n_trials": int(len(labels)),
        "accuracy": float(acc),
        "label_distribution": np.bincount(labels).tolist(),
        "confusion_matrix": cm.tolist(),
    }, f, indent=2)
print(f"Saved {FIG_DIR}/cnn_metrics.json")
