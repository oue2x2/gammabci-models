"""
Headless rollout of the LSTM copilot in its training env (SJ4DirectionsEnv).
Saves a cursor-trajectory figure to test/figures/.

Run from the bci_raspy/ directory (the env's relative paths assume that CWD).
"""
import os, sys, yaml, json, types
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# pyautogui is imported by modules/SJutil/Gamify.py but not used during headless
# rollouts; pyautogui's macOS build chain is broken on Python 3.8 + arm64. Stub it.
sys.modules.setdefault("pyautogui", types.SimpleNamespace(
    size=lambda: (1920, 1080),
    position=lambda: (0, 0),
    moveTo=lambda *a, **k: None,
    click=lambda *a, **k: None,
    FAILSAFE=False,
))

# Resolve paths relative to this file
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
BCI_RASPY = os.path.abspath(os.path.join(HERE, "bci_raspy"))

# bci_raspy must be CWD because env.py loads SJtools/copilot/targets/*.yaml as relative paths
os.chdir(BCI_RASPY)
sys.path.insert(0, BCI_RASPY)

from sb3_contrib import RecurrentPPO
from SJtools.copilot.env import SJ4DirectionsEnv

MODEL_PATH = os.path.join(REPO, "lstm_copilot", "best_model.zip")
CFG_PATH = os.path.join(REPO, "lstm_copilot", "model.yaml")
FIG_DIR = os.path.join(REPO, "test", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

with open(CFG_PATH) as f:
    cfg = yaml.safe_load(f)

env = SJ4DirectionsEnv(
    render=False, showSoftmax=False, showVelocity=False,
    softmax_type="normal_target",
    reward_type="baseLinDistDecay.yaml",
    holdtime=2.0,
    stillCS=0.0,
    extra_targets_yaml="dir-8-close.yaml",
    obs=cfg["obs_dim"]["obs"],
    action=cfg["action_dim"]["action"],
    action_param=[k for kv in cfg["action_dim"].get("action_param", {}).items() for k in kv],
    historyDim=cfg["obs_dim"]["history"],
    velReplaceSoftmax=cfg["obs_dim"]["velReplaceSoftmax"],
    center_out_back=True,
    copilotYamlParam=cfg,
)

model = RecurrentPPO.load(MODEL_PATH, env=env, device="cpu")
print(f"Loaded {MODEL_PATH}")
print(f"  policy params: {sum(p.numel() for p in model.policy.parameters())}")

N_EPISODES = 16
trajectories = []  # list of dicts: cursor_path, target_pos, hit, length
rng = np.random.RandomState(0)

for ep in range(N_EPISODES):
    obs = env.reset()
    state = None
    episode_starts = np.ones((1,), dtype=bool)
    cursor_path = []
    target_pos = None
    done = False
    reward_sum = 0.0
    t = 0
    while not done and t < 1500:
        action, state = model.predict(
            obs, state=state, episode_start=episode_starts, deterministic=True
        )
        obs, reward, done, info = env.step(action)
        episode_starts = np.array([False])
        cursor_path.append(np.array(info["cursor_pos"], dtype=float).copy())
        target_pos = np.array(info["target_pos"], dtype=float).copy()
        reward_sum += float(reward)
        t += 1
    cursor_path = np.array(cursor_path)
    # "hit" = ended early via successful hold (not timeout); SJ4DirectionsEnv reward shape varies,
    # but a clean trial ends on successful hold within ~600 steps.
    hit = bool(done and t < 1000)
    trajectories.append({
        "cursor_path": cursor_path,
        "target_pos": target_pos,
        "hit": hit,
        "length": t,
        "reward_sum": reward_sum,
    })
    print(f"  ep {ep:2d}: target={target_pos.tolist()} len={t:4d} hit={hit} R={reward_sum:.2f}")

# ---- Plot ----
fig, ax = plt.subplots(figsize=(7, 7))
# Draw the workspace boundary and origin
ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, color="0.7", lw=1))
ax.scatter([0], [0], c="k", s=30, zorder=5, label="origin")
# Draw all unique target positions as faint markers
unique_targets = {tuple(t["target_pos"].tolist()) for t in trajectories}
for tx, ty in unique_targets:
    ax.scatter([tx], [ty], c="0.5", marker="s", s=120, alpha=0.4, zorder=2)

# Plot each trajectory colored by hit/miss
for tr in trajectories:
    p = tr["cursor_path"]
    if p.shape[0] < 2: continue
    color = "#2ca02c" if tr["hit"] else "#d62728"
    ax.plot(p[:, 0], p[:, 1], color=color, lw=1.0, alpha=0.7)
    ax.scatter([p[0, 0]], [p[0, 1]], c="k", s=10, zorder=4)
    ax.scatter([p[-1, 0]], [p[-1, 1]], c=color, s=25, zorder=4, marker="x")

n_hit = sum(t["hit"] for t in trajectories)
ax.set_title(f"LSTM copilot rollouts (best_model, normal_target softmax)\n"
             f"{n_hit}/{N_EPISODES} hits, mean length {np.mean([t['length'] for t in trajectories]):.0f}")
ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2); ax.set_aspect("equal")
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.grid(alpha=0.3)
fig.tight_layout()
out = os.path.join(FIG_DIR, "lstm_cursor_paths.png")
fig.savefig(out, dpi=120)
print(f"\nSaved {out}")
print(f"Hit rate: {n_hit}/{N_EPISODES} = {n_hit/N_EPISODES:.2f}")

# Save metrics
with open(os.path.join(FIG_DIR, "lstm_metrics.json"), "w") as f:
    json.dump({
        "n_episodes": N_EPISODES,
        "n_hits": n_hit,
        "hit_rate": n_hit / N_EPISODES,
        "mean_length": float(np.mean([t["length"] for t in trajectories])),
        "mean_reward": float(np.mean([t["reward_sum"] for t in trajectories])),
    }, f, indent=2)
