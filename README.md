# Reinforcement Learning for Reconfigurable Intelligent Surface (RIS) Optimization

## Project Overview

This project applies reinforcement learning to optimize Reconfigurable Intelligent Surface (RIS) phase configurations for 6G mmWave wireless communication systems. Two RL approaches are implemented and compared:

| Approach | Algorithm | RIS Scale | Channel Model | Training |
|----------|-----------|-----------|---------------|----------|
| **PPO** (primary) | Proximal Policy Optimization | 256 elements (16x16), 4 groups, 16 phases | Geometric free-space path loss at 28 GHz | Online, rollout-based |
| **DQN** (baseline) | Deep Q-Network | 12 elements, 3 groups, 4 phases | Correlated Rayleigh fading (AR(1)) | Offline, dataset-driven |

## PPO Approach

The PPO agent sequentially selects discrete phase configurations for grouped RIS elements to maximise channel gain between a transmitter and receiver. It trains on randomised TX/RX geometries and generalises across the coverage area.

### System Model

- **Carrier frequency**: 28 GHz (mmWave)
- **RIS array**: 16x16 = 256 elements, half-wavelength spacing, positioned at (25, 2, 0) m
- **Grouping**: 4 column groups, each with 16 discrete phase levels (0° to 337.5° in 22.5° steps)
- **Episode**: 12 steps (3 passes over 4 groups), group order randomised per episode
- **Channel**: Geometric free-space path loss with per-element delay computation; no small-scale fading
- **Measurement noise**: Multiplicative Gaussian noise on gain observations (~SNR 20 dB)

### Observation Space (14-dimensional)

The agent observes only what a real RIS controller would have access to:

| Feature | Dimension | Description |
|---------|-----------|-------------|
| Phase encoding | 2G = 8 | sin/cos of current group phases |
| Normalised gain | 1 | `(gain - gain_rand) / (gain_opt - gain_rand)`, clipped |
| Step fraction | 1 | `t / T` (temporal position) |
| Group one-hot | G = 4 | Which group is being updated |

The agent does **not** receive a pre-computed per-group gain table. It learns to map from its current configuration and observed channel quality to the best phase assignment.

### Reward

Absolute normalised gain (linear scale) at each step:

```
r = clip( (gain_new - gain_rand) / (gain_opt - gain_rand), 0, 1 )
```

### Training Configuration

```python
UPDATES       = 3000
ROLLOUT_STEPS = 2048          # ~6.1M total env steps
LR            = 1e-4
gamma         = 0.99
gae_lambda    = 0.95
clip_eps      = 0.2
entropy_coef  = 0.05
n_epochs_ppo  = 4
batch_size    = 128
```

Training geometry: TX at x in [1, 5] m, RX at x in [38, 62] m and z in [-3, 3] m — randomised every episode.

### PPO Results

- Converges to ~93% of continuous optimal gain within 500 updates on CPU
- On fixed-geometry inference, reaches **>96% of continuous optimal** and **>97% of discrete ceiling** (brute-force K^G = 65,536 configs)
- Spatial performance heatmaps show consistent generalisation across the full RX sweep region

### Running the PPO Notebook

Open `ppo/train_ris_ppo_colab_final.ipynb` in Google Colab (T4 GPU recommended). Cells in order:

1. **Mount Drive** — saves/loads model checkpoints to Google Drive
2. **Environment & model definitions** — constants, channel model, `RISEnv`, `ActorCritic`, PPO update
3. **Channel visualisation** — optimal phase/delay heatmaps, gain-vs-active-elements plot, phasor diagram
4. **Training loop** — 3000 updates with periodic evaluation logging
5. **Save model & plots** — exports to `/content/ppo_train/` and Drive
6. **Fixed-geometry inference** — step-by-step episode with PPO, random, and brute-force discrete ceiling comparison
7. **Spatial performance heatmap** — sweeps RX over a 2D grid, produces PPO-vs-ceiling heatmaps
8. **Absolute gain heatmaps** — dB-scale gain maps for PPO, ceiling, and gap

There is also a standalone training script at `ppo/train_ris_ppo.py`:

```bash
python ppo/train_ris_ppo.py --updates 500 --obs-mode full --noise-level 0.1
```

## DQN Approach

The DQN approach uses offline reinforcement learning on a pre-generated dataset of channel realisations with a multi-user MIMO system.

### System Model

- **RIS elements**: 12, divided into 3 groups with 4 phase options each (action space: 4^3 = 64)
- **Base station**: 4 antennas with MRT precoding
- **Users**: 2 mobile devices
- **Channel**: Time-varying correlated Rayleigh fading (AR(1), correlation = 0.85)
- **Reward**: Multi-objective — 70% sum rate + 30% Jain's fairness index

### Running the DQN Pipeline

```bash
# 1. Generate dataset (20,000 samples)
python data_generation/data_generation.py

# 2. Validate dataset (optional)
python data_generation/analysis/dataset_check.py

# 3. Train DQN
python dqn/deep_q_network.py

# 4. Test performance
python dqn/test/test_ris_dqn.py
```

### DQN Results

- +2.6% reward improvement over random policy
- 90.6% action space coverage (58/64 actions)
- Huber loss with BatchNorm, Xavier init, and Z-score state normalisation

## File Structure

```
RLRIS/
├── ppo/
│   ├── train_ris_ppo_colab_final.ipynb   # PPO Colab notebook (primary)
│   └── train_ris_ppo.py                  # Standalone PPO training script
├── data_generation/
│   ├── data_generation.py                # Dataset generation for DQN
│   └── analysis/
│       ├── dataset_check.py              # Dataset quality validation
│       └── compare_reward_functions.py   # Reward function comparison
├── dqn/
│   ├── deep_q_network.py                # DQN model and training
│   └── test/
│       ├── test_ris_dqn.py              # Performance evaluation
│       └── simple_test.py               # Quick sanity check
├── README.md
├── requirements.txt
└── repo_structure.txt
```

## Prerequisites

```bash
pip install numpy torch matplotlib plotly tqdm
```

For the Colab notebook, all dependencies are pre-installed. GPU (T4) is recommended but not required.

## Authors

- **Soubhik Majumdar** — PPO and DQN implementation
- **Katherine Sarna** — Project collaboration

## Repositories

**Capstone**: https://github.com/ksarnaEE/EE297A-Capstone-Soubhik-Katherine  
**Personal**: https://github.com/SoubhikMajumdar/RLRIS

---

*Last updated: May 2026*
