# RIS 6G Deep Q-Network (DQN) Implementation

## 📡 Project Overview

This project implements a Deep Q-Network (DQN) for optimizing Reconfigurable Intelligent Surface (RIS) configurations in 6G wireless communication systems. The system uses reinforcement learning to automatically select optimal RIS phase configurations to maximize signal quality and user fairness.

## 🎯 Key Features

- **Discrete RIS Control**: 64 possible RIS configurations (3 groups × 4 phase options = 4³ = 64 actions)
- **Multi-User System**: 2 users with fairness-aware optimization
- **DQN Training**: Offline reinforcement learning with improved Double DQN architecture
- **Physics-Based Heuristic**: Epsilon-greedy policy with best-of-N action selection
- **Time-Varying Channels**: Correlated fading model (AR(1) process with 85% correlation)
- **Multi-Objective Reward**: Balances sum rate (70%) and Jain's fairness index (30%)
- **State & Reward Normalization**: Z-score normalization for stable training
- **Comprehensive Testing**: Performance comparison with random policy and detailed analysis
- **Dataset Validation**: Q-Q plots, action coverage, state distribution analysis
- **Reproducible Results**: Fixed random seeds and deterministic operations for consistent performance

## 🏗️ System Architecture

### RIS Configuration
- **RIS Elements**: 12 elements divided into 3 groups
- **Base Station**: 4 antennas with MRT precoding
- **Users**: 2 mobile devices (multi-user MIMO system)
- **Action Space**: 64 discrete RIS phase configurations (3 groups × 4 phase options = 4³)
- **State Space**: 7D features per timestep:
  - Effective channel strength per user (normalized by noise) × 2
  - Current SINR per user (log scale) × 2
  - Direct channel strength per user (normalized by noise) × 2
  - BS→RIS channel quality (normalized by noise) × 1

### Channel Model
- **BS→RIS Channel**: H_BR (12×4 complex matrix)
- **RIS→User Channels**: h_RU (12×1 per user, 2 users total)
- **Direct BS→User**: h_BU (4×1 per user, 2 users total)
- **Effective Channel**: g_u = h_BU[u] + h_RU[u]^H × Φ × H_BR (per user)
- **Time-Varying**: Correlated fading with AR(1) process (correlation = 0.85, variation = 0.15)
- **Transmit Power**: 30 dBm (1 W)
- **Noise Floor**: -100 dBm

## 📊 Performance Results

### Current Performance (Latest Version - 2-User System)
- **DQN vs Random**: +2.6% reward improvement over random policy
- **Action Diversity**: 90.6% of action space utilized (58/64 actions)
- **Q-Value Discrimination**: Clear separation between good and bad actions
- **Fairness**: Multi-objective reward balances sum rate and user fairness
- **SINR Performance**: Optimized for both users simultaneously

### Training Results
- **Loss Function**: Huber Loss (SmoothL1Loss) for robust training
- **Architecture**: BatchNorm layers, Xavier initialization, reduced dropout
- **Learning Rate**: 1e-4 with StepLR scheduling (step_size=25, gamma=0.8)
- **Target Network Updates**: Every 5 epochs for stable Q-learning
- **Early Stopping**: Patience=15 epochs
- **Training Time**: ~2-3 minutes on CPU
- **Model Size**: ~1.2 MB (dqn_model.pth)

## 🚀 Quick Start

### Prerequisites
```bash
pip install pandas numpy torch matplotlib
```

Run all commands from the **project root**.

### 1. Generate Dataset
```bash
python data_generation/data_generation.py
```
- Generates 20,000 samples with multi-objective reward function
- Uses epsilon-greedy physics-based heuristic policy (ε=0.3)
- Writes `data_generation/output/ris_dataset.csv` (2-user system)

### 2. Validate Dataset (Optional)
```bash
python data_generation/analysis/dataset_check.py
```
- Validates dataset quality for offline RL
- Saves validation plots in `data_generation/analysis/`: `checkds_*.png`
- Checks action coverage, state distribution, reward characteristics

### 2b. Compare Reward Functions (Optional)
```bash
python data_generation/analysis/compare_reward_functions.py
```
- Compares reward functions (pf, sum_rate, fairness, multi_obj, etc.)
- Saves `data_generation/analysis/reward_function_comparison.png`

### 3. Train DQN
```bash
python dqn/deep_q_network.py
```
- Trains improved DQN with BatchNorm, Huber loss, Xavier init
- Applies state and reward normalization (Z-score)
- Saves model as `dqn/train/dqn_model.pth` with normalization statistics
- Saves `dqn/train/training_loss.png`

### 4. Test Performance
```bash
python dqn/test/test_ris_dqn.py
```
- Compares DQN vs random policy (200 steps per episode)
- Saves performance plots in `dqn/test/`
- Evaluates action distribution and Q-value spread

## 📁 File Structure

See `repo_structure.txt` for the full layout. Summary:

```
RLRIS/
├── data_generation/
│   ├── data_generation.py       # Dataset generation and RIS environment
│   ├── output/
│   │   └── ris_dataset.csv      # Generated dataset (20,000 samples)
│   └── analysis/
│       ├── dataset_check.py     # Dataset validation
│       ├── compare_reward_functions.py  # Reward function comparison
│       ├── reward_function_comparison.png
│       └── checkds_*.png        # Validation plots (7 plots)
├── dqn/
│   ├── deep_q_network.py       # DQN model and training
│   ├── train/
│   │   ├── dqn_model.pth
│   │   └── training_loss.png
│   └── test/
│       ├── test_ris_dqn.py
│       ├── simple_test.py
│       └── *.png                # Performance plots
├── README.md
├── requirements.txt
├── repo_structure.txt
└── RIS6gDatasetGen.ipynb
```

## 🔧 Configuration

### Environment Parameters
```python
N = 12   # RIS elements
M = 4    # BS antennas  
U = 2    # Users (multi-user system)
G = 3    # RIS groups
K = 4    # Phase options per group
# Action space: K^G = 4^3 = 64 actions
```

### Training Parameters
```python
epochs = 30
batch_size = 16
learning_rate = 1e-4          # Increased from 5e-5
gamma = 0.9                    # Discount factor
target_update_freq = 5         # More frequent updates
patience = 15                  # Early stopping patience
weight_decay = 1e-5            # L2 regularization
loss_fn = SmoothL1Loss()      # Huber loss (robust to outliers)
```

### Reward Functions
- **'multi_obj'**: Multi-objective reward (α=0.7 sum rate + β=0.3 fairness × sum rate) **[CURRENT]**
- **'fairness'**: Jain's fairness index reward
- **'pf'**: Proportional fairness
- **'sum_rate'**: Sum rate maximization
- **'min_sinr'**: Min-SINR maximization
- **'robust'**: Robust reward with penalties

### Multi-Objective Reward Function
The current reward function balances total throughput and user fairness:
```python
reward = 0.7 × sum_rate + 0.3 × fairness × sum_rate
```
Where:
- **sum_rate** = Σ log₂(1 + SINRᵤ) for all users
- **fairness** = (Σ rates)² / (U × Σ rates²) - Jain's Fairness Index (0-1)
  - 1.0 = Perfect fairness (all users get equal rates)
  - 0.0 = Completely unfair (one user gets everything)

## 📈 Results Analysis

### Training Loss
- **Loss Function**: Huber Loss (SmoothL1Loss) - robust to outliers
- **Convergence**: Stable training with early stopping
- **Normalization**: State and reward normalization for stable gradients

### Action Distribution
- **Action Space Coverage**: 90.6% (58/64 actions used)
- **Exploration**: Good diversity with physics-based heuristic
- **Q-Value Spread**: Clear discrimination between good and bad actions

### Multi-User Performance
- **2-User System**: Optimized for both users simultaneously
- **Fairness**: Balanced resource allocation via multi-objective reward
- **SINR**: Optimized for both users' signal quality
- **Reward Improvement**: +2.6% over random policy

### Dataset Quality
- **Sample Size**: 20,000 transitions
- **Action Coverage**: Excellent (90.6% of action space)
- **State Distribution**: Well-distributed across feature space
- **Reward Distribution**: Appropriate for offline RL (Q-Q plots validate normality)
- **Offline RL Suitability**: High-quality dataset for DQN training

## 🛠️ Customization

### Changing System Parameters
Edit `data_generation/data_generation.py`:
```python
env = SimpleRISEnv(N=16, M=4, U=3, G=4, K=4)  # Multi-user system
```

### Different Reward Functions
```python
env = SimpleRISEnv(reward_func='min_sinr')  # Min-SINR reward
```

### Training Parameters
Edit `dqn/deep_q_network.py`:
```python
trainer = train_dqn(
    epochs=100,           # More training
    batch_size=32,        # Larger batches
    lr=1e-4,             # Higher learning rate
    gamma=0.95           # Higher discount factor
)
```

## 🆕 Recent Improvements

### System Upgrades
- **Multi-User System**: Upgraded from 1 user to 2 users (U=2)
- **Expanded Action Space**: 64 actions (3 groups × 4 phase options)
- **7D State Space**: Enhanced feature representation for 2-user system
- **Multi-Objective Reward**: Balances sum rate (70%) and fairness (30%)

### DQN Architecture Improvements
- **Batch Normalization**: Added BatchNorm1d after each linear layer
- **Xavier Initialization**: Proper weight initialization for stable training
- **Reduced Dropout**: 0.05/0.05/0.0 (was 0.15/0.15/0.1) for better learning
- **Huber Loss**: SmoothL1Loss instead of MSE for robustness to outliers
- **Increased Learning Rate**: 1e-4 (was 5e-5) for faster convergence
- **More Frequent Target Updates**: Every 5 epochs (was 10)
- **Weight Decay**: L2 regularization (1e-5) for generalization
- **State & Reward Normalization**: Z-score normalization for stable gradients

### Data Generation Improvements
- **Physics-Based Heuristic**: Evaluates actions on current channels (not future predictions)
- **Epsilon-Greedy Policy**: ε=0.3 for balanced exploration/exploitation
- **Best-of-N Strategy**: Selects best action from N candidates
- **Time-Varying Channels**: AR(1) process with 85% correlation (realistic fading)

### Validation & Analysis
- **Dataset Validation**: Comprehensive quality checks (`dataset_check.py`)
- **Q-Q Plots**: Validates reward distribution for offline RL suitability
- **Action Coverage Analysis**: Ensures good exploration of action space
- **State Distribution Analysis**: Validates feature representation quality
- **Reproducibility**: Fixed random seeds and deterministic operations

## 🔬 Research Applications

This implementation can be used for:
- **RIS Optimization**: Finding optimal phase configurations for multi-user systems
- **6G Research**: Studying RIS-assisted communication with fairness constraints
- **Offline RL**: Dataset quality validation and offline reinforcement learning
- **Multi-Objective Optimization**: Balancing throughput and fairness in wireless systems
- **Performance Analysis**: Comparing DQN vs random/heuristic policies
- **Fairness Metrics**: Evaluating Jain's fairness index in resource allocation

## 📊 Fairness Explanation

**Jain's Fairness Index** measures how evenly data rates are distributed across users:
- **Formula**: `fairness = (Σ rates)² / (U × Σ rates²)`
- **Range**: 0 (unfair) to 1 (perfectly fair)
- **1.0**: All users get equal rates (perfect fairness)
- **0.5-0.7**: Some users get much more (unfair)
- **0.0**: One user gets everything (completely unfair)

In this system, fairness ensures both users receive reasonable performance, preventing one user from being starved while optimizing total throughput.

## 📚 References

- **RIS Technology**: Reconfigurable Intelligent Surfaces for 6G
- **DQN Algorithm**: Deep Q-Network for discrete action spaces
- **Wireless Communication**: MIMO systems with RIS assistance
- **Reinforcement Learning**: Offline RL for wireless optimization

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 👥 Authors

- **Soubhik** - RIS 6G DQN Implementation
- **Katherine** - Project Collaboration

## 🔗 Repositories

**Capstone Repository**: https://github.com/ksarnaEE/EE297A-Capstone-Soubhik-Katherine.git  
**Branch**: Soubhik

**Personal Repository**: https://github.com/SoubhikMajumdar/RLRIS.git  
**Branch**: main

---

*Last updated: February 2026*