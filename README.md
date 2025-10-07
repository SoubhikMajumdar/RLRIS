# RIS 6G Deep Q-Network (DQN) Implementation

## 📡 Project Overview

This project implements a Deep Q-Network (DQN) for optimizing Reconfigurable Intelligent Surface (RIS) configurations in 6G wireless communication systems. The system uses reinforcement learning to automatically select optimal RIS phase configurations to maximize signal quality and user fairness.

## 🎯 Key Features

- **Discrete RIS Control**: 256 possible RIS configurations (4 groups × 4 phase options)
- **DQN Training**: Offline reinforcement learning with Double DQN architecture
- **Multiple Reward Functions**: Proportional fairness, sum rate, min-SINR, and fairness-based rewards
- **Comprehensive Testing**: Performance comparison with random policy and detailed analysis
- **Reproducible Results**: Fixed random seeds for consistent performance

## 🏗️ System Architecture

### RIS Configuration
- **RIS Elements**: 16 elements divided into 4 groups
- **Base Station**: 4 antennas with MRT precoding
- **Users**: 3 mobile devices
- **Action Space**: 256 discrete RIS phase configurations
- **State Space**: 6D features (channel norms + SINR values)

### Channel Model
- **BS→RIS Channel**: H_BR (16×4 complex matrix)
- **RIS→User Channels**: h_RU (16×1 for each user)
- **Direct BS→User**: h_BU (4×1 for each user)
- **Effective Channel**: g = h_BU + h_RU^H × Φ × H_BR

## 📊 Performance Results

### Current Performance (with Fixed Seeds)
- **DQN vs Random**: +1.5% reward improvement, +5.2% SINR improvement
- **Average SINR**: 4.5 dB (2.82 linear)
- **Peak SINR**: 8.2 dB (8.18 linear)
- **Action Diversity**: 46.1% of action space used (118/256 actions)

### Training Results
- **Loss Convergence**: 339 → 26 (stable learning over 30 epochs)
- **Training Time**: ~2-3 minutes on CPU
- **Model Size**: ~1.2 MB (dqn_model.pth)

## 🚀 Quick Start

### Prerequisites
```bash
pip install pandas numpy torch matplotlib
```

### 1. Generate Dataset
```bash
python data_generation.py
```
- Generates 4,000 samples with fairness reward function
- Creates `out/ris_dataset.csv`

### 2. Train DQN
```bash
python deep_q_network.py
```
- Trains DQN for 30 epochs
- Saves model as `dqn_model.pth`
- Generates `training_loss.png`

### 3. Test Performance
```bash
python test_ris_dqn.py
```
- Compares DQN vs random policy
- Generates performance plots and analysis

## 📁 File Structure

```
RLRIS/
├── data_generation.py          # Dataset generation and RIS environment
├── deep_q_network.py           # DQN training implementation
├── test_ris_dqn.py            # Comprehensive testing suite
├── compare_reward_functions.py # Reward function comparison
├── simple_test.py             # Simple testing script
├── RIS6gDatasetGen.ipynb      # Original Jupyter notebook
├── out/
│   └── ris_dataset.csv        # Generated dataset
├── dqn_model.pth              # Trained DQN model
├── training_loss.png          # Training loss curve
├── performance_comparison.png  # DQN vs Random comparison
├── action_distribution.png    # Action usage analysis
├── episode_analysis.png       # Detailed episode trajectory
└── reward_function_comparison.png # Reward function analysis
```

## 🔧 Configuration

### Environment Parameters
```python
N = 16  # RIS elements
M = 4   # BS antennas  
U = 3   # Users
G = 4   # RIS groups
K = 4   # Phase options per group
```

### Training Parameters
```python
epochs = 30
batch_size = 16
learning_rate = 5e-5
gamma = 0.9
target_update_freq = 15
```

### Reward Functions
- **'fairness'**: Jain's fairness index (current best)
- **'pf'**: Proportional fairness
- **'sum_rate'**: Sum rate maximization
- **'min_sinr'**: Min-SINR maximization
- **'robust'**: Robust reward with penalties

## 📈 Results Analysis

### Training Loss
- **Initial Loss**: 339.77
- **Final Loss**: 25.71
- **Convergence**: Stable after epoch 20

### Action Distribution
- **Most Used Action**: 82 (83 times)
- **Top 5 Actions**: 82, 138, 250, 142, 69
- **Exploration**: 46.1% of action space utilized

### SINR Performance
- **Average**: 4.5 dB (2.82 linear)
- **Range**: 1.1 - 8.2 dB
- **Improvement**: 5.2% better than random policy

## 🛠️ Customization

### Changing System Parameters
Edit `data_generation.py`:
```python
env = SimpleRISEnv(N=32, M=8, U=6, G=8, K=4)  # Larger system
```

### Different Reward Functions
```python
env = SimpleRISEnv(reward_func='min_sinr')  # Min-SINR reward
```

### Training Parameters
Edit `deep_q_network.py`:
```python
trainer = train_dqn(
    epochs=100,           # More training
    batch_size=32,        # Larger batches
    lr=1e-4,             # Higher learning rate
    gamma=0.95           # Higher discount factor
)
```

## 🔬 Research Applications

This implementation can be used for:
- **RIS Optimization**: Finding optimal phase configurations
- **6G Research**: Studying RIS-assisted communication
- **RL Algorithms**: Testing different reward functions
- **Performance Analysis**: Comparing DQN vs other methods

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

## 📄 License

This project is part of EE297A Capstone at UC Berkeley.

## 👥 Authors

- **Soubhik** - RIS 6G DQN Implementation
- **Katherine** - Project Collaboration

## 🔗 Repository

**GitHub**: https://github.com/ksarnaEE/EE297A-Capstone-Soubhik-Katherine.git

**Branch**: Soubhik

---

*Last updated: December 2024*