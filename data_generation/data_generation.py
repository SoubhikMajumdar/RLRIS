"""
RIS 6G Dataset Generation
========================

This module generates an offline RL dataset for discrete RIS control (6G).
It simulates BS–RIS–UE channels, builds a discrete RIS codebook, runs a 
behavior policy (ε-greedy over a simple heuristic), computes rates/SINR 
with MRT precoding, and logs clean tuples (s,a,r,s') for DQN/CQL/etc.
"""

import os
import json
import numpy as np
import pandas as pd

# Set random seeds for reproducibility
np.random.seed(42)

# ---- Helpers ----
def db2lin(x_db): 
    """Convert dB to linear scale"""
    return 10.0**(x_db/10.0)

def lin2db(x): 
    """Convert linear scale to dB"""
    return 10.0*np.log10(max(x,1e-30))

def complex_randn(shape): 
    """Generate complex random numbers"""
    return (np.random.randn(*shape) + 1j*np.random.randn(*shape))/np.sqrt(2)

def path_loss(distance, d0=1.0, n=2.0, PL0_dB=30, fc_GHz=3.0):
    """
    Free-space path loss model with configurable exponent
    
    PL(d) = PL(d0) + 10*n*log10(d/d0)
    where PL0 = 20*log10(4*pi*d0*fc/c) at reference distance d0
    
    Args:
        distance: Distance in meters
        d0: Reference distance (1m)
        n: Path loss exponent (2.0 = free space, 2.5-4.0 = urban/NLOS)
        PL0_dB: Path loss at reference distance (dB) - typically 30-40 dB at 1m
        fc_GHz: Carrier frequency in GHz (default 3.0 GHz for sub-6GHz)
    
    Returns:
        Path loss coefficient (linear scale, < 1.0 for attenuation)
    """
    if distance < d0:
        distance = d0
    # Standard path loss formula
    PL_dB = PL0_dB + 10 * n * np.log10(distance / d0)
    return db2lin(-PL_dB)  # Convert to linear scale (attenuation factor)

# ---- RIS Codebook ----
class RISCodebook:
    """Discrete RIS codebook for phase control"""
    def __init__(self, N=64, G=8, K=8):
        """
        Initialize RIS codebook
        
        Args:
            N: Number of RIS elements
            G: Number of groups
            K: Number of phase options per group
        """
        assert N % G == 0
        self.N, self.G, self.K = N, G, K
        self.group_size = N // G
        self.phase_options = np.linspace(0, 2*np.pi, num=K, endpoint=False)

    def size(self):
        """Return total number of possible actions"""
        return self.K**self.G

    def action_to_diag(self, a: int):
        """Convert integer action to diagonal phase matrix"""
        # Convert integer a to group-phase assignment
        digits = []
        x = a
        #The RIS is partitioned into G groups, requiring G phase decisions.
        for _ in range(self.G):
            digits.append(x % self.K)
            x //= self.K
        #Since modulo extracts digits from least significant to most significant (right-to-left), the list is reversed to map the indices correctly to Group 1, Group 2, ..., Group G.    
        digits = digits[::-1]
        #Discrete Phase Shift: This implements phase quantization, where the RIS elements are limited to K distinct phase shifts
        phases = self.phase_options[digits]
        #Grouped RIS Architecture: This implements grouped phase control. This is a practical simplification where $N/G$ adjacent elements share a single control signal, reducing the complexity of the control hardware and the size of the action space from K^N to K^G
        phase_vec = np.repeat(phases, self.group_size)
        return np.diag(np.exp(1j*phase_vec))

# ---- Channel + SINR ----
def build_effective(H_BR, h_RU, h_BU, Phi):
    """Build effective channels through RIS"""
    GU = []
    for u in range(len(h_RU)):
        g = h_BU[u].conj().T + h_RU[u].conj().T @ Phi @ H_BR
        GU.append(g.reshape(1,-1))
    return GU

def mrt_precoder(GU, Ptx):
    """Maximum Ratio Transmission precoder"""
    U = len(GU)
    if U == 0: 
        return []
    W = []
    for g in GU:
        v = g.conj().T
        w = v / (np.linalg.norm(v) + 1e-12) * np.sqrt(Ptx / U)
        W.append(w)
    return W

def compute_sinr(GU, W, noise):
    """Compute SINR for each user"""
    U = len(GU)
    out = np.zeros(U)
    for u in range(U):
        g = GU[u]
        num = abs((g @ W[u]).item())**2  # scalar
        denom = noise + sum(abs((g @ W[v]).item())**2 for v in range(U) if v != u)
        out[u] = float(num / denom)
    return out

# ---- Reward Functions ----
def pf_reward(sinr):
    """Proportional fairness reward (current)"""
    return float(np.sum(np.log2(1+sinr)))

def sum_rate_reward(sinr):
    """Sum rate reward - maximizes total throughput"""
    return float(np.sum(np.log2(1+sinr)))

def min_sinr_reward(sinr):
    """Min-SINR reward - maximizes worst user performance"""
    return float(np.min(sinr))

def weighted_sum_reward(sinr, weights=None):
    """Weighted sum reward - customizable user priorities"""
    if weights is None:
        weights = np.ones(len(sinr))
    return float(np.sum(weights * np.log2(1+sinr)))

def fairness_reward(sinr):
    """Jain's fairness index reward - balances user fairness"""
    if len(sinr) <= 1:
        return float(np.sum(np.log2(1+sinr)))
    
    rates = np.log2(1+sinr)
    sum_rates = np.sum(rates)
    sum_squared_rates = np.sum(rates**2)
    
    if sum_squared_rates == 0:
        return 0.0
    
    fairness = (sum_rates**2) / (len(sinr) * sum_squared_rates)
    return float(sum_rates * fairness)  # Combine rate and fairness

def energy_efficiency_reward(sinr, power_consumption=1.0):
    """Energy efficiency reward - rate per unit power"""
    total_rate = np.sum(np.log2(1+sinr))
    return float(total_rate / power_consumption)

def robust_reward(sinr, target_sinr=2.0):
    """Robust reward - penalizes users below target SINR"""
    rates = np.log2(1+sinr)
    penalty = np.sum(np.maximum(0, target_sinr - sinr))
    return float(np.sum(rates) - 0.1 * penalty)

def multi_objective_reward(sinr, alpha=0.7, beta=0.3):
    """Multi-objective reward - balances sum rate and fairness"""
    sum_rate = np.sum(np.log2(1+sinr))
    
    if len(sinr) > 1:
        rates = np.log2(1+sinr)
        fairness = (np.sum(rates)**2) / (len(sinr) * np.sum(rates**2))
    else:
        fairness = 1.0
    
    return float(alpha * sum_rate + beta * fairness * sum_rate)

# ---- Environment ----
class SimpleRISEnv:
    """Simple RIS environment for dataset generation with time-varying channels"""
    def __init__(self, N=64, M=8, U=6, G=8, K=8, reward_func='pf', channel_variation=0.1,
                 enable_path_loss=True, d_BR=30.0, d_RU=None, d_BU=None):
        """
        Initialize RIS environment
        
        Args:
            N: Number of RIS elements
            M: Number of BS antennas
            U: Number of users
            G: Number of RIS groups
            K: Number of phase options per group
            reward_func: Reward function to use ('pf', 'sum_rate', 'min_sinr', 'fairness', 'robust', 'multi_obj')
            channel_variation: Channel variation rate (0=static, 0.3=high mobility)
            enable_path_loss: Enable distance-based path loss (default: True)
            d_BR: Distance from BS to RIS in meters (default: 30m)
            d_RU: List of distances from RIS to each user in meters (default: [10, 15, ...] for U users)
            d_BU: List of distances from BS to each user (direct path) in meters (default: [50, 60, ...] for U users)
        """
        self.N, self.M, self.U = N, M, U
        self.cb = RISCodebook(N, G, K)
        self.noise = db2lin(-100)   # ~ -100 dBm noise floor
        # 30 dBm = 10^(30/10) = 1000 mW = 1 W transmit power
        self.Ptx = db2lin(30)     # 30 dBm = 1 W
        self.channel_variation = channel_variation  # How much channels vary
        
        # Path loss configuration
        self.enable_path_loss = enable_path_loss
        if enable_path_loss:
            self.d_BR = d_BR
            # Default distances: RIS-User (LOS, shorter), BS-User (NLOS, longer, often blocked)
            if d_RU is None:
                self.d_RU = [10.0 + 5.0*u for u in range(U)]  # 10m, 15m, 20m, ...
            else:
                self.d_RU = d_RU if isinstance(d_RU, list) else [d_RU] * U
            if d_BU is None:
                self.d_BU = [50.0 + 10.0*u for u in range(U)]  # 50m, 60m, 70m, ...
            else:
                self.d_BU = d_BU if isinstance(d_BU, list) else [d_BU] * U
            
            # Path loss exponents: LOS for BS-RIS and RIS-User, NLOS for direct BS-User
            self.n_BR = 2.0   # Free space (LOS)
            self.n_RU = 2.0   # Free space (LOS)
            self.n_BU = 3.5   # Urban NLOS (obstacles, higher attenuation)
            self.PL0_dB = 30  # Path loss at 1m reference distance
            
            # Compute path loss coefficients (will be applied to channels)
            self.beta_BR = path_loss(self.d_BR, n=self.n_BR, PL0_dB=self.PL0_dB)
            self.beta_RU = [path_loss(d, n=self.n_RU, PL0_dB=self.PL0_dB) for d in self.d_RU]
            self.beta_BU = [path_loss(d, n=self.n_BU, PL0_dB=self.PL0_dB) for d in self.d_BU]
        else:
            # No path loss: all channels have unit power
            self.beta_BR = 1.0
            self.beta_RU = [1.0] * U
            self.beta_BU = [1.0] * U
        
        # Set reward function
        self.reward_func_name = reward_func
        if reward_func == 'pf':
            self.reward_func = pf_reward
        elif reward_func == 'sum_rate':
            self.reward_func = sum_rate_reward
        elif reward_func == 'min_sinr':
            self.reward_func = min_sinr_reward
        elif reward_func == 'fairness':
            self.reward_func = fairness_reward
        elif reward_func == 'robust':
            self.reward_func = robust_reward
        elif reward_func == 'multi_obj':
            self.reward_func = multi_objective_reward
        else:
            self.reward_func = pf_reward  # Default

    def extract_state(self, GU, sinr):
        """
        Extract and normalize state features from channels and SINR
        
        Features are normalized by noise power for scale consistency.
        Further standardization (mean=0, std=1) is applied during DQN training
        in RISDataset to handle different feature scales (channel powers vs log-SINR).
        """
        feats = []
        
        # 1. Effective channel strength per user (normalized by noise power)
        feats += [np.linalg.norm(g)**2 / (self.noise + 1e-10) for g in GU]
        
        # 2. Current SINR per user (log scale for better numerical range)
        feats += [np.log10(s + 1e-10) for s in sinr]
        
        # 3. Direct channel strength per user (normalized by noise power)
        feats += [np.linalg.norm(self.h_BU[u])**2 / (self.noise + 1e-10) for u in range(self.U)]
        
        # 4. BS→RIS channel quality (normalized by noise power per element)
        feats += [np.linalg.norm(self.H_BR)**2 / (self.H_BR.size * self.noise + 1e-10)]
        
        return np.array(feats, dtype=np.float32)

    def reset(self):
        """Reset environment to initial state"""
        # Generate unit-variance channels
        H_BR_unit = complex_randn((self.N, self.M))
        h_RU_unit = [complex_randn((self.N,)) for _ in range(self.U)]
        h_BU_unit = [complex_randn((self.M,)) for _ in range(self.U)]
        
        # Apply path loss (if enabled)
        self.H_BR = np.sqrt(self.beta_BR) * H_BR_unit
        self.h_RU = [np.sqrt(self.beta_RU[u]) * h_RU_unit[u] for u in range(self.U)]
        self.h_BU = [np.sqrt(self.beta_BU[u]) * h_BU_unit[u] for u in range(self.U)]
        
        # Start with identity RIS (no phase control)
        Phi = np.eye(self.N)
        GU = build_effective(self.H_BR, self.h_RU, self.h_BU, Phi)
        W = mrt_precoder(GU, self.Ptx)
        sinr = compute_sinr(GU, W, self.noise)
        self.state = self.extract_state(GU, sinr)
        return self.state

    def step(self, a):
        """Take action and return next state, reward, done, info"""
        # Apply time-varying channel model (fading/mobility)
        if self.channel_variation > 0:
            # Correlated fading: old_channel * (1-alpha) + new_channel * alpha
            # Path loss is preserved: we evolve the unit-variance component, then reapply path loss
            alpha = self.channel_variation
            
            # Extract unit-variance components (divide by sqrt of path loss)
            H_BR_unit = self.H_BR / (np.sqrt(self.beta_BR) + 1e-10)
            h_RU_unit = [self.h_RU[u] / (np.sqrt(self.beta_RU[u]) + 1e-10) for u in range(self.U)]
            h_BU_unit = [self.h_BU[u] / (np.sqrt(self.beta_BU[u]) + 1e-10) for u in range(self.U)]
            
            # Evolve unit-variance channels
            H_BR_unit = np.sqrt(1 - alpha**2) * H_BR_unit + alpha * complex_randn((self.N, self.M))
            for u in range(self.U):
                h_RU_unit[u] = np.sqrt(1 - alpha**2) * h_RU_unit[u] + alpha * complex_randn((self.N,))
                h_BU_unit[u] = np.sqrt(1 - alpha**2) * h_BU_unit[u] + alpha * complex_randn((self.M,))
            
            # Reapply path loss
            self.H_BR = np.sqrt(self.beta_BR) * H_BR_unit
            self.h_RU = [np.sqrt(self.beta_RU[u]) * h_RU_unit[u] for u in range(self.U)]
            self.h_BU = [np.sqrt(self.beta_BU[u]) * h_BU_unit[u] for u in range(self.U)]
        
        # Compute performance with current action
        Phi = self.cb.action_to_diag(a)
        GU = build_effective(self.H_BR, self.h_RU, self.h_BU, Phi)
        W = mrt_precoder(GU, self.Ptx)
        sinr = compute_sinr(GU, W, self.noise)
        r = self.reward_func(sinr)
        s_next = self.extract_state(GU, sinr)
        return s_next, r, False, {"mean_sinr": np.mean(sinr), "min_sinr": np.min(sinr), "max_sinr": np.max(sinr)}

# ---- Dataset Generation ----
def heuristic_policy(env, state, epsilon=0.2):
    """
    Epsilon-greedy policy with physics-based heuristic
    
    Evaluates candidate actions on CURRENT channels using a best-of-N strategy.
    Uses current channels (not future predictions) to avoid noise from single-sample
    channel variation. This approach is stable and appropriate for slowly-varying 
    channels (correlation ~85%).
    
    Args:
        env: RIS environment
        state: Current state (not directly used, but kept for API consistency)
        epsilon: Exploration probability (default: 0.2)
        
    Returns:
        action: Selected action (int in [0, K^G))
    """
    if np.random.random() < epsilon:
        # Exploration: uniform random action
        return np.random.randint(0, env.cb.size())
    
    # Exploitation: best-of-N heuristic on CURRENT channels (not future predictions)
    best_action = 0
    best_reward = -float('inf')
    
    # Evaluate a random subset of actions
    n_candidates = min(20, env.cb.size())  # Try up to 20 actions
    candidate_actions = np.random.choice(env.cb.size(), size=n_candidates, replace=False)
    
    # Test each candidate action on CURRENT channels
    for a in candidate_actions:
        # Apply RIS phase configuration
        Phi = env.cb.action_to_diag(a)
        
        # Compute effective channels
        GU = build_effective(env.H_BR, env.h_RU, env.h_BU, Phi)
        
        # Design MRT precoder
        W = mrt_precoder(GU, env.Ptx)
        
        # Compute SINR
        sinr = compute_sinr(GU, W, env.noise)
        
        # Evaluate reward
        reward = env.reward_func(sinr)
        
        # Track best action
        if reward > best_reward:
            best_reward = reward
            best_action = a
    
    return best_action

def _default_output_dir():
    """Default CSV output: data_generation/output/"""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def generate_dataset(episodes=50, steps=200, outdir=None, diverse=True):
    """
    Generate offline RL dataset

    Args:
        episodes: Number of episodes to generate
        steps: Number of steps per episode  
        outdir: Output directory for dataset (default: data_generation/output/)
        diverse: Use diverse dataset generation strategy
        
    Returns:
        pandas.DataFrame: Generated dataset
    """
    if outdir is None:
        outdir = _default_output_dir()
    if diverse:
        return generate_diverse_dataset(episodes, steps, outdir)
    else:
        return generate_simple_dataset(episodes, steps, outdir)

def generate_diverse_dataset(episodes=100, steps=200, outdir=None):
    """
    Generate highly diverse offline RL dataset with uniform action coverage
    
    Uses multi-objective reward (alpha=0.7, beta=0.3) to balance sum rate and fairness
    """
    if outdir is None:
        outdir = _default_output_dir()
    env = SimpleRISEnv(N=12, M=4, U=2, G=3, K=4, reward_func='multi_obj', channel_variation=0.15)
    rows = []
    n_actions = env.cb.size()
    
    for ep in range(episodes):
        s = env.reset()
        for t in range(steps):
            rand_val = np.random.random()
            
            if rand_val < 0.3:
                # 50%: Pure uniform random
                a = np.random.randint(0, n_actions)
                
            elif rand_val < 0.8:
                # 30%: Physics-based heuristic
                a = heuristic_policy(env, s, epsilon=0.0)  # Pure exploitation
                
            else:
                # 20%: Sequential exploration
                a = (ep * steps + t) % n_actions
            
            s_next, r, d, info = env.step(a)
            done_flag = (t == steps - 1)
            
            rows.append({
                "episode": ep, "t": t,
                "state": json.dumps(s.tolist()),
                "action": a, "reward": r,
                "next_state": json.dumps(s_next.tolist()),
                "done": done_flag,
                "mean_sinr": info["mean_sinr"]
            })
            s = s_next
    
    df = pd.DataFrame(rows)
    os.makedirs(outdir, exist_ok=True)
    df.to_csv(os.path.join(outdir, "ris_dataset.csv"), index=False)
    print(f"Saved dataset: {df.shape}")
    return df

def generate_simple_dataset(episodes=25, steps=200, outdir=None):
    """
    Legacy simple dataset generation with heuristic
    
    Uses multi-objective reward (alpha=0.7, beta=0.3) to balance sum rate and fairness
    """
    if outdir is None:
        outdir = _default_output_dir()
    env = SimpleRISEnv(N=12, M=4, U=2, G=3, K=4, reward_func='multi_obj', channel_variation=0.15)
    rows = []
    
    for ep in range(episodes):
        s = env.reset()
        for t in range(steps):
            a = heuristic_policy(env, s, epsilon=0.2)
            s_next, r, d, info = env.step(a)
            done_flag = (t == steps - 1)
            rows.append({
                "episode": ep, "t": t,
                "state": json.dumps(s.tolist()),
                "action": a, "reward": r,
                "next_state": json.dumps(s_next.tolist()),
                "done": done_flag,
                "mean_sinr": info["mean_sinr"]
            })
            s = s_next
    
    df = pd.DataFrame(rows)
    os.makedirs(outdir, exist_ok=True)
    df.to_csv(os.path.join(outdir, "ris_dataset.csv"), index=False)
    print("Saved dataset:", df.shape)
    return df

if __name__ == "__main__":
    # Writes dataset to data_generation/output/ris_dataset.csv (default)
    print("Generating diverse RIS 6G dataset...")
    print("Configuration: 2 users, 4 BS antennas, 12 RIS elements, 64 actions (G=3, K=4), 20,000 samples")
    print("Time-varying channels enabled (variation=0.15)")
    print("Path loss enabled: BS-RIS (30m, n=2.0), RIS-User (10-15m, n=2.0), BS-User (50-60m, n=3.5)")
    df = generate_dataset(episodes=100, steps=200, diverse=True)
    print("Dataset shape:", df.shape)
    print("First few rows:")
    print(df.head())
    
    # Show action distribution in dataset
    actions = df['action'].values
    unique_actions = len(np.unique(actions))
    total_actions = 64  # 4^3
    print(f"\nAction diversity in dataset: {unique_actions}/{total_actions} actions ({unique_actions/total_actions*100:.1f}%)")
    print(f"Samples per action (avg): {len(actions)/unique_actions:.0f}")
