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
        for _ in range(self.G):
            digits.append(x % self.K)
            x //= self.K
        digits = digits[::-1]
        phases = self.phase_options[digits]
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
        num = abs(g @ W[u])**2
        denom = noise + sum(abs(g @ W[v])**2 for v in range(U) if v != u)
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
    """Simple RIS environment for dataset generation"""
    def __init__(self, N=64, M=8, U=6, G=8, K=8, reward_func='pf'):
        """
        Initialize RIS environment
        
        Args:
            N: Number of RIS elements
            M: Number of BS antennas
            U: Number of users
            G: Number of RIS groups
            K: Number of phase options per group
            reward_func: Reward function to use ('pf', 'sum_rate', 'min_sinr', 'fairness', 'robust', 'multi_obj')
        """
        self.N, self.M, self.U = N, M, U
        self.cb = RISCodebook(N, G, K)
        self.noise = db2lin(-100/10)   # ~ -100 dBm noise
        self.Ptx = db2lin(30 - 30)     # 1 W transmit power
        
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
        """Extract state features from channels and SINR"""
        feats = []
        # Effective channel strength per user
        feats += [np.linalg.norm(g)**2 for g in GU]
        # Current SINR per user
        feats += list(sinr)
        return np.array(feats, dtype=np.float32)

    def reset(self):
        """Reset environment to initial state"""
        self.H_BR = complex_randn((self.N, self.M))
        self.h_RU = [complex_randn((self.N,)) for _ in range(self.U)]
        self.h_BU = [complex_randn((self.M,)) for _ in range(self.U)]
        # Start with identity RIS (no phase control)
        Phi = np.eye(self.N)
        GU = build_effective(self.H_BR, self.h_RU, self.h_BU, Phi)
        W = mrt_precoder(GU, self.Ptx)
        sinr = compute_sinr(GU, W, self.noise)
        self.state = self.extract_state(GU, sinr)
        return self.state

    def step(self, a):
        """Take action and return next state, reward, done, info"""
        Phi = self.cb.action_to_diag(a)
        GU = build_effective(self.H_BR, self.h_RU, self.h_BU, Phi)
        W = mrt_precoder(GU, self.Ptx)
        sinr = compute_sinr(GU, W, self.noise)
        r = self.reward_func(sinr)
        s_next = self.extract_state(GU, sinr)
        return s_next, r, False, {"mean_sinr": np.mean(sinr), "min_sinr": np.min(sinr), "max_sinr": np.max(sinr)}

# ---- Dataset Generation ----
def generate_dataset(episodes=20, steps=200, outdir="out"):
    """
    Generate offline RL dataset
    
    Args:
        episodes: Number of episodes to generate
        steps: Number of steps per episode
        outdir: Output directory for dataset
        
    Returns:
        pandas.DataFrame: Generated dataset
    """
    # Use smaller action space to avoid memory issues
    # Use 'fairness' reward function for better performance
    env = SimpleRISEnv(N=16, M=4, U=3, G=4, K=4, reward_func='fairness')  # Best performing reward
    rows = []
    
    for ep in range(episodes):
        s = env.reset()
        for t in range(steps):
            a = np.random.randint(0, env.cb.size())  # random policy
            s_next, r, d, info = env.step(a)
            # mark last step in the episode as done
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
    # Example usage
    print("Generating RIS 6G dataset...")
    df = generate_dataset(episodes=20, steps=200, outdir="out")
    print("Dataset shape:", df.shape)
    print("First few rows:")
    print(df.head())
