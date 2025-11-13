"""
RIS 6G Dataset Validation Suite
================================

This script provides comprehensive validation of the offline RL dataset
to ensure it's suitable for training DQN. It checks:
- Action space coverage
- State distribution
- Reward characteristics
- Transition dynamics
- SINR sanity
- Offline RL suitability

Run this BEFORE training your DQN to catch issues early!
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA
import seaborn as sns

# Set random seeds
np.random.seed(42)

class DatasetValidator:
    """Comprehensive dataset validation"""
    
    def __init__(self, csv_path='out/ris_dataset.csv'):
        """
        Initialize validator
        
        Args:
            csv_path: Path to dataset CSV file
        """
        self.csv_path = csv_path
        self.df = None
        self.states = None
        self.next_states = None
        self.actions = None
        self.rewards = None
        self.issues = []
        
    def load_dataset(self):
        """Load and parse dataset"""
        print(f"Loading dataset from {self.csv_path}...")
        
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Dataset not found at {self.csv_path}")
        
        self.df = pd.read_csv(self.csv_path)
        
        # Parse JSON states
        self.states = np.array([json.loads(s) for s in self.df['state']], dtype=np.float32)
        self.next_states = np.array([json.loads(s) for s in self.df['next_state']], dtype=np.float32)
        self.actions = self.df['action'].values
        self.rewards = self.df['reward'].values
        self.dones = self.df['done'].values
        
        print(f"[OK] Dataset loaded: {len(self.df)} samples")
        print(f"   Episodes: {self.df['episode'].nunique()}")
        print(f"   State dim: {self.states.shape[1]}")
        
    def validate_action_coverage(self):
        """Check if all actions are represented"""
        print("\n" + "="*70)
        print("1. ACTION SPACE COVERAGE")
        print("="*70)
        
        # Determine total number of actions
        n_actions = int(self.actions.max() + 1)
        action_counts = np.bincount(self.actions, minlength=n_actions)
        unique_actions = np.sum(action_counts > 0)
        
        print(f"Total possible actions: {n_actions}")
        print(f"Actions represented:    {unique_actions}")
        print(f"Coverage:               {unique_actions/n_actions*100:.1f}%")
        
        # Check coverage
        if unique_actions < n_actions:
            missing = np.where(action_counts == 0)[0]
            print(f"[WARNING] Missing {len(missing)} actions: {missing.tolist()}")
            self.issues.append(f"Missing {len(missing)} actions")
        else:
            print("[OK] Full action space coverage")
        
        # Check distribution uniformity
        min_count = action_counts[action_counts > 0].min()
        max_count = action_counts[action_counts > 0].max()
        mean_count = action_counts[action_counts > 0].mean()
        
        print(f"\nSamples per action:")
        print(f"  Min:  {min_count}")
        print(f"  Max:  {max_count}")
        print(f"  Mean: {mean_count:.1f}")
        print(f"  Std:  {action_counts[action_counts > 0].std():.1f}")
        
        # Compute Gini coefficient (0=uniform, 1=concentrated)
        sorted_counts = np.sort(action_counts[action_counts > 0])
        n = len(sorted_counts)
        cumsum = np.cumsum(sorted_counts)
        gini = (2 * np.sum((np.arange(n) + 1) * sorted_counts)) / (n * cumsum[-1]) - (n + 1) / n
        print(f"  Gini coefficient: {gini:.3f} (0=uniform, 1=concentrated)")
        
        if gini > 0.5:
            print(f"[WARNING] High action concentration (Gini={gini:.3f})")
            self.issues.append(f"High action concentration (Gini={gini:.3f})")
        
        # Plot action distribution
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Bar chart
        ax1.bar(range(n_actions), action_counts, alpha=0.7, color='steelblue', edgecolor='navy')
        ax1.set_xlabel('Action', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Action Distribution', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Histogram
        ax2.hist(action_counts[action_counts > 0], bins=20, alpha=0.7, 
                color='coral', edgecolor='darkred')
        ax2.set_xlabel('Samples per Action', fontsize=12)
        ax2.set_ylabel('Number of Actions', fontsize=12)
        ax2.set_title('Distribution of Sample Counts', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('checkds_action.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("\n[OK] Saved: checkds_action.png")
        
        return unique_actions, n_actions
    
    def validate_state_distribution(self):
        """Check state space distribution"""
        print("\n" + "="*70)
        print("2. STATE SPACE DISTRIBUTION")
        print("="*70)
        
        n_features = self.states.shape[1]
        
        print(f"State dimensionality: {n_features}")
        print(f"\nFeature statistics:")
        
        for i in range(n_features):
            feat = self.states[:, i]
            print(f"\n  Feature {i}:")
            print(f"    Range:  [{feat.min():.2f}, {feat.max():.2f}]")
            print(f"    Mean:   {feat.mean():.2f}")
            print(f"    Std:    {feat.std():.2f}")
            print(f"    Median: {np.median(feat):.2f}")
            
            # Check for degenerate features
            if feat.std() < 1e-6:
                print(f"    [WARNING] Near-zero variance!")
                self.issues.append(f"Feature {i} has near-zero variance")
            
            # Check for NaN/Inf
            if not np.all(np.isfinite(feat)):
                print(f"    [ERROR] Contains NaN or Inf!")
                self.issues.append(f"Feature {i} contains NaN or Inf")
        
        # Check for invalid values
        if not np.all(np.isfinite(self.states)):
            print(f"\n[ERROR] Dataset contains NaN or Inf values!")
            self.issues.append("Dataset contains NaN or Inf values")
        else:
            print(f"\n[OK] All state values are finite")
        
        # Plot state distributions
        fig, axes = plt.subplots(2, (n_features + 1) // 2, figsize=(14, 8))
        axes = axes.flatten()
        
        for i in range(n_features):
            ax = axes[i]
            ax.hist(self.states[:, i], bins=50, alpha=0.7, 
                   color='skyblue', edgecolor='navy')
            ax.set_xlabel(f'State[{i}]', fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.set_title(f'Feature {i} Distribution', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
        
        # Hide unused subplots
        for i in range(n_features, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig('checkds_state.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("\n[OK] Saved: checkds_state.png")
        
        return n_features
    
    def validate_rewards(self):
        """Check reward distribution and sanity"""
        print("\n" + "="*70)
        print("3. REWARD DISTRIBUTION")
        print("="*70)
        
        print(f"Reward statistics:")
        print(f"  Mean:   {self.rewards.mean():.3f}")
        print(f"  Std:    {self.rewards.std():.3f}")
        print(f"  Min:    {self.rewards.min():.3f}")
        print(f"  Max:    {self.rewards.max():.3f}")
        print(f"  Median: {np.median(self.rewards):.3f}")
        print(f"  Range:  {self.rewards.max() - self.rewards.min():.3f}")
        
        # Check for degenerate rewards
        if self.rewards.std() < 0.01:
            print(f"[WARNING] Very low reward variance (std={self.rewards.std():.4f})")
            self.issues.append(f"Low reward variance (std={self.rewards.std():.4f})")
        else:
            print(f"[OK] Good reward variance")
        
        # Check for NaN/Inf
        if not np.all(np.isfinite(self.rewards)):
            print(f"[ERROR] Rewards contain NaN or Inf!")
            self.issues.append("Rewards contain NaN or Inf")
        
        # Plot reward analysis
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Histogram
        ax1.hist(self.rewards, bins=50, alpha=0.7, color='lightgreen', edgecolor='darkgreen')
        ax1.axvline(self.rewards.mean(), color='r', linestyle='--', linewidth=2, label='Mean')
        ax1.axvline(np.median(self.rewards), color='b', linestyle='--', linewidth=2, label='Median')
        ax1.set_xlabel('Reward', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Reward Distribution', fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Reward over time (first 1000 steps)
        ax2.plot(self.rewards[:1000], alpha=0.6, linewidth=0.8)
        ax2.set_xlabel('Step', fontsize=12)
        ax2.set_ylabel('Reward', fontsize=12)
        ax2.set_title('Reward Trajectory (first 1000 steps)', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 3. Q-Q plot (normality check)
        stats.probplot(self.rewards, dist="norm", plot=ax3)
        ax3.set_title('Q-Q Plot (Normality Check)', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 4. Box plot
        ax4.boxplot(self.rewards, vert=True, patch_artist=True,
                   boxprops=dict(facecolor='lightblue', alpha=0.7))
        ax4.set_ylabel('Reward', fontsize=12)
        ax4.set_title('Reward Box Plot', fontsize=13, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('checkds_reward.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("\n[OK] Saved: checkds_reward.png")
    
    def validate_transitions(self):
        """Check state transition dynamics"""
        print("\n" + "="*70)
        print("4. TRANSITION DYNAMICS")
        print("="*70)
        
        # Compute state changes
        state_deltas = self.next_states - self.states
        
        print(f"State transition statistics:")
        print(f"  Mean absolute change per feature:")
        for i in range(self.states.shape[1]):
            print(f"    Feature {i}: {np.mean(np.abs(state_deltas[:, i])):.4f}")
        
        # Check for teleportation (unrealistic jumps)
        jump_sizes = np.linalg.norm(state_deltas, axis=1)
        max_jump = np.max(jump_sizes)
        mean_jump = np.mean(jump_sizes)
        
        print(f"\n  State jump magnitudes:")
        print(f"    Mean: {mean_jump:.3f}")
        print(f"    Max:  {max_jump:.3f}")
        
        if max_jump > 10:
            print(f"[WARNING] Large state jumps detected (max={max_jump:.2f})")
            self.issues.append(f"Large state jumps (max={max_jump:.2f})")
        
        # Check state correlation (for time-varying channels)
        print(f"\n  State correlation (s_t vs s_t+1):")
        for i in range(self.states.shape[1]):
            corr = np.corrcoef(self.states[:, i], self.next_states[:, i])[0, 1]
            print(f"    Feature {i}: {corr:.3f}")
            
            if corr < 0.5:
                print(f"      [WARNING] Low correlation (expected high for slow channels)")
        
        # Plot transition analysis
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. State change distribution
        ax1.hist(jump_sizes, bins=50, alpha=0.7, color='orange', edgecolor='darkorange')
        ax1.set_xlabel('State Jump Magnitude', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Distribution of State Changes', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 2. State trajectory (Feature 0)
        ax2.plot(self.states[:200, 0], alpha=0.7, label='State[0]')
        ax2.set_xlabel('Step', fontsize=12)
        ax2.set_ylabel('State[0] Value', fontsize=12)
        ax2.set_title('State[0] Trajectory (first 200 steps)', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 3. State correlation matrix
        state_corr = np.corrcoef(self.states.T)
        im = ax3.imshow(state_corr, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
        ax3.set_xlabel('Feature', fontsize=12)
        ax3.set_ylabel('Feature', fontsize=12)
        ax3.set_title('State Feature Correlation Matrix', fontsize=13, fontweight='bold')
        plt.colorbar(im, ax=ax3)
        
        # 4. Scatter: state vs next_state (Feature 0)
        sample_idx = np.random.choice(len(self.states), size=min(1000, len(self.states)), replace=False)
        ax4.scatter(self.states[sample_idx, 0], self.next_states[sample_idx, 0], 
                   alpha=0.3, s=10)
        ax4.plot([self.states[:, 0].min(), self.states[:, 0].max()],
                [self.states[:, 0].min(), self.states[:, 0].max()],
                'r--', linewidth=2, label='y=x')
        ax4.set_xlabel('State[0] at t', fontsize=12)
        ax4.set_ylabel('State[0] at t+1', fontsize=12)
        ax4.set_title('State Transition (Feature 0)', fontsize=13, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('checkds_transition.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("\n[OK] Saved: checkds_transition.png")
    
    def validate_action_reward_structure(self):
        """Check if actions differentiate rewards"""
        print("\n" + "="*70)
        print("5. ACTION-REWARD STRUCTURE")
        print("="*70)
        
        # Group rewards by action
        action_rewards = self.df.groupby('action')['reward'].agg(['mean', 'std', 'count'])
        
        print(f"Action-reward relationship:")
        print(f"  Best action:  {action_rewards['mean'].idxmax()} "
              f"(reward={action_rewards['mean'].max():.3f})")
        print(f"  Worst action: {action_rewards['mean'].idxmin()} "
              f"(reward={action_rewards['mean'].min():.3f})")
        print(f"  Reward range: {action_rewards['mean'].max() - action_rewards['mean'].min():.3f}")
        print(f"  Mean reward std across actions: {action_rewards['mean'].std():.3f}")
        
        # Check if actions matter
        reward_range = action_rewards['mean'].max() - action_rewards['mean'].min()
        if reward_range < 0.1:
            print(f"[WARNING] Actions have very similar rewards (range={reward_range:.3f})")
            self.issues.append(f"Low action differentiation (range={reward_range:.3f})")
        else:
            print(f"[OK] Good action differentiation")
        
        # Plot action-reward relationship
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # 1. Mean reward per action
        actions_sorted = action_rewards.sort_values('mean', ascending=False)
        ax1.bar(range(len(actions_sorted)), actions_sorted['mean'], 
               yerr=actions_sorted['std'], alpha=0.7, capsize=3,
               color='mediumseagreen', edgecolor='darkgreen')
        ax1.set_xlabel('Action (sorted by reward)', fontsize=12)
        ax1.set_ylabel('Mean Reward', fontsize=12)
        ax1.set_title('Average Reward per Action', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 2. Reward distribution by top 10 actions
        top_10_actions = action_rewards.nlargest(10, 'mean').index
        reward_data = [self.df[self.df['action'] == a]['reward'].values for a in top_10_actions]
        ax2.boxplot(reward_data, labels=top_10_actions, patch_artist=True,
                   boxprops=dict(facecolor='lightcoral', alpha=0.7))
        ax2.set_xlabel('Action', fontsize=12)
        ax2.set_ylabel('Reward', fontsize=12)
        ax2.set_title('Reward Distribution for Top 10 Actions', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('checkds_action_reward.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("\n[OK] Saved: checkds_action_reward.png")
    
    def validate_sinr(self):
        """Check SINR values are physically reasonable"""
        print("\n" + "="*70)
        print("6. SINR VALIDATION")
        print("="*70)
        
        if 'mean_sinr' not in self.df.columns:
            print("[WARNING] No SINR data in dataset")
            return
        
        mean_sinr = self.df['mean_sinr'].values
        sinr_db = 10 * np.log10(mean_sinr + 1e-10)
        
        print(f"SINR statistics (linear scale):")
        print(f"  Mean: {mean_sinr.mean():.2f}")
        print(f"  Std:  {mean_sinr.std():.2f}")
        print(f"  Min:  {mean_sinr.min():.2f}")
        print(f"  Max:  {mean_sinr.max():.2f}")
        
        print(f"\nSINR statistics (dB scale):")
        print(f"  Mean: {sinr_db.mean():.2f} dB")
        print(f"  Min:  {sinr_db.min():.2f} dB")
        print(f"  Max:  {sinr_db.max():.2f} dB")
        
        # Check for unrealistic values
        if np.any(sinr_db < -20):
            print(f"[WARNING] Very low SINR detected (<-20 dB)")
            self.issues.append("Very low SINR values (<-20 dB)")
        if np.any(sinr_db > 40):
            print(f"[WARNING] Unrealistically high SINR detected (>40 dB)")
            self.issues.append("Very high SINR values (>40 dB)")
        
        # Check reward-SINR correlation
        corr = np.corrcoef(self.rewards, mean_sinr)[0, 1]
        print(f"\nReward-SINR correlation: {corr:.3f}")
        
        if corr < 0.5:
            print(f"[WARNING] Weak reward-SINR correlation (expected strong)")
            self.issues.append(f"Weak reward-SINR correlation ({corr:.3f})")
        else:
            print(f"[OK] Strong reward-SINR correlation")
        
        # Plot SINR analysis
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # 1. SINR distribution (dB)
        ax1.hist(sinr_db, bins=50, alpha=0.7, color='plum', edgecolor='purple')
        ax1.axvline(sinr_db.mean(), color='r', linestyle='--', linewidth=2, label='Mean')
        ax1.set_xlabel('SINR (dB)', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('SINR Distribution', fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Reward vs SINR scatter
        sample_idx = np.random.choice(len(self.rewards), size=min(2000, len(self.rewards)), replace=False)
        ax2.scatter(mean_sinr[sample_idx], self.rewards[sample_idx], alpha=0.3, s=10)
        ax2.set_xlabel('SINR (linear)', fontsize=12)
        ax2.set_ylabel('Reward', fontsize=12)
        ax2.set_title(f'Reward vs SINR (corr={corr:.3f})', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('checkds_sinr.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("\n[OK] Saved: checkds_sinr.png")
    
    def validate_offline_rl_suitability(self):
        """Check dataset properties for offline RL"""
        print("\n" + "="*70)
        print("7. OFFLINE RL SUITABILITY")
        print("="*70)
        
        # 1. Behavior policy quality (average return)
        episode_returns = []
        for ep in self.df['episode'].unique():
            ep_data = self.df[self.df['episode'] == ep]
            episode_returns.append(ep_data['reward'].sum())
        
        print(f"Behavior policy performance:")
        print(f"  Mean return:   {np.mean(episode_returns):.2f}")
        print(f"  Std return:    {np.std(episode_returns):.2f}")
        print(f"  Min return:    {np.min(episode_returns):.2f}")
        print(f"  Max return:    {np.max(episode_returns):.2f}")
        
        # 2. Dataset diversity (action entropy)
        n_actions = int(self.actions.max() + 1)
        action_probs = np.bincount(self.actions, minlength=n_actions) / len(self.actions)
        action_probs = action_probs[action_probs > 0]
        entropy = -np.sum(action_probs * np.log(action_probs))
        max_entropy = np.log(n_actions)
        
        print(f"\nDataset diversity:")
        print(f"  Action entropy:       {entropy:.3f}")
        print(f"  Max entropy:          {max_entropy:.3f}")
        print(f"  Normalized entropy:   {entropy/max_entropy:.3f}")
        print(f"  (1.0 = perfectly uniform)")
        
        if entropy / max_entropy < 0.7:
            print(f"[WARNING] Low action diversity (entropy ratio={entropy/max_entropy:.3f})")
            self.issues.append(f"Low action diversity ({entropy/max_entropy:.3f})")
        else:
            print(f"[OK] Good action diversity")
        
        # 3. State coverage (PCA-based approximation)
        if len(self.states) > 100:
            pca = PCA(n_components=min(2, self.states.shape[1]))
            states_2d = pca.fit_transform(self.states)
            coverage = np.prod(states_2d.max(axis=0) - states_2d.min(axis=0))
            explained_var = pca.explained_variance_ratio_.sum()
            
            print(f"\nState space coverage (PCA-based):")
            print(f"  2D coverage area:        {coverage:.2f}")
            print(f"  Explained variance (2D): {explained_var:.3f}")
        
        # Plot offline RL metrics
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # 1. Episode returns
        ax1.plot(episode_returns, 'o-', linewidth=2, markersize=6, alpha=0.7)
        ax1.axhline(np.mean(episode_returns), color='r', linestyle='--', 
                   linewidth=2, label='Mean')
        ax1.set_xlabel('Episode', fontsize=12)
        ax1.set_ylabel('Total Return', fontsize=12)
        ax1.set_title('Episode Returns (Behavior Policy)', fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Action entropy over time (rolling window)
        window = 200
        entropies = []
        for i in range(0, len(self.actions) - window, window // 2):
            window_actions = self.actions[i:i+window]
            probs = np.bincount(window_actions, minlength=n_actions) / window
            probs = probs[probs > 0]
            ent = -np.sum(probs * np.log(probs))
            entropies.append(ent)
        
        ax2.plot(entropies, linewidth=2, alpha=0.7)
        ax2.axhline(max_entropy, color='r', linestyle='--', linewidth=2, 
                   label='Max Entropy')
        ax2.set_xlabel('Window Index', fontsize=12)
        ax2.set_ylabel('Action Entropy', fontsize=12)
        ax2.set_title('Action Diversity Over Time', fontsize=13, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('checkds_offline_rl.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("\n[OK] Saved: checkds_offline_rl.png")
    
    def generate_summary_report(self):
        """Generate final summary report"""
        print("\n" + "="*70)
        print("VALIDATION SUMMARY")
        print("="*70)
        
        if len(self.issues) == 0:
            print("[OK] ALL CHECKS PASSED - Dataset is ready for offline RL training!")
        else:
            print(f"[WARNING] Found {len(self.issues)} potential issues:\n")
            for i, issue in enumerate(self.issues, 1):
                print(f"  {i}. {issue}")
            print("\n[WARNING] Review issues before training. Some may be acceptable.")
        
        print("\n" + "="*70)
        print("Generated validation plots:")
        print("  - checkds_action.png")
        print("  - checkds_state.png")
        print("  - checkds_reward.png")
        print("  - checkds_transition.png")
        print("  - checkds_action_reward.png")
        print("  - checkds_sinr.png")
        print("  - checkds_offline_rl.png")
        print("="*70)
        
        return len(self.issues) == 0
    
    def run_full_validation(self):
        """Run complete validation suite"""
        print("\n" + "="*70)
        print("RIS OFFLINE RL DATASET VALIDATION")
        print("="*70)
        
        # Load dataset
        self.load_dataset()
        
        # Run all validation checks
        self.validate_action_coverage()
        self.validate_state_distribution()
        self.validate_rewards()
        self.validate_transitions()
        self.validate_action_reward_structure()
        self.validate_sinr()
        self.validate_offline_rl_suitability()
        
        # Generate summary
        passed = self.generate_summary_report()
        
        return passed


def main():
    """Main validation function"""
    
    # Check if dataset exists
    dataset_path = 'out/ris_dataset.csv'
    if not os.path.exists(dataset_path):
        print(f"[ERROR] Dataset not found at {dataset_path}")
        print("Please run data_generation.py first to generate the dataset")
        return
    
    # Run validation
    validator = DatasetValidator(dataset_path)
    passed = validator.run_full_validation()
    
    # Exit code based on validation result
    if passed:
        print("\n[OK] Dataset validation PASSED - Ready to train!")
        return 0
    else:
        print("\n[WARNING] Dataset validation completed with warnings - Review issues")
        return 1


if __name__ == "__main__":
    exit(main())
