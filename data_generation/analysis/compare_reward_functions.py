"""
Reward Function Comparison for RIS 6G DQN
=========================================

This script compares different reward functions to help choose the best one
for your RIS 6G system.
"""

import sys
from pathlib import Path
# Project root (script is in data_generation/analysis/) so "from data_generation.data_generation import" works
_proj = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_proj))

import os
import numpy as np
import matplotlib.pyplot as plt
from data_generation.data_generation import (
    SimpleRISEnv, pf_reward, sum_rate_reward, min_sinr_reward,
    fairness_reward, robust_reward, multi_objective_reward
)

def test_reward_functions(n_episodes=10, n_steps=50):
    """Test different reward functions and compare their performance"""
    
    reward_functions = {
        'Proportional Fairness': 'pf',
        'Sum Rate': 'sum_rate', 
        'Min SINR': 'min_sinr',
        'Fairness': 'fairness',
        'Robust': 'robust',
        'Multi-Objective': 'multi_obj'
    }
    
    results = {}
    
    print("Testing Different Reward Functions...")
    print("=" * 60)
    
    for name, func_key in reward_functions.items():
        print(f"\nTesting {name} reward function...")
        
        # Create environment with specific reward function
        env = SimpleRISEnv(N=16, M=4, U=3, G=4, K=4, reward_func=func_key)
        
        episode_rewards = []
        episode_sinrs = []
        min_sinrs = []
        max_sinrs = []
        
        for episode in range(n_episodes):
            state = env.reset()
            episode_reward = 0
            episode_sinr = []
            episode_min_sinr = []
            episode_max_sinr = []
            
            for step in range(n_steps):
                # Random action for fair comparison
                action = np.random.randint(0, env.cb.size())
                next_state, reward, done, info = env.step(action)
                
                episode_reward += reward
                episode_sinr.append(info['mean_sinr'])
                episode_min_sinr.append(info['min_sinr'])
                episode_max_sinr.append(info['max_sinr'])
                
                state = next_state
                if done:
                    break
            
            episode_rewards.append(episode_reward)
            episode_sinrs.append(np.mean(episode_sinr))
            min_sinrs.append(np.mean(episode_min_sinr))
            max_sinrs.append(np.mean(episode_max_sinr))
        
        # Store results
        results[name] = {
            'rewards': episode_rewards,
            'mean_sinr': episode_sinrs,
            'min_sinr': min_sinrs,
            'max_sinr': max_sinrs,
            'avg_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'avg_mean_sinr': np.mean(episode_sinrs),
            'avg_min_sinr': np.mean(min_sinrs),
            'avg_max_sinr': np.mean(max_sinrs)
        }
        
        print(f"  Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
        print(f"  Average Mean SINR: {np.mean(episode_sinrs):.2f}")
        print(f"  Average Min SINR: {np.mean(min_sinrs):.2f}")
        print(f"  Average Max SINR: {np.mean(max_sinrs):.2f}")
    
    return results

def plot_reward_comparison(results):
    """Plot comparison of different reward functions"""
    
    # Extract data for plotting
    names = list(results.keys())
    avg_rewards = [results[name]['avg_reward'] for name in names]
    avg_mean_sinrs = [results[name]['avg_mean_sinr'] for name in names]
    avg_min_sinrs = [results[name]['avg_min_sinr'] for name in names]
    avg_max_sinrs = [results[name]['avg_max_sinr'] for name in names]
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Average Rewards
    bars1 = ax1.bar(names, avg_rewards, color='skyblue', alpha=0.7, edgecolor='navy')
    ax1.set_title('Average Episode Rewards', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Average Reward')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars1, avg_rewards):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # Average Mean SINR
    bars2 = ax2.bar(names, avg_mean_sinrs, color='lightgreen', alpha=0.7, edgecolor='darkgreen')
    ax2.set_title('Average Mean SINR', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Mean SINR')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    for bar, value in zip(bars2, avg_mean_sinrs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # Min SINR (worst user performance)
    bars3 = ax3.bar(names, avg_min_sinrs, color='lightcoral', alpha=0.7, edgecolor='darkred')
    ax3.set_title('Average Min SINR (Worst User)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Min SINR')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    for bar, value in zip(bars3, avg_min_sinrs):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # Max SINR (best user performance)
    bars4 = ax4.bar(names, avg_max_sinrs, color='gold', alpha=0.7, edgecolor='orange')
    ax4.set_title('Average Max SINR (Best User)', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Max SINR')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)
    
    for bar, value in zip(bars4, avg_max_sinrs):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    _out_dir = os.path.dirname(os.path.abspath(__file__))  # data_generation/analysis/
    _plot_path = os.path.join(_out_dir, 'reward_function_comparison.png')
    plt.savefig(_plot_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close plot to avoid display issues
    print("[OK] Reward function comparison plot saved to data_generation/analysis/reward_function_comparison.png")

def analyze_reward_characteristics(results):
    """Analyze the characteristics of each reward function"""
    
    print("\n" + "="*60)
    print("REWARD FUNCTION ANALYSIS")
    print("="*60)
    
    # Find best performers
    best_reward = max(results.keys(), key=lambda x: results[x]['avg_reward'])
    best_mean_sinr = max(results.keys(), key=lambda x: results[x]['avg_mean_sinr'])
    best_min_sinr = max(results.keys(), key=lambda x: results[x]['avg_min_sinr'])
    best_max_sinr = max(results.keys(), key=lambda x: results[x]['avg_max_sinr'])
    
    print(f"\nBEST PERFORMERS:")
    print(f"  Highest Average Reward: {best_reward}")
    print(f"  Best Mean SINR: {best_mean_sinr}")
    print(f"  Best Min SINR (Fairness): {best_min_sinr}")
    print(f"  Best Max SINR: {best_max_sinr}")
    
    # Calculate fairness metrics
    print(f"\nFAIRNESS ANALYSIS:")
    for name, data in results.items():
        fairness_ratio = data['avg_min_sinr'] / data['avg_max_sinr']
        print(f"  {name}: Min/Max ratio = {fairness_ratio:.3f} (higher = more fair)")
    
    # Recommendations
    print(f"\nRECOMMENDATIONS:")
    print(f"  • For MAXIMUM THROUGHPUT: Use '{best_reward}' reward function")
    print(f"  • For USER FAIRNESS: Use '{best_min_sinr}' reward function")
    print(f"  • For BALANCED PERFORMANCE: Use 'Multi-Objective' reward function")
    print(f"  • For ROBUSTNESS: Use 'Robust' reward function")

def main():
    """Main comparison function"""
    print("RIS 6G Reward Function Comparison")
    print("=" * 60)
    
    # Test all reward functions
    results = test_reward_functions(n_episodes=10, n_steps=50)
    
    # Plot comparison
    plot_reward_comparison(results)
    
    # Analyze characteristics
    analyze_reward_characteristics(results)
    
    print(f"\n[OK] Reward function comparison completed!")
    print(f"Generated file: data_generation/analysis/reward_function_comparison.png")

if __name__ == "__main__":
    main()
