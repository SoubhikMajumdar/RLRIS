"""
RIS 6G DQN Testing Script
========================

This script tests the trained DQN model on the RIS 6G environment.
It includes performance evaluation, action analysis, and comparison with random policy.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from data_generation import SimpleRISEnv, generate_dataset
from deep_q_network import DQN, DQNTrainer, RISDataset

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

def load_trained_model(model_path='dqn_model.pth', state_dim=6, n_actions=256):
    """Load the trained DQN model"""
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found!")
        return None
    
    # Initialize trainer and load model
    trainer = DQNTrainer(state_dim, n_actions, device='cpu')
    trainer.load_model(model_path)
    print(f"Loaded trained model from {model_path}")
    return trainer

def test_model_performance(trainer, env, n_episodes=5, n_steps=100):
    """Test the trained model on the environment"""
    print(f"\nTesting DQN model for {n_episodes} episodes, {n_steps} steps each...")
    
    episode_rewards = []
    episode_sinrs = []
    action_distribution = {}
    
    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        episode_sinr = []
        
        for step in range(n_steps):
            # Get action from trained model
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                q_values = trainer.q_net(state_tensor)
                action = q_values.argmax().item()
            
            # Take action in environment
            next_state, reward, done, info = env.step(action)
            
            # Record statistics
            episode_reward += reward
            episode_sinr.append(info['mean_sinr'])
            
            # Track action distribution
            action_distribution[action] = action_distribution.get(action, 0) + 1
            
            state = next_state
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        episode_sinrs.append(np.mean(episode_sinr))
        print(f"Episode {episode+1}: Reward = {episode_reward:.2f}, Avg SINR = {np.mean(episode_sinr):.2f}")
    
    return episode_rewards, episode_sinrs, action_distribution

def test_random_policy(env, n_episodes=5, n_steps=100):
    """Test random policy for comparison"""
    print(f"\nTesting random policy for {n_episodes} episodes, {n_steps} steps each...")
    
    episode_rewards = []
    episode_sinrs = []
    
    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        episode_sinr = []
        
        for step in range(n_steps):
            # Random action
            action = np.random.randint(0, env.cb.size())
            
            # Take action in environment
            next_state, reward, done, info = env.step(action)
            
            # Record statistics
            episode_reward += reward
            episode_sinr.append(info['mean_sinr'])
            
            state = next_state
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        episode_sinrs.append(np.mean(episode_sinr))
        print(f"Episode {episode+1}: Reward = {episode_reward:.2f}, Avg SINR = {np.mean(episode_sinr):.2f}")
    
    return episode_rewards, episode_sinrs

def plot_performance_comparison(dqn_rewards, dqn_sinrs, random_rewards, random_sinrs):
    """Plot performance comparison between DQN and random policy"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Reward comparison
    episodes = range(1, len(dqn_rewards) + 1)
    ax1.plot(episodes, dqn_rewards, 'b-o', label='DQN', linewidth=2, markersize=6)
    ax1.plot(episodes, random_rewards, 'r-s', label='Random', linewidth=2, markersize=6)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Episode Rewards Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # SINR comparison
    ax2.plot(episodes, dqn_sinrs, 'b-o', label='DQN', linewidth=2, markersize=6)
    ax2.plot(episodes, random_sinrs, 'r-s', label='Random', linewidth=2, markersize=6)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Average SINR')
    ax2.set_title('Average SINR Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Performance comparison plot saved as 'performance_comparison.png'")

def analyze_action_distribution(action_distribution, n_actions=256):
    """Analyze and visualize action distribution"""
    print(f"\nAction Distribution Analysis:")
    print(f"Total unique actions used: {len(action_distribution)}")
    print(f"Total possible actions: {n_actions}")
    print(f"Action diversity: {len(action_distribution)/n_actions*100:.1f}%")
    
    # Find most/least used actions
    sorted_actions = sorted(action_distribution.items(), key=lambda x: x[1], reverse=True)
    print(f"\nTop 5 most used actions:")
    for action, count in sorted_actions[:5]:
        print(f"  Action {action}: {count} times")
    
    print(f"\nTop 5 least used actions:")
    for action, count in sorted_actions[-5:]:
        print(f"  Action {action}: {count} times")
    
    # Plot action distribution
    actions = list(action_distribution.keys())
    counts = list(action_distribution.values())
    
    plt.figure(figsize=(12, 6))
    plt.bar(actions, counts, alpha=0.7, color='skyblue', edgecolor='navy')
    plt.xlabel('Action')
    plt.ylabel('Frequency')
    plt.title('DQN Action Distribution')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('action_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Action distribution plot saved as 'action_distribution.png'")

def test_single_episode_detailed(trainer, env):
    """Test a single episode with detailed analysis"""
    print(f"\nDetailed Single Episode Analysis:")
    
    state = env.reset()
    states_history = [state.copy()]
    actions_history = []
    rewards_history = []
    sinrs_history = []
    
    for step in range(50):  # Shorter episode for detailed analysis
        # Get Q-values for current state
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q_values = trainer.q_net(state_tensor).squeeze().numpy()
        
        # Select action
        action = q_values.argmax()
        
        # Take action
        next_state, reward, done, info = env.step(action)
        
        # Record everything
        states_history.append(next_state.copy())
        actions_history.append(action)
        rewards_history.append(reward)
        sinrs_history.append(info['mean_sinr'])
        
        print(f"Step {step+1}: Action={action}, Reward={reward:.3f}, SINR={info['mean_sinr']:.3f}, Q-max={q_values.max():.3f}")
        
        state = next_state
        if done:
            break
    
    # Plot episode trajectory
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Rewards over time
    ax1.plot(rewards_history, 'b-', linewidth=2)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Reward')
    ax1.set_title('Rewards Over Time')
    ax1.grid(True, alpha=0.3)
    
    # SINR over time
    ax2.plot(sinrs_history, 'g-', linewidth=2)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('SINR')
    ax2.set_title('SINR Over Time')
    ax2.grid(True, alpha=0.3)
    
    # Actions over time
    ax3.plot(actions_history, 'r-o', linewidth=2, markersize=4)
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Action')
    ax3.set_title('Actions Over Time')
    ax3.grid(True, alpha=0.3)
    
    # State evolution (first 3 dimensions)
    states_array = np.array(states_history)
    ax4.plot(states_array[:, 0], label='State[0]', linewidth=2)
    ax4.plot(states_array[:, 1], label='State[1]', linewidth=2)
    ax4.plot(states_array[:, 2], label='State[2]', linewidth=2)
    ax4.set_xlabel('Step')
    ax4.set_ylabel('State Value')
    ax4.set_title('State Evolution (First 3 Dimensions)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('episode_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Episode analysis plot saved as 'episode_analysis.png'")

def main():
    """Main testing function"""
    print("RIS 6G DQN Testing Suite")
    print("=" * 50)
    
    # Check if model exists
    if not os.path.exists('dqn_model.pth'):
        print("No trained model found! Please run deep_q_network.py first.")
        return
    
    # Load trained model
    trainer = load_trained_model()
    if trainer is None:
        return
    
    # Create test environment (same as training)
    env = SimpleRISEnv(N=16, M=4, U=3, G=4, K=4)
    print(f"Test environment: {env.N} RIS elements, {env.M} BS antennas, {env.U} users")
    print(f"Action space size: {env.cb.size()}")
    
    # Test 1: Performance comparison
    print("\n" + "="*50)
    print("TEST 1: Performance Comparison")
    print("="*50)
    
    dqn_rewards, dqn_sinrs, action_dist = test_model_performance(trainer, env, n_episodes=5, n_steps=100)
    random_rewards, random_sinrs = test_random_policy(env, n_episodes=5, n_steps=100)
    
    # Calculate statistics
    dqn_avg_reward = np.mean(dqn_rewards)
    random_avg_reward = np.mean(random_rewards)
    dqn_avg_sinr = np.mean(dqn_sinrs)
    random_avg_sinr = np.mean(random_sinrs)
    
    print(f"\nPerformance Summary:")
    print(f"DQN Average Reward: {dqn_avg_reward:.2f}")
    print(f"Random Average Reward: {random_avg_reward:.2f}")
    print(f"Improvement: {((dqn_avg_reward - random_avg_reward) / random_avg_reward * 100):.1f}%")
    print(f"\nDQN Average SINR: {dqn_avg_sinr:.2f}")
    print(f"Random Average SINR: {random_avg_sinr:.2f}")
    print(f"SINR Improvement: {((dqn_avg_sinr - random_avg_sinr) / random_avg_sinr * 100):.1f}%")
    
    # Plot comparison
    plot_performance_comparison(dqn_rewards, dqn_sinrs, random_rewards, random_sinrs)
    
    # Test 2: Action distribution analysis
    print("\n" + "="*50)
    print("TEST 2: Action Distribution Analysis")
    print("="*50)
    analyze_action_distribution(action_dist, env.cb.size())
    
    # Test 3: Detailed single episode
    print("\n" + "="*50)
    print("TEST 3: Detailed Episode Analysis")
    print("="*50)
    test_single_episode_detailed(trainer, env)
    
    print("\nAll tests completed successfully!")
    print("Generated files:")
    print("  - performance_comparison.png")
    print("  - action_distribution.png") 
    print("  - episode_analysis.png")

if __name__ == "__main__":
    main()
