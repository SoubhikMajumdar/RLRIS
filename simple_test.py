"""
Simple RIS 6G DQN Test
=======================
"""

import os
import numpy as np
import torch
from data_generation import SimpleRISEnv
from deep_q_network import DQN, DQNTrainer

def test_dqn():
    """Simple DQN test without Unicode characters"""
    
    print("RIS 6G DQN Test")
    print("=" * 40)
    
    # Check if model exists
    if not os.path.exists('dqn_model.pth'):
        print("No trained model found!")
        return
    
    # Load model
    trainer = DQNTrainer(state_dim=6, n_actions=256, device='cpu')
    trainer.load_model('dqn_model.pth')
    print("Loaded trained model")
    
    # Create test environment
    env = SimpleRISEnv(N=16, M=4, U=3, G=4, K=4, reward_func='fairness')
    print(f"Environment: {env.N} RIS elements, {env.M} BS antennas, {env.U} users")
    print(f"Action space: {env.cb.size()} actions")
    
    # Test DQN performance
    print("\nTesting DQN performance...")
    dqn_rewards = []
    dqn_sinrs = []
    
    for episode in range(5):
        state = env.reset()
        episode_reward = 0
        episode_sinr = []
        
        for step in range(50):
            # Get action from DQN
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                q_values = trainer.q_net(state_tensor)
                action = q_values.argmax().item()
            
            # Take action
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            episode_sinr.append(info['mean_sinr'])
            state = next_state
            
            if done:
                break
        
        dqn_rewards.append(episode_reward)
        dqn_sinrs.append(np.mean(episode_sinr))
        print(f"Episode {episode+1}: Reward = {episode_reward:.2f}, Avg SINR = {np.mean(episode_sinr):.2f}")
    
    # Test random policy
    print("\nTesting random policy...")
    random_rewards = []
    random_sinrs = []
    
    for episode in range(5):
        state = env.reset()
        episode_reward = 0
        episode_sinr = []
        
        for step in range(50):
            # Random action
            action = np.random.randint(0, env.cb.size())
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            episode_sinr.append(info['mean_sinr'])
            state = next_state
            
            if done:
                break
        
        random_rewards.append(episode_reward)
        random_sinrs.append(np.mean(episode_sinr))
        print(f"Episode {episode+1}: Reward = {episode_reward:.2f}, Avg SINR = {np.mean(episode_sinr):.2f}")
    
    # Results
    print("\n" + "=" * 40)
    print("RESULTS:")
    print(f"DQN Average Reward: {np.mean(dqn_rewards):.2f}")
    print(f"Random Average Reward: {np.mean(random_rewards):.2f}")
    print(f"DQN Average SINR: {np.mean(dqn_sinrs):.2f}")
    print(f"Random Average SINR: {np.mean(random_sinrs):.2f}")
    
    improvement_reward = ((np.mean(dqn_rewards) - np.mean(random_rewards)) / np.mean(random_rewards)) * 100
    improvement_sinr = ((np.mean(dqn_sinrs) - np.mean(random_sinrs)) / np.mean(random_sinrs)) * 100
    
    print(f"Reward Improvement: {improvement_reward:.1f}%")
    print(f"SINR Improvement: {improvement_sinr:.1f}%")
    
    if improvement_reward > 0:
        print("DQN outperforms random policy!")
    else:
        print("DQN underperforms random policy")

if __name__ == "__main__":
    test_dqn()
