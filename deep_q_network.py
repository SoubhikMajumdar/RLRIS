"""
Deep Q Network for RIS 6G Control (WITH STATE + REWARD NORMALIZATION)
====================================================================

Implements a Deep Q Network (DQN) for offline reinforcement learning
on the RIS 6G dataset with state and reward normalization.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# ---- Dataset Loader ----
class RISDataset(Dataset):
    """PyTorch Dataset for RIS data with state & reward normalization"""
    def __init__(self, csv_path, normalize=True):
        """
        Initialize dataset from CSV file
        
        Args:
            csv_path: Path to the CSV dataset file
            normalize: Whether to normalize states and rewards
        """
        self.data = pd.read_csv(csv_path)
        self.states = np.vstack(self.data["state"].apply(json.loads).values)
        self.next_states = np.vstack(self.data["next_state"].apply(json.loads).values)
        self.actions = self.data["action"].values
        self.rewards = self.data["reward"].values.astype(np.float32)
        self.dones = self.data["done"].values.astype(np.float32)
        
        self.state_dim = self.states.shape[1]
        self.n_actions = int(self.data["action"].max() + 1)
        
        self.normalize = normalize

        # ============= STATE NORMALIZATION =============
        if self.normalize:
            self.state_mean = self.states.mean(axis=0).astype(np.float32)
            self.state_std = self.states.std(axis=0).astype(np.float32) + 1e-8
            
            print("\n" + "="*60)
            print("STATE NORMALIZATION STATISTICS")
            print("="*60)
            print(f"Original state ranges:")
            for i in range(self.state_dim):
                print(f"  Feature {i}: [{self.states[:, i].min():.2f}, {self.states[:, i].max():.2f}]")
            
            # Normalize states
            self.states = (self.states - self.state_mean) / self.state_std
            self.next_states = (self.next_states - self.state_mean) / self.state_std
            
            print(f"\nNormalization parameters:")
            print(f"  Mean: {self.state_mean}")
            print(f"  Std:  {self.state_std}")
            
            print(f"\nNormalized state statistics:")
            print(f"  Mean: {self.states.mean(axis=0)}")
            print(f"  Std:  {self.states.std(axis=0)}")
            print("="*60 + "\n")
        else:
            self.state_mean = None
            self.state_std = None
            print("\n[WARNING] State normalization disabled")
        # ===============================================

        # ============= REWARD NORMALIZATION =============
        if self.normalize:
            self.reward_mean = self.rewards.mean()
            self.reward_std = self.rewards.std() + 1e-8
            original_rewards = self.rewards.copy()
            self.rewards = (self.rewards - self.reward_mean) / self.reward_std

            print("\nReward normalization:")
            print(f"  Original: mean={self.reward_mean:.2f}, std={self.reward_std:.2f}")
            print(f"  Range: [{original_rewards.min():.2f}, {original_rewards.max():.2f}]")
            print(f"  Normalized: mean={self.rewards.mean():.4f}, std={self.rewards.std():.4f}")
            print(f"  Range: [{self.rewards.min():.2f}, {self.rewards.max():.2f}]")
        else:
            self.reward_mean = None
            self.reward_std = None
        # ====================================================

        print(f"Dataset loaded: {len(self.states)} samples")
        print(f"State dimension: {self.state_dim}")
        print(f"Number of actions: {self.n_actions}")

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return {
            'state': torch.tensor(self.states[idx], dtype=torch.float32),
            'next_state': torch.tensor(self.next_states[idx], dtype=torch.float32),
            'action': torch.tensor(self.actions[idx], dtype=torch.long),
            'reward': torch.tensor(self.rewards[idx], dtype=torch.float32),
            'done': torch.tensor(self.dones[idx], dtype=torch.float32)
        }
    
    def normalize_state(self, state):
        """Normalize a single state using stored statistics"""
        if self.normalize and self.state_mean is not None:
            return (state - self.state_mean) / self.state_std
        return state


# ---- DQN Network ----
class DQN(nn.Module):
    """Deep Q Network for RIS control"""
    def __init__(self, state_dim, n_actions, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.15),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.15),
            
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        return self.net(x)


# ---- Training Functions ----
class DQNTrainer:
    """DQN Trainer class with normalization support"""
    def __init__(self, state_dim, n_actions, lr=5e-5, gamma=0.95, device='cpu'):
        self.device = torch.device(device)
        self.gamma = gamma
        
        # Initialize networks
        self.q_net = DQN(state_dim, n_actions).to(self.device)
        self.target_net = DQN(state_dim, n_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        
        # Initialize optimizer and loss
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.8)
        self.loss_fn = nn.MSELoss()
        
        # Normalization stats
        self.state_mean = None
        self.state_std = None
        self.reward_mean = None
        self.reward_std = None
        self.normalize_states = False

    def train_epoch(self, dataloader):
        """Train for one epoch"""
        self.q_net.train()
        losses = []
        
        for batch in dataloader:
            states = batch['state'].to(self.device)
            next_states = batch['next_state'].to(self.device)
            actions = batch['action'].to(self.device)
            rewards = batch['reward'].to(self.device)
            dones = batch['done'].to(self.device)
            
            qvals = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            
            with torch.no_grad():
                next_actions = self.q_net(next_states).argmax(1)
                next_qvals = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                targets = rewards + self.gamma * next_qvals * (1 - dones)
            
            loss = self.loss_fn(qvals, targets)
            
            self.optimizer.zero_grad()
            loss.backward()
            
            # ---- FIX 4: Gradient Clipping Check ----
            grad_norm = torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
            if grad_norm > 10:
                print(f"  Warning: Large gradient norm: {grad_norm:.2f}")
            # ---------------------------------------
            
            self.optimizer.step()
            losses.append(loss.item())
        
        self.scheduler.step()
        return np.mean(losses)

    def update_target_network(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def evaluate(self, states):
        self.q_net.eval()
        with torch.no_grad():
            states_tensor = torch.tensor(states, dtype=torch.float32).to(self.device)
            qvals = self.q_net(states_tensor)
        return qvals.cpu().numpy()

    def select_action(self, state, epsilon=0.0):
        if self.normalize_states and self.state_mean is not None:
            state = (state - self.state_mean) / self.state_std
        
        if np.random.random() < epsilon:
            qvals = self.evaluate(state.reshape(1, -1))
            n_actions = qvals.shape[1]
            return np.random.randint(n_actions)
        else:
            qvals = self.evaluate(state.reshape(1, -1))
            return int(np.argmax(qvals[0]))

    def save_model(self, path):
        """Save model to file including normalization statistics"""
        torch.save({
            'q_net_state_dict': self.q_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'state_mean': self.state_mean,
            'state_std': self.state_std,
            'reward_mean': self.reward_mean,  # NEW
            'reward_std': self.reward_std,    # NEW
            'normalize_states': self.normalize_states
        }, path)
        print(f"[OK] Model saved with normalization stats")

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(checkpoint['q_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if 'state_mean' in checkpoint:
            self.state_mean = checkpoint['state_mean']
            self.state_std = checkpoint['state_std']
            self.reward_mean = checkpoint.get('reward_mean', None)
            self.reward_std = checkpoint.get('reward_std', None)
            self.normalize_states = checkpoint['normalize_states']
            print("[OK] Loaded normalization statistics")
        else:
            print("[WARNING] No normalization stats found in checkpoint")


# ---- Main Training Function ----
def train_dqn(dataset_path, epochs=100, batch_size=64, lr=5e-5, gamma=0.95, 
              target_update_freq=10, device='cpu', save_path='dqn_model.pth'):
    dataset = RISDataset(dataset_path, normalize=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    trainer = DQNTrainer(
        state_dim=dataset.state_dim,
        n_actions=dataset.n_actions,
        lr=lr,
        gamma=gamma,
        device=device
    )
    
    trainer.state_mean = dataset.state_mean
    trainer.state_std = dataset.state_std
    trainer.reward_mean = dataset.reward_mean
    trainer.reward_std = dataset.reward_std
    trainer.normalize_states = dataset.normalize
    print(f"[OK] Normalization statistics transferred to trainer")
    
    print("\nStarting DQN training...")
    loss_history = []
    best_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(epochs):
        loss = trainer.train_epoch(dataloader)
        loss_history.append(loss)
        
        if loss < best_loss:
            best_loss = loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        if epoch % target_update_freq == 0:
            trainer.update_target_network()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(loss_history) + 1), loss_history, 'b-', linewidth=2, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('DQN Training Loss Over Epochs')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('training_loss.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    trainer.save_model(save_path)
    print(f"Model saved to {save_path}")
    print(f"Loss plot saved to training_loss.png")
    
    return trainer


# ---- Evaluation Functions ----
def evaluate_model(trainer, dataset, n_samples=5):
    sample_states = dataset.states[:n_samples]
    qvals = trainer.evaluate(sample_states)
    
    print(f"\nQ-values for first {n_samples} states:")
    print(qvals)
    print(f"Chosen actions: {qvals.argmax(1)}")
    
    return qvals


if __name__ == "__main__":
    dataset_path = "out/ris_dataset.csv"
    
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}")
        print("Please run data_generation.py first to generate the dataset")
    else:
        trainer = train_dqn(
            dataset_path=dataset_path,
            epochs=100,
            batch_size=64,
            lr=5e-5,        # Reduced learning rate
            gamma=0.95,     # Adjusted discount factor
            target_update_freq=10,
            device='cpu'
        )
        
        dataset = RISDataset(dataset_path)
        evaluate_model(trainer, dataset, n_samples=5)
