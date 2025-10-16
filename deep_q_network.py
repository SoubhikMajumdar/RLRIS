"""
Deep Q Network for RIS 6G Control
================================

This module implements a Deep Q Network (DQN) for offline reinforcement learning
on the RIS 6G dataset. It includes dataset loading, network architecture, 
training loop, and evaluation functions.
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
    """PyTorch Dataset for RIS data"""
    def __init__(self, csv_path):
        """
        Initialize dataset from CSV file
        
        Args:
            csv_path: Path to the CSV dataset file
        """
        self.data = pd.read_csv(csv_path)
        self.states = np.vstack(self.data["state"].apply(json.loads).values)
        self.next_states = np.vstack(self.data["next_state"].apply(json.loads).values)
        self.actions = self.data["action"].values
        self.rewards = self.data["reward"].values.astype(np.float32)
        self.dones = self.data["done"].values.astype(np.float32)
        
        self.state_dim = self.states.shape[1]
        self.n_actions = int(self.data["action"].max() + 1)
        
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

# ---- DQN Network ----
class DQN(nn.Module):
    """Deep Q Network for RIS control"""
    def __init__(self, state_dim, n_actions, hidden_dim=256):
        """
        Initialize DQN network with improved architecture for richer state
        
        Args:
            state_dim: Dimension of state space (now 10D with balanced features)
            n_actions: Number of possible actions
            hidden_dim: Hidden layer dimension
        """
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
        """Forward pass through the network"""
        return self.net(x)

# ---- Training Functions ----
class DQNTrainer:
    """DQN Trainer class"""
    def __init__(self, state_dim, n_actions, lr=1e-3, gamma=0.99, device='cpu'):
        """
        Initialize DQN trainer
        
        Args:
            state_dim: State dimension
            n_actions: Number of actions
            lr: Learning rate
            gamma: Discount factor
            device: Device to run on ('cpu' or 'cuda')
        """
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

    def train_epoch(self, dataloader):
        """Train for one epoch"""
        self.q_net.train()
        losses = []
        
        for batch in dataloader:
            # Move batch to device
            states = batch['state'].to(self.device)
            next_states = batch['next_state'].to(self.device)
            actions = batch['action'].to(self.device)
            rewards = batch['reward'].to(self.device)
            dones = batch['done'].to(self.device)
            
            # Current Q-values
            qvals = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            
            # Target Q-values with Double DQN to reduce overestimation
            with torch.no_grad():
                # Use current network to select actions, target network to evaluate
                next_actions = self.q_net(next_states).argmax(1)
                next_qvals = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                targets = rewards + self.gamma * next_qvals * (1 - dones)
            
            # Compute loss and update
            loss = self.loss_fn(qvals, targets)
            
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            losses.append(loss.item())
        
        # Update learning rate
        self.scheduler.step()
        
        return np.mean(losses)

    def update_target_network(self):
        """Update target network with current Q-network weights"""
        self.target_net.load_state_dict(self.q_net.state_dict())

    def evaluate(self, states):
        """Evaluate Q-values for given states"""
        self.q_net.eval()
        with torch.no_grad():
            states_tensor = torch.tensor(states, dtype=torch.float32).to(self.device)
            qvals = self.q_net(states_tensor)
        return qvals.cpu().numpy()

    def save_model(self, path):
        """Save model to file"""
        torch.save({
            'q_net_state_dict': self.q_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }, path)

    def load_model(self, path):
        """Load model from file"""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(checkpoint['q_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

# ---- Main Training Function ----
def train_dqn(dataset_path, epochs=100, batch_size=32, lr=1e-4, gamma=0.95, 
              target_update_freq=10, device='cpu', save_path='dqn_model.pth'):
    """
    Train DQN on RIS dataset
    
    Args:
        dataset_path: Path to dataset CSV file
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        gamma: Discount factor
        target_update_freq: Frequency of target network updates
        device: Device to use ('cpu' or 'cuda')
        save_path: Path to save trained model
    """
    # Load dataset
    dataset = RISDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize trainer
    trainer = DQNTrainer(
        state_dim=dataset.state_dim,
        n_actions=dataset.n_actions,
        lr=lr,
        gamma=gamma,
        device=device
    )
    
    # Training loop with early stopping
    print("Starting DQN training...")
    loss_history = []
    best_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(epochs):
        loss = trainer.train_epoch(dataloader)
        loss_history.append(loss)
        
        # Early stopping
        if loss < best_loss:
            best_loss = loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        # Update target network
        if epoch % target_update_freq == 0:
            trainer.update_target_network()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")
    
    # Plot loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(loss_history) + 1), loss_history, 'b-', linewidth=2, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('DQN Training Loss Over Epochs')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('training_loss.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close plot to avoid display issues
    
    # Save model
    trainer.save_model(save_path)
    print(f"Model saved to {save_path}")
    print(f"Loss plot saved to training_loss.png")
    
    return trainer

# ---- Evaluation Functions ----
def evaluate_model(trainer, dataset, n_samples=5):
    """
    Evaluate trained model on sample states
    
    Args:
        trainer: Trained DQN trainer
        dataset: RIS dataset
        n_samples: Number of sample states to evaluate
    """
    sample_states = dataset.states[:n_samples]
    qvals = trainer.evaluate(sample_states)
    
    print(f"Q-values for first {n_samples} states:")
    print(qvals)
    print(f"Chosen actions: {qvals.argmax(1)}")
    
    return qvals

if __name__ == "__main__":
    # Example usage
    dataset_path = "out/ris_dataset.csv"
    
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}")
        print("Please run data_generation.py first to generate the dataset")
    else:
        # Train DQN with optimized parameters for single-user system
        trainer = train_dqn(
            dataset_path=dataset_path,
            epochs=80,
            batch_size=64,
            lr=2e-4,
            gamma=0.98,
            target_update_freq=5,
            device='cpu'
        )
        
        # Evaluate model
        dataset = RISDataset(dataset_path)
        evaluate_model(trainer, dataset, n_samples=5)
