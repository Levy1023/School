import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque
import random

class DQNetwork(nn.Module):
    """
    Deep Q-Network neural network architecture
    """
    def __init__(self, state_size, action_size, hidden_size=64):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    """
    Experience replay buffer for storing and sampling transitions
    """
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards), 
                np.array(next_states), np.array(dones))
    
    def size(self):
        return len(self.buffer)

class DQNAgent:
    """
    Deep Q-Network agent for reinforcement learning
    """
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.99
        self.learning_rate = 0.001
        self.batch_size = 64
        self.memory = ReplayBuffer(10000)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy_net = DQNetwork(state_size, action_size).to(self.device)
        self.target_net = DQNetwork(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()
        
    def select_action(self, state, training=True):
        """
        Select action using epsilon-greedy policy
        """
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return q_values.argmax(1).item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """
        Store transition in replay buffer
        """
        self.memory.push(state, action, reward, next_state, done)
    
    def train(self):
        """
        Train the network using experience replay
        """
        if self.memory.size() < self.batch_size:
            return 0
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        loss = self.loss_fn(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """
        Update target network with policy network weights
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def decay_epsilon(self):
        """
        Decay exploration rate
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save(self, filename):
        """
        Save model weights
        """
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filename)
        print(f"Model saved to {filename}")
    
    def load(self, filename):
        """
        Load model weights
        """
        checkpoint = torch.load(filename)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        print(f"Model loaded from {filename}")

def train_agent(episodes=500, render_freq=None):
    """
    Main training loop for the DQN agent
    """
    env = gym.make('CartPole-v1')
    
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    agent = DQNAgent(state_size, action_size)
    
    scores = []
    avg_scores = []
    losses = []
    epsilons = []
    
    target_update_freq = 10
    
    print("Starting training...")
    print(f"State size: {state_size}")
    print(f"Action size: {action_size}")
    print(f"Device: {agent.device}")
    print()
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        episode_losses = []
        
        while not done:
            action = agent.select_action(state)
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.store_transition(state, action, reward, next_state, done)
            
            loss = agent.train()
            if loss > 0:
                episode_losses.append(loss)
            
            state = next_state
            total_reward += reward
        
        agent.decay_epsilon()
        
        if episode % target_update_freq == 0:
            agent.update_target_network()
        
        scores.append(total_reward)
        avg_score = np.mean(scores[-100:])
        avg_scores.append(avg_score)
        epsilons.append(agent.epsilon)
        
        if episode_losses:
            losses.append(np.mean(episode_losses))
        else:
            losses.append(0)
        
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{episodes} | Score: {total_reward:.0f} | "
                  f"Avg Score: {avg_score:.2f} | Epsilon: {agent.epsilon:.3f} | "
                  f"Loss: {losses[-1]:.4f}")
        
        if avg_score >= 195 and episode >= 100:
            print(f"\nEnvironment solved in {episode + 1} episodes!")
            print(f"Average score: {avg_score:.2f}")
            break
    
    env.close()
    
    agent.save('dqn_cartpole_model.pth')
    
    return agent, scores, avg_scores, losses, epsilons

def plot_training_results(scores, avg_scores, losses, epsilons):
    """
    Plot training metrics
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].plot(scores, alpha=0.6, label='Episode Score')
    axes[0, 0].plot(avg_scores, linewidth=2, label='Average Score (100 episodes)')
    axes[0, 0].axhline(y=195, color='r', linestyle='--', label='Target (195)')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_title('Training Scores Over Time')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(losses)
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Training Loss Over Time')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(epsilons)
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Epsilon')
    axes[1, 0].set_title('Exploration Rate (Epsilon) Decay')
    axes[1, 0].grid(True, alpha=0.3)
    
    window = 50
    if len(scores) >= window:
        rolling_avg = np.convolve(scores, np.ones(window)/window, mode='valid')
        axes[1, 1].plot(range(window-1, len(scores)), rolling_avg)
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_title(f'Rolling Average Score ({window} episodes)')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
    print("\nTraining results plot saved: training_results.png")
    plt.show()

def test_agent(agent, episodes=10, render=False):
    """
    Test the trained agent
    """
    if render:
        env = gym.make('CartPole-v1', render_mode='human')
    else:
        env = gym.make('CartPole-v1')
    
    test_scores = []
    
    print("\nTesting trained agent...")
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state, training=False)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            total_reward += reward
        
        test_scores.append(total_reward)
        print(f"Test Episode {episode + 1}: Score = {total_reward}")
    
    env.close()
    
    avg_test_score = np.mean(test_scores)
    print(f"\nAverage test score over {episodes} episodes: {avg_test_score:.2f}")
    
    return test_scores

def main():
    """
    Main execution function
    """
    print("=" * 60)
    print("DQN AGENT FOR CARTPOLE BALANCING")
    print("=" * 60)
    
    agent, scores, avg_scores, losses, epsilons = train_agent(episodes=500)
    
    plot_training_results(scores, avg_scores, losses, epsilons)
    
    test_scores = test_agent(agent, episodes=10)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Final average score (last 100 episodes): {np.mean(scores[-100:]):.2f}")
    print(f"Best episode score: {max(scores):.0f}")
    print(f"Average test score: {np.mean(test_scores):.2f}")
    print(f"Model saved: dqn_cartpole_model.pth")
    print(f"Results plot saved: training_results.png")

if __name__ == "__main__":
    main()
