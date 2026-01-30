# BalanceBot: Deep Q-Network Agent for CartPole

A reinforcement learning agent that learns to balance a pole on a moving cart using Deep Q-Learning (DQN).

## Project Overview

This project implements a DQN agent that learns to solve the classic CartPole-v1 environment from Gymnasium. The agent uses deep reinforcement learning to discover an optimal control policy through trial and error interactions with the environment.

## Features

- Deep Q-Network (DQN) implementation
- Experience replay buffer for stable training
- Target network for improved convergence
- Epsilon-greedy exploration strategy
- Comprehensive training visualizations
- Model saving and loading
- Test mode for evaluating trained agent

## Algorithms Used

**Deep Q-Network (DQN)**
- Value-based reinforcement learning
- Neural network Q-function approximation
- Experience replay to break correlation
- Target network for stability
- Epsilon-greedy exploration/exploitation

## Environment

**CartPole-v1** from Gymnasium

**State Space:**
- Cart Position: -2.4 to 2.4
- Cart Velocity: -Inf to Inf
- Pole Angle: -0.209 to 0.209 radians
- Pole Angular Velocity: -Inf to Inf

**Action Space:**
- 0: Push cart left
- 1: Push cart right

**Reward:**
- +1 for each timestep pole remains balanced
- Max score: 500
- Solved when average score >= 195 over 100 episodes

## Installation

### Prerequisites
- Python 3.8 or higher

### Install Dependencies

```bash
pip install gymnasium torch numpy matplotlib
```

Or use the requirements file:

```bash
pip install -r rl_requirements.txt
```

## Running the Agent

### Train from Scratch

```bash
python dqn_cartpole_agent.py
```

This will:
1. Initialize the DQN agent
2. Train for up to 500 episodes
3. Save the trained model
4. Generate training plots
5. Test the agent for 10 episodes

### Training Output

During training, you'll see:
- Episode number and score
- Average score over last 100 episodes
- Current exploration rate (epsilon)
- Training loss

### Files Generated

- **dqn_cartpole_model.pth**: Trained model weights
- **training_results.png**: Visualization of training metrics

## Training Results

The agent typically:
- Solves the environment in 250-350 episodes
- Achieves average test scores above 400
- Trains in 5-15 minutes on CPU
- Reaches max score of 500 regularly

## Hyperparameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Learning Rate | 0.001 | Step size for gradient descent |
| Gamma (γ) | 0.99 | Discount factor for future rewards |
| Epsilon Start | 1.0 | Initial exploration rate |
| Epsilon Min | 0.01 | Minimum exploration rate |
| Epsilon Decay | 0.995 | Exponential decay rate |
| Batch Size | 64 | Experience replay sample size |
| Buffer Size | 10000 | Replay buffer capacity |
| Hidden Units | 64 | Neural network layer size |
| Target Update | 10 | Episodes between target net updates |

## Code Structure

```
dqn_cartpole_agent.py
├── DQNetwork: Neural network architecture
├── ReplayBuffer: Experience storage and sampling
├── DQNAgent: Main agent with training logic
├── train_agent(): Training loop
├── plot_training_results(): Visualization
├── test_agent(): Evaluation
└── main(): Entry point
```

## Key Components

### DQNetwork
- 3-layer fully connected neural network
- ReLU activation functions
- Input: 4-dimensional state
- Output: 2 Q-values (one per action)

### ReplayBuffer
- Stores transition tuples
- Random sampling for training
- Breaks temporal correlation
- Efficient memory management

### DQNAgent
- Epsilon-greedy action selection
- Experience replay training
- Target network updates
- Model save/load functionality

## Viewing Results

The training generates a 4-panel visualization:

1. **Top Left**: Episode scores with moving average
2. **Top Right**: Training loss over time
3. **Bottom Left**: Epsilon decay curve
4. **Bottom Right**: 50-episode rolling average

## Testing a Trained Model

To test without retraining:

```python
import gymnasium as gym
from dqn_cartpole_agent import DQNAgent

env = gym.make('CartPole-v1')
agent = DQNAgent(state_size=4, action_size=2)
agent.load('dqn_cartpole_model.pth')

# Test for 10 episodes
from dqn_cartpole_agent import test_agent
test_agent(agent, episodes=10)
```

## Performance Analysis

**Strengths:**
- Consistent learning across runs
- High final performance
- Efficient training time
- Stable convergence

**Limitations:**
- Specific to CartPole environment
- Sensitive to hyperparameters
- No transfer learning capability
- Requires discrete action space

## Potential Improvements

- Prioritized experience replay
- Double DQN for better Q-value estimates
- Dueling DQN architecture
- Reward shaping
- Automated hyperparameter tuning

## Academic Context

This project was developed for an Artificial Intelligence course at Western Governors University. It demonstrates:
- Reinforcement learning algorithm implementation
- Deep neural network integration
- Experience replay techniques
- Exploration/exploitation strategies
- Performance evaluation methods

## References

- Mnih et al. (2015). Human-level control through deep reinforcement learning. Nature.
- Sutton & Barto (2018). Reinforcement Learning: An Introduction.
- Gymnasium Documentation: https://gymnasium.farama.org

## Author

Christopher Garcia  
Western Governors University  
Artificial Intelligence Course  
January 2025

## License

This is a student project for educational purposes.
