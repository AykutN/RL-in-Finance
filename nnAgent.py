import torch as T
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(QNetwork, self).__init__()
        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = T.relu(self.linear1(state))
        x = T.relu(self.linear2(x))
        q_values = self.linear3(x)
        return q_values
    
class ReplayBuffer():
    def __init__(self, max_size, state_dim):
        self.max_size = max_size
        self.state_dim = state_dim
        self.memory = []
        self.position = 0

    def store(self, state, action, reward, next_state, done):
        state = np.array(state) if not isinstance(state, np.ndarray) else state
        next_state = np.array(next_state) if not isinstance(next_state, np.ndarray) else next_state

        if np.isnan(state).any() or np.isnan(next_state).any():
            print("NaN detected in state or next_state")
            return

        if len(self.memory) < self.max_size:
            self.memory.append(None)
        self.memory[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.max_size

    def sample(self, batch_size):
        indices = np.random.choice(len(self.memory), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.memory[idx] for idx in indices])

        states = np.array(states)
        next_states = np.array(next_states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        dones = np.array(dones)

        if np.isnan(states).any() or np.isnan(next_states).any():
            print("NaN detected in sampled states or next_states")
            return None, None, None, None, None

        return (states, actions, rewards, next_states, dones)

class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=0.0005, gamma = 0.995, buffer_size = 10000, batch_size = 64):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma

        self.epsilon_decay = 0.99
        self.epsilon_min = 0.01
        self.epsilon = 1.0
        self.batch_size = batch_size

        self.q_network = QNetwork(state_dim, action_dim)
        self.target_network = QNetwork(state_dim, action_dim)
        self.step = 0  # Initialize step counter

        self.target_network.eval()
        self.update_target_network()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        #self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.9)
        
        self.criterion = nn.MSELoss()
        self.buffer = ReplayBuffer(buffer_size, state_dim)

        print("DQNAgent initialized")

    def update_target_network(self):
            self.target_network.load_state_dict(self.q_network.state_dict())
            

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_dim)
        else:
            state = T.tensor(state, dtype=T.float32).reshape(-1)
            q_values = self.q_network(state)
            return T.argmax(q_values, dim=0).item()

    def train(self):
        if len(self.buffer.memory) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        if states is None:
            print("Sampling failed due to NaN values")
            return None

        states = T.tensor(states, dtype=T.float32).reshape(self.batch_size, -1)
        next_states = T.tensor(next_states, dtype=T.float32).reshape(self.batch_size, -1)
        actions = T.tensor(actions, dtype=T.int64).unsqueeze(1)
        rewards = T.tensor(rewards, dtype=T.float32)
        dones = T.tensor(dones, dtype=T.float32)

        q_values = self.q_network(states).gather(1, actions)

        with T.no_grad():
            next_q_values = self.target_network(next_states).max(1, keepdim=True)[0]
            targets = rewards + self.gamma * next_q_values * (1 - dones)

        if T.isnan(q_values).any() or T.isnan(targets).any() or T.isinf(q_values).any() or T.isinf(targets).any():
            print("NaN or inf detected in q_values or targets")
            return None

        loss = self.criterion(q_values, targets)

        if T.isnan(loss) or T.isinf(loss):
            print("NaN or inf detected in loss")
            return None

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        #self.lr_scheduler.step()


        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        self.step += 1
        
        return loss.item()
