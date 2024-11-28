import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import dataLoader as dl

class Env():
    def __init__(self, data):
        self.num_assets = data.shape[1]
        self.action_space = self._generate_action_space()
        self.n_actions = len(self.action_space)
        self.state_space_dim = self.num_assets
        # data
        self.data = data
        self.current_step = 0  # başlangıç zamanı
        self.portfolio_value = 1000  # başlangıç sermayesi
        self.previous_portfolio_value = self.portfolio_value
        self.allocation = np.zeros(data.shape[1])

    def _generate_action_space(self):
        actions = ["up %10", "down %10", "nothing"]
        action_space = []
        for action1 in actions:
            for action2 in actions:
                action_space.append((action1, action2))
        return action_space

    def reset(self):
        self.current_step = 0
        self.portfolio_value = 1000
        self.previous_portfolio_value = self.portfolio_value
        self.allocation = np.zeros(self.data.shape[1])
        state = self._get_state()
        return self._normalize_state(state)

    def step(self, action):
        self.current_step += 1

        action1, action2 = self.action_space[action]
        if action1 == "up %10":
            self.allocation[0] += 10
        elif action1 == "down %10":
            self.allocation[0] -= 10

        if action2 == "up %10":
            self.allocation[1] += 10
        elif action2 == "down %10":
            self.allocation[1] -= 10

        self.allocation = np.clip(self.allocation, -10, 10) / 100

        returns = self.data.iloc[self.current_step].values
        self.portfolio_value *= (1 + np.dot(self.allocation, returns))

        if np.isnan(self.portfolio_value) or np.isinf(self.portfolio_value):
            print(f"Invalid portfolio value: {self.portfolio_value}")
            self.portfolio_value = 1000

        
        reward = self.portfolio_value - self.previous_portfolio_value
        reward = np.clip(reward, -10, 10) 

        done = self.current_step >= len(self.data) - 1
        state = self._get_state()
        return self._normalize_state(state), reward, done

    def _get_state(self):
        """Durumu döndürmek için kullanılır."""
        window_size = 10
        start_index = max(0, self.current_step - window_size)
        end_index = self.current_step
        state = self.data.iloc[start_index:end_index].values
        return state

    def _normalize_state(self, state):
        """Sabit boyutta state döndürmek için doldurma ve normalizasyon."""
        fixed_size = 10  # Her state sabit olarak 10 veri içermeli
        state = np.array(state)
        if len(state) < fixed_size:
            padding = np.zeros((fixed_size - len(state), state.shape[1]))
            state = np.vstack((padding, state))

        # Min-max normalizasyonu
        state_min = np.min(state, axis=0)
        state_max = np.max(state, axis=0)
        normalized_state = (state - state_min) / (state_max - state_min + 1e-8)  # 1e-8 sıfıra bölmeyi önler

        return normalized_state
