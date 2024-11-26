import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import dataLoader as dl

class Env():
    def __init__(self, data):
        self.action_space = ["up %10", "down %10", "nothing"]
        self.n_actions = len(self.action_space)
        self.num_assets = data.shape[1]
        self.state_space_dim = self.num_assets
        #data
        self.data = data
        self.current_step = 0 # başlangıç zamanı
        self.portfolio_value = 1000 # başlangıç sermayesi
        self.allocation = np.zeros(data.shape[1])
        

    def reset(self):
        self.current_step = 0
        self.portfolio_value = 1000
        self.allocation = np.zeros(self.data.shape[1])  # Her varlık için portföy dağılımı
        state = self._get_state()
        return self._normalize_state(state)  # Normalize edilmiş sabit boyutlu state döndür

    def step(self, action):
        self.current_step += 1
        if action == 0:
            self.allocation += 10
        elif action == 1:
            self.allocation -= 10
        self.allocation = np.clip(self.allocation, -1, 1)
        returns = self.data.iloc[self.current_step].values
        self.portfolio_value *= (1 + np.dot(self.allocation, returns))
        if np.isnan(returns).any() or np.isinf(returns).any():
            print(f"Invalid returns: {returns}")

        if np.isnan(self.allocation).any() or np.isinf(self.allocation).any():
            print(f"Invalid allocation: {self.allocation}")


        # NaN ve inf kontrolü
        if np.isnan(self.portfolio_value) or np.isinf(self.portfolio_value):
            print(f"Invalid portfolio value: {self.portfolio_value}")
            self.portfolio_value = 1000  # Reset to initial value

        reward = self.portfolio_value - 1000
        
        # NaN ve inf kontrolü
        if np.isnan(reward) or np.isinf(reward):
            print(f"Invalid reward: {reward}")
            reward = 0  # Reset to zero

        # Ödül normalizasyonu
        reward = np.clip(reward, -1, 1)

        done = self.current_step >= len(self.data) - 1
        state = self._get_state()
        return self._normalize_state(state), reward, done  # Normalize edilmiş state
    
    def _get_state(self):
        """Durumu döndürmek için kullanılır."""
        # Örnek: Son 10 adımın getirileri
        window_size = 10
        start_index = max(0, self.current_step - window_size)
        end_index = self.current_step
        state = self.data.iloc[start_index:end_index].values
        return state

    def _normalize_state(self, state):
        """Sabit boyutta state döndürmek için doldurma."""
        fixed_size = 10  # Her state sabit olarak 10 veri içermeli
        state = np.array(state)
        if len(state) < fixed_size:
            padding = np.zeros((fixed_size - len(state), state.shape[1]))
            state = np.vstack((padding, state))
        return state