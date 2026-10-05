import torch
import torch.nn as nn
import math
from typing import List

class Solution:
    def xavier_init(self, fan_in: int, fan_out: int, set_seed: bool = True, as_tensor: bool = False) -> List[List[float]]:
        if set_seed:
            torch.manual_seed(0)
        std = math.sqrt(2.0 / (fan_in + fan_out))
        w = torch.randn(fan_out, fan_in) * std
        if as_tensor:
            return w
        return torch.round(w, decimals=4).tolist()

    def kaiming_init(self, fan_in: int, fan_out: int, set_seed: bool = True, as_tensor: bool = False) -> List[List[float]]:
        if set_seed:
            torch.manual_seed(0)
        std = math.sqrt(2.0 / fan_in)
        w = torch.randn(fan_out, fan_in) * std
        if as_tensor:
            return w
        return torch.round(w, decimals=4).tolist()

    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> List[float]:
        # 1. Seed once at the start
        torch.manual_seed(0)
        
        # 2. Collect weights using helper methods without resetting the seed
        curr_in = input_dim
        weights = []
        for _ in range(num_layers):
            if init_type == 'kaiming':
                mat = self.kaiming_init(curr_in, hidden_dim, set_seed=False, as_tensor=True)
            elif init_type == 'xavier':
                mat = self.xavier_init(curr_in, hidden_dim, set_seed=False, as_tensor=True)
            else:
                mat = torch.randn(hidden_dim, curr_in)
            weights.append(mat)
            curr_in = hidden_dim
            
        # 3. Sample initial input vector
        x = torch.randn(1, input_dim)
        
        # 4. Sequentially propagate activations through layers
        res = []
        for mat in weights:
            x = x @ mat.T
            x = torch.relu(x)
            res.append(round(x.std().item(), 2))
            
        return res