"""
A custom PyTorch linear layer.
"""

import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import init
import math

class Linear(nn.Module):
    def __init__(self, input_size, output_size):
        super(Linear, self).__init__()
        self.weights = Parameter(
            torch.zeros((output_size, input_size))
        )
        self.bias = Parameter(
            torch.zeros(output_size)
        )

        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0

        init.kaiming_uniform_(self.weights)
        init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return x @ self.weights.T + self.bias