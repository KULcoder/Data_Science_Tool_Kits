"""
This file is intended for writing a mlp using pytorch from scratch.
"""

import torch 
import torch.nn as nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from models.layers_pytorch.activations import relu
from models.layers.normalizations.softmax import softmax
from models.layers_pytorch.linear import Linear
from Trianer.trainer_pytorch import Trainer
from data.pytorch.mnist import get_loaders

class MLP(nn.Module):
    def __init__(self):
        # Very brutal 3072, 128, 10 mlp
        super(MLP, self).__init__()
        self.layer1 = Linear(784, 3072)
        self.layer2 = Linear(3072, 128)
        self.layer3 = Linear(128, 10)

    def forward(self, x):
        x = relu(self.layer1(x))
        x = relu(self.layer2(x))
        x = softmax(self.layer3(x))
        return x

if __name__ == '__main__':
    pass
    