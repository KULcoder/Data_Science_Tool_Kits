"""
This provides an example on using custom attention layer
to do image classification task using transformer model.

The model size is not very big, and the performance is also not superior:
around 90% test accuracy.
"""

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from models.layers_pytorch.attentions import ScaledDotProductAttention
import torch.nn as nn

class MNISTSelfAttentionModel(nn.Module):
    def __init__(self, input_dim, d_k, num_classes):
        super(MNISTSelfAttentionModel, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_dim, d_k)
        self.attention = ScaledDotProductAttention(d_k)
        self.fc2 = nn.Linear(d_k, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        Q = K = V = x.unsqueeze(1) # Use the same input for Q, K, V
        attn_output, _ = self.attention(Q, K, V)
        attn_output = attn_output.squeeze(1)
        out = self.fc2(attn_output)
        return out
