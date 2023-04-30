import torch.nn as nn
from MultiHeadAttention import MultiHeadAttention

class TransformerBlock(nn.Module):
    ''' Implements the Transformer Encoder block:
            [LayerNorm, Multi-head attention, residual,
             LayerNorm, MLP, residual]

    '''
    def __init__(self, input_dim, num_heads, mlp_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()

        # TODO define LayerNorm (use nn.LayerNorm -- https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html)
        self.layer_norm1 = None
        self.layer_norm2 = None

        # TODO define MLP (one linear, one activation function, one linear -- go back to the ViT paper to find which
        #      activation function you should be using)
        self.mlp = None

        # TODO define MultiHeadAttention
        self.multihead_attention = None

    def forward(self, x):
        # TODO apply LayerNorm
        # TODO define Q, K, V
        # TODO pass (Q, K, V) to MultiHeadAttention
        # TODO sum with 1st residual connection

        # TODO apply LayerNorm
        # TODO pass to MLP
        # TODO sum with 2nd residual connection
        return x
