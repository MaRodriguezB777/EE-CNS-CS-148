import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ScaledDotProductAttention import ScaledDotProductAttention
from MultiHeadAttention import MultiHeadAttention
from TransformerBlock import TransformerBlock
from ViT import ViT

# Set random seed to keep experiments consistent
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

class ViT_Attention_tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d_k = 12
        # batch of 3 elements, each with 4 vectors of size dk=12
        cls.x = torch.tensor(np.array(3 * [np.random.rand(4, cls.d_k)]))

        # attention
        sdpa = ScaledDotProductAttention(cls.d_k)
        cls.att_output, _ = sdpa.forward(cls.x, cls.x, cls.x)

    def test_isimplemented(self):
        self.assertIsNotNone(self.att_output, "Attention is not implemented")

    def test_shape(self):
        self.assertEqual(self.att_output.shape,
                         self.x.shape,
                         "Shape of output should be (batch_size, num_queries, d_k)")

    def test_outputs(self):
        expected_output, _ = F._scaled_dot_product_attention(self.x, self.x, self.x)
        self.assertTrue(torch.allclose(self.att_output, expected_output),
                         'Incorrect outputs from SDPA')

class ViT_Multihead_tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d_input = 12
        cls.num_heads = 3
        cls.d_k = 4

        # batch of 3 elements, each with 4 vectors of size dk=12
        cls.x = torch.tensor(np.array(3 * [np.random.rand(4, cls.d_input)])).to(torch.float32)

        # multihead attention
        mha = MultiHeadAttention(cls.d_input, cls.num_heads)
        cls.mha_output, _ = mha.forward(cls.x, cls.x, cls.x)

    def test_isimplemented(self):
        self.assertFalse(torch.allclose(0*self.x, self.mha_output),
                        "Multihead is not implemented")

    def test_shape(self):
        self.assertEqual(self.mha_output.shape,
                         self.x.shape,
                         "Shape of output should be (batch_size, num_queries, d_input)")

    def test_outputs(self):
        # Create both custom and PyTorch MultiHeadAttention modules
        mha = MultiHeadAttention(self.d_input, self.num_heads)
        expected_mha = nn.modules.activation.MultiheadAttention(self.d_input, self.num_heads, batch_first=True)

        # Make sure W_Q, W_K, W_V are initialized exactly as in PyTorch's MultiheadAttention
        mha.W_Q.weight = nn.Parameter(expected_mha.in_proj_weight[:self.num_heads*self.d_k, :])
        mha.W_Q.bias = nn.Parameter(expected_mha.in_proj_bias[:self.num_heads*self.d_k])

        mha.W_K.weight = nn.Parameter(expected_mha.in_proj_weight[self.num_heads*self.d_k:2*self.num_heads*self.d_k, :])
        mha.W_K.bias = nn.Parameter(expected_mha.in_proj_bias[self.num_heads*self.d_k:2*self.num_heads*self.d_k])

        mha.W_V.weight = nn.Parameter(expected_mha.in_proj_weight[2*self.num_heads*self.d_k:, :])
        mha.W_V.bias = nn.Parameter(expected_mha.in_proj_bias[2*self.num_heads*self.d_k:])

        mha.W_O.weight = nn.Parameter(expected_mha.out_proj.weight)
        mha.W_O.bias = nn.Parameter(expected_mha.out_proj.bias)

        # multihead attention - forward
        self.mha_output, _ = mha.forward(self.x, self.x, self.x)
        expected_output, _ = expected_mha.forward(self.x, self.x, self.x)

        self.assertTrue(torch.allclose(self.mha_output, expected_output),
                         'Incorrect outputs from MHA')

class ViT_Transformerblock_tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.input_dim = 12
        cls.num_heads = 3
        cls.mlp_dim = 256
        # batch of 3 elements, each with 5 tokens of size input_dim=12
        cls.x = torch.tensor(np.array(3 * [np.ones((5, cls.input_dim))*1.])).to(torch.float32)

        # transformer block
        tb = TransformerBlock(cls.input_dim, cls.num_heads, cls.mlp_dim)
        cls.tb_output = tb.forward(cls.x)

    def test_shape(self):
        self.assertEqual(self.tb_output.shape,
                         self.x.shape,
                         "Shape of output should be (batch_size, num_tokens, input_dim)")

    def test_outputs(self):
        tb_output_0_0 = self.tb_output[0][0]
        expected_output_0_0 = torch.tensor([1.1292, 1.1134, 1.1492, 1.4138, 0.6081, 0.9295,
                                            1.0086, 1.2664, 0.8715, 0.9355, 0.9873, 0.8053])
        self.assertTrue(torch.allclose(tb_output_0_0, expected_output_0_0,rtol=1e-03, atol=1e-05),
                         'Incorrect outputs from SDPA')

if __name__ == '__main__':
    # Run tests in provided order
    unittest.TestLoader.sortTestMethodsUsing = None
    unittest.main(verbosity=2)