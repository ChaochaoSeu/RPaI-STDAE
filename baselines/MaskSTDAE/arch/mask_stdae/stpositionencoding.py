import torch
import torch.nn as nn
import math

# Design 2D positional encoding to capture both temporal and spatial information
class SpatioTemporalPositionalEncoding(nn.Module):

    def __init__(self, num_feat, max_len=10000):
        """
        Args:
            num_feat: Feature dimension (must be divisible by 4)
            max_len: Maximum length used for normalization
        """
        super().__init__()

        assert num_feat % 4 == 0, "num_feat must be divisible by 4"
        self.num_feat = num_feat
        self.max_len = max_len

    def forward(self, x):
        """
        Args:
            x: Input tensor [B, H, W, C] or [B, N, P, D]

        Returns:
            Tensor with added positional encoding
        """
        B, H, W, C = x.shape

        # Generate position grid
        y_pos = torch.arange(H, dtype=torch.float32, device=x.device)
        x_pos = torch.arange(W, dtype=torch.float32, device=x.device)

        # Compute frequency
        div_term = torch.exp(torch.arange(0, C // 2, 2, device=x.device).float() *
                             (-math.log(self.max_len) / (C // 2)))

        # Initialize positional encoding
        pos_enc = torch.zeros(H, W, C, device=x.device)

        # Y-axis positional encoding (first C/2 dimensions)
        pos_enc[:, :, 0:C // 2:2] = torch.sin(y_pos.unsqueeze(1).unsqueeze(2) * div_term)
        pos_enc[:, :, 1:C // 2:2] = torch.cos(y_pos.unsqueeze(1).unsqueeze(2) * div_term)

        # X-axis positional encoding (last C/2 dimensions)
        pos_enc[:, :, C // 2::2] = torch.sin(x_pos.unsqueeze(0).unsqueeze(2) * div_term)
        pos_enc[:, :, C // 2 + 1::2] = torch.cos(x_pos.unsqueeze(0).unsqueeze(2) * div_term)

        # Add batch dimension and sum with input
        pos_enc = pos_enc.unsqueeze(0).expand(B, -1, -1, -1)

        return x + pos_enc, pos_enc
