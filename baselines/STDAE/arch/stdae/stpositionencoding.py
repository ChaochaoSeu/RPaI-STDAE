import torch
import torch.nn as nn
import math

# 设计二维位置编码，同时捕捉时间和空间位置信息：
class SpatioTemporalPositionalEncoding(nn.Module):

    def __init__(self, num_feat, max_len=10000):
        """
        Args:
            num_feat: 特征维度（必须是4的倍数）
            max_len: 最大长度，用于归一化
        """
        super().__init__()

        assert num_feat % 4 == 0, "num_feat must be divisible by 4"
        self.num_feat = num_feat
        self.max_len = max_len

    def forward(self, x):
        """
        Args:
            x: 输入张量 [B, H, W, C] 或 [B, N, P, D]

        Returns:
            带位置编码的张量
        """
        B, H, W, C = x.shape

        # 生成位置网格
        y_pos = torch.arange(H, dtype=torch.float32, device=x.device)
        x_pos = torch.arange(W, dtype=torch.float32, device=x.device)

        # 计算频率
        div_term = torch.exp(torch.arange(0, C // 2, 2, device=x.device).float() *
                             (-math.log(self.max_len) / (C // 2)))

        # 初始化位置编码
        pos_enc = torch.zeros(H, W, C, device=x.device)

        # Y方向位置编码（前C/2维）
        pos_enc[:, :, 0:C // 2:2] = torch.sin(y_pos.unsqueeze(1).unsqueeze(2) * div_term)
        pos_enc[:, :, 1:C // 2:2] = torch.cos(y_pos.unsqueeze(1).unsqueeze(2) * div_term)

        # X方向位置编码（后C/2维）
        pos_enc[:, :, C // 2::2] = torch.sin(x_pos.unsqueeze(0).unsqueeze(2) * div_term)
        pos_enc[:, :, C // 2 + 1::2] = torch.cos(x_pos.unsqueeze(0).unsqueeze(2) * div_term)

        # 添加batch维度并与输入相加
        pos_enc = pos_enc.unsqueeze(0).expand(B, -1, -1, -1)

        return x + pos_enc, pos_enc

