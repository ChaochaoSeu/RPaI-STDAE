import torch
import torch.nn as nn



class GRU(nn.Module):
    """
    简化版多步预测GRU，使用PyTorch内置GRU
    """

    def __init__(self, input_dim, hidden_dim, num_layers, output_dim=1, dropout=0.1):
        super(GRU, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim

        # 使用PyTorch内置GRU
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # 输出投影层
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool, **kwargs) -> torch.Tensor:
        """
        Args:
            x: [B, T, N, D]
        Returns:
            [B, T, N, output_dim]
        """
        B, T, N, D = history_data.shape

        # 重塑为GRU期望的格式 [B*N, T, D]
        x_reshaped = history_data.permute(0, 2, 1, 3).contiguous().view(B * N, T, D)

        # 通过GRU
        gru_output, _ = self.gru(x_reshaped)  # [B*N, T, hidden_dim]

        # 通过输出投影层
        output = self.output_projection(gru_output)  # [B*N, T, output_dim]

        # 重塑回原始格式
        output = output.view(B, N, T, self.output_dim).permute(0, 2, 1, 3)  # [B, T, N, output_dim]

        return output