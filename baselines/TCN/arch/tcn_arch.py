import torch
from torch import nn
import torch.nn.functional as F


class Chomp1d(nn.Module):
    """移除padding产生的未来信息，确保因果性"""

    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TCNBlock(nn.Module):
    """TCN基本块，包含残差连接 - 移除weight_norm提高稳定性"""

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super(TCNBlock, self).__init__()

        # 计算padding以保持序列长度
        padding = (kernel_size - 1) * dilation

        # 第一个卷积层 - 移除weight_norm
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # 第二个卷积层 - 移除weight_norm
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # 组合网络
        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )

        # 残差连接的维度匹配
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

        self.init_weights()

    def init_weights(self):
        """初始化权重 - 使用更稳定的初始化"""
        # 使用Xavier/Glorot初始化
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.bias)

        if self.downsample is not None:
            nn.init.xavier_uniform_(self.downsample.weight)
            nn.init.zeros_(self.downsample.bias)

    def forward(self, x):
        """前向传播"""
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCN(nn.Module):
    """稳定的TCN模型"""

    def __init__(self, num_layers, in_dim, hidden_dim, output_dim, kernel_size=3, dropout=0.2):
        super(TCN, self).__init__()

        self.num_layers = num_layers
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.dropout_p = dropout

        # 输入编码层
        self.encoder = nn.Conv1d(in_dim, hidden_dim, kernel_size=1)
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)

        # TCN层
        self.tcn_layers = nn.ModuleList()
        for i in range(num_layers):
            dilation = 2 ** i
            tcn_block = TCNBlock(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=kernel_size,
                dilation=dilation,
                dropout=dropout
            )
            self.tcn_layers.append(tcn_block)

        # 输出投影层
        self.proj = nn.Conv1d(hidden_dim, output_dim, kernel_size=1)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor = None,
                batch_seen: int = None, epoch: int = None, train: bool = True,
                **kwargs) -> torch.Tensor:
        """
        前向传播

        Args:
            history_data (torch.Tensor): 输入数据，shape [B, L, N, C]

        Returns:
            torch.Tensor: 输出数据，shape [B, L, N, output_dim]
        """
        # 输入验证
        if torch.isnan(history_data).any() or torch.isinf(history_data).any():
            print("Warning: Input contains NaN or Inf values!")
            history_data = torch.nan_to_num(history_data, nan=0.0, posinf=1e6, neginf=-1e6)

        B, T, N, D = history_data.shape

        # 重塑为卷积期望的格式: (B*N, D, T)
        x = history_data.permute(0, 2, 3, 1).contiguous()  # [B, N, D, T]
        x = x.view(B * N, D, T)  # [B*N, D, T]

        # 特征编码
        x = self.encoder(x)  # (B*N, hidden_dim, T)

        # 检查编码后是否有异常值
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("Warning: Encoder output contains NaN or Inf!")
            x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)

        # 通过TCN层
        for i, tcn_block in enumerate(self.tcn_layers):
            x_before = x
            x = tcn_block(x)

            # 检查每层输出
            if torch.isnan(x).any() or torch.isinf(x).any():
                print(f"Warning: TCN layer {i} output contains NaN or Inf!")
                x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)

            # 检查是否全为0
            if torch.all(x == 0):
                print(f"Warning: TCN layer {i} output is all zeros!")
                x = x_before * 0.1  # 使用前一层的缩小版本

        # 输出投影
        x = self.proj(x)  # (B*N, output_dim, T)

        # 最终检查
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("Warning: Final output contains NaN or Inf!")
            x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)

        # 🔥 修正维度重塑错误
        x = x.permute(0, 2, 1)  # (B*N, T, output_dim)
        x = x.view(B, N, T, self.output_dim)  # 正确的view操作
        x = x.permute(0, 2, 1, 3)  # (B, T, N, output_dim)

        return x


class StableTrainingTCN(nn.Module):
    """额外稳定性改进的TCN版本"""

    def __init__(self, num_layers, in_dim, hidden_dim, output_dim, kernel_size=3, dropout=0.1):
        super(StableTrainingTCN, self).__init__()

        self.num_layers = num_layers
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # 降低dropout率提高稳定性
        self.dropout_p = min(dropout, 0.1)

        # 输入归一化
        self.input_norm = nn.LayerNorm(in_dim)

        # 特征编码
        self.encoder = nn.Sequential(
            nn.Conv1d(in_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )

        # TCN层 - 使用更保守的设置
        self.tcn_layers = nn.ModuleList()
        for i in range(num_layers):
            dilation = 2 ** i
            layer = nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size,
                          padding=(kernel_size - 1) * dilation, dilation=dilation),
                Chomp1d((kernel_size - 1) * dilation),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout_p)
            )
            self.tcn_layers.append(layer)

        # 输出层
        self.output_layer = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=1),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Conv1d(hidden_dim // 2, output_dim, kernel_size=1)
        )

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """稳定的权重初始化"""
        if isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, history_data: torch.Tensor, **kwargs) -> torch.Tensor:
        """稳定的前向传播"""
        B, T, N, D = history_data.shape

        # 输入归一化
        x = self.input_norm(history_data)

        # 重塑维度
        x = x.permute(0, 2, 3, 1).contiguous().view(B * N, D, T)

        # 编码
        x = self.encoder(x)

        # TCN处理
        for layer in self.tcn_layers:
            residual = x
            x = layer(x)
            # 残差连接
            x = x + residual
            # 防止梯度爆炸
            x = torch.clamp(x, -10, 10)

        # 输出
        x = self.output_layer(x)

        # 重塑回原始格式
        x = x.permute(0, 2, 1).contiguous()  # (B*N, T, output_dim)
        x = x.view(B, N, T, self.output_dim).permute(0, 2, 1, 3)

        return x

