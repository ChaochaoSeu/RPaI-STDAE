import torch
import torch.nn
from torch import nn
from dataclasses import dataclass

@dataclass
class LSTMOpt:
    input_dim: int
    hidden_dim: int
    num_layers: int
    dirnum: int

class LSTMLinear(nn.Module):
    def __init__(self, input_dim: int, num_lstm_units: int, output_dim: int, bias: float = 0.0):
        super(LSTMLinear, self).__init__()
        self._input_dim = input_dim
        self._num_lstm_units = num_lstm_units
        self._output_dim = output_dim
        self._bias_init_value = bias
        # 修改权重维度以接收特征而不是单一值
        self.weights = nn.Parameter(
            torch.FloatTensor(self._input_dim + self._num_lstm_units, self._output_dim)
        )
        self.biases = nn.Parameter(torch.FloatTensor(self._output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights)
        nn.init.constant_(self.biases, self._bias_init_value)

    def forward(self, inputs, hidden_state):
        batch_size, num_nodes, features_dim = inputs.shape
        # inputs已经是(batch_size, num_nodes, features_dim)
        # hidden_state (batch_size, num_nodes, num_lstm_units)
        hidden_state = hidden_state.reshape(
            (batch_size, num_nodes, self._num_lstm_units)
        )
        # [inputs, hidden_state] "[x, h]" (batch_size, num_nodes, features_dim + num_lstm_units)
        concatenation = torch.cat((inputs, hidden_state), dim=2)
        # [x, h] (batch_size * num_nodes, features_dim + num_lstm_units)
        concatenation = concatenation.reshape((-1, features_dim + self._num_lstm_units))
        # [x, h]W + b (batch_size * num_nodes, output_dim)
        outputs = concatenation @ self.weights + self.biases
        # [x, h]W + b (batch_size, num_nodes, output_dim)
        outputs = outputs.reshape((batch_size, num_nodes, self._output_dim))
        # [x, h]W + b (batch_size, num_nodes * output_dim)
        outputs = outputs.reshape((batch_size, num_nodes * self._output_dim))
        return outputs

    def hyperparameters(self):
        return {
            "input_dim": self._input_dim,
            "num_lstm_units": self._num_lstm_units,
            "output_dim": self._output_dim,
            "bias_init_value": self._bias_init_value,
        }

class LSTMCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super(LSTMCell, self).__init__()
        self._input_dim = input_dim  # num_nodes
        self._hidden_dim = hidden_dim
        # 线性层更新为接收features_dim而不是1
        self.linear_gates = LSTMLinear(self._input_dim, self._hidden_dim, self._hidden_dim * 4, bias=1.0)

    def forward(self, inputs, states):
        # 在LSTM中，状态包括隐藏状态h和单元状态c
        hidden_state, cell_state = states

        # 计算所有门值: [i, f, g, o] = ?([x, h]W + b)
        # gates_output (batch_size, num_nodes * (4 * num_lstm_units))
        gates_output = self.linear_gates(inputs, hidden_state)

        # 将输出分成四个部分，对应四个门
        # 每个门: (batch_size, num_nodes * num_lstm_units)
        i, f, g, o = torch.chunk(gates_output, chunks=4, dim=1)

        # 应用激活函数
        i = torch.sigmoid(i)  # 输入门
        f = torch.sigmoid(f)  # 遗忘门
        g = torch.tanh(g)  # 候选细胞状态
        o = torch.sigmoid(o)  # 输出门

        # 更新细胞状态: c_t = f_t * c_{t-1} + i_t * g_t
        new_cell_state = f * cell_state + i * g

        # 计算新的隐藏状态: h_t = o_t * tanh(c_t)
        new_hidden_state = o * torch.tanh(new_cell_state)

        return (new_hidden_state, new_cell_state), new_hidden_state

    @property
    def hyperparameters(self):
        return {
            "input_dim": self._input_dim,
            "hidden_dim": self._hidden_dim,
        }




class LSTM(nn.Module):
    def __init__(self, short_input_dim, dir_num, lstm_hidden_dim, lstm_num_layers, output_steps, output_dim, mlp_hidden_dim, restruct_hidden_dim=96):
        super().__init__()
        self.short_input_dim = short_input_dim # 短时输入特征数量
        self.dir_num = dir_num # 默认为12
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_num_layers = lstm_num_layers
        self.output_steps = output_steps
        self.output_dim = output_dim
        self.mlp_hidden_dim = mlp_hidden_dim
        self.restruct_hidden_dim = restruct_hidden_dim

        # LSTM 配置
        lstm_opt = LSTMOpt(
            input_dim=short_input_dim,
            hidden_dim=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            dirnum=dir_num
        )

        self.fc_his_t = nn.Sequential(nn.Linear(restruct_hidden_dim, mlp_hidden_dim), nn.ReLU(), nn.Linear(mlp_hidden_dim, lstm_hidden_dim), nn.ReLU())
        self.fc_his_s = nn.Sequential(nn.Linear(restruct_hidden_dim, mlp_hidden_dim), nn.ReLU(), nn.Linear(mlp_hidden_dim, lstm_hidden_dim), nn.ReLU())

        # ========== LSTM 层 ==========
        self.lstm_layers = nn.ModuleList()
        for layer_idx in range(lstm_num_layers):
            input_dim = lstm_opt.input_dim if layer_idx == 0 else lstm_opt.hidden_dim
            self.lstm_layers.append(LSTMCell(input_dim, lstm_opt.hidden_dim))
        # ========== 输出投影层 ==========
        self.output_projection = nn.Linear(lstm_hidden_dim, output_dim)

        # ========== 多步预测头 ==========
        self.prediction_head = nn.Sequential(
            nn.Linear(lstm_hidden_dim, lstm_hidden_dim),
            nn.ReLU(),
            nn.Linear(lstm_hidden_dim, output_steps * output_dim)
        )


    def forward(self, input, hidden_states_t, hidden_states_s):
        """
            feed forward function
            Args:
                input (torch.Tensor): short_term_history with shape [B, L, N, D]
                hidden_states_t (torch.Tensor | None): temporal states, [B, N, d]
                hidden_states_s (torch.Tensor | None): spatial states, [B, N, d]

            Returns:
                torch.Tensor: prediction with shape [B, N, output_steps, output_dim]
            """
        B, L, N, D = input.shape
        x = input.permute(0, 3, 2, 1).contiguous() # [B, N, D, L]

        x = x.permute(0, 3, 2, 1).contiguous()  # [B, L, N, D]

        # ====== 处理 hidden_states 融合逻辑 ======
        init_hidden_state = None
        if hidden_states_t is not None and hidden_states_s is not None:
            # 两个都有
            hidden_states_t = self.fc_his_t(hidden_states_t)
            hidden_states_s = self.fc_his_s(hidden_states_s)
            fused_state = hidden_states_t + hidden_states_s
            init_hidden_state = fused_state.reshape(B, N * self.lstm_hidden_dim)
        elif hidden_states_t is not None:
            # 只有 temporal
            hidden_states_t = self.fc_his_t(hidden_states_t)
            init_hidden_state = hidden_states_t.reshape(B, N * self.lstm_hidden_dim)
        elif hidden_states_s is not None:
            # 只有 spatial
            hidden_states_s = self.fc_his_s(hidden_states_s)
            init_hidden_state = hidden_states_s.reshape(B, N * self.lstm_hidden_dim)
        else:
            # 两个都是 None → 全零初始化
            init_hidden_state = torch.zeros(B, N * self.lstm_hidden_dim, device=input.device)


        # 初始化所有层的隐藏状态和单元状态
        lstm_hidden_states = []
        lstm_cell_states = []

        for layer_idx in range(self.lstm_num_layers):
            if layer_idx == 0:
                # 第一层使用历史状态初始化
                lstm_hidden_states.append(init_hidden_state)
                lstm_cell_states.append(torch.zeros_like(init_hidden_state))
            else:
                # 其他层用零初始化
                lstm_hidden_states.append(init_hidden_state)
                lstm_cell_states.append(torch.zeros_like(init_hidden_state))
        outputs = []
        for t in range(L):
            current_input = x [:, t]  # [B, N, D]

            # 通过所有LSTM层
            for layer_idx in range(self.lstm_num_layers):
                if layer_idx == 0:
                    layer_input = current_input
                else:
                    # 重塑上一层的输出为当前层的输入
                    layer_input = lstm_hidden_states[layer_idx - 1].reshape(B, N, self.lstm_hidden_dim)
                lstm_hidden_states[layer_idx] = init_hidden_state + lstm_hidden_states[layer_idx]
                # 传递到LSTM单元
                states = (lstm_hidden_states[layer_idx], lstm_cell_states[layer_idx])
                (lstm_hidden_states[layer_idx], lstm_cell_states[layer_idx]), output = self.lstm_layers[layer_idx](layer_input, states)
            # 收集最后一层的输出
            last_output = lstm_hidden_states[-1].reshape(B, N, self.lstm_hidden_dim)
            outputs.append(last_output)

        # 使用最后的隐藏状态进行多步预测
        final_hidden = outputs[-1]  # [B, N, hidden_dim]
        # 多步预测
        predictions = self.prediction_head(final_hidden)  # [B, N, output_steps * output_dim]
        predictions = predictions.reshape(B, N, self.output_steps, self.output_dim)

        return predictions


