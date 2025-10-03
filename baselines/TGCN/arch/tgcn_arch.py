import argparse
import torch
import torch.nn as nn


def calculate_laplacian_with_self_loop(matrix):
    """
    Calculate the normalized Laplacian matrix with self-loop.

    Parameters:
    matrix: Adjacency matrix

    Returns:
    normalized_laplacian: Normalized Laplacian matrix
    """
    # Add self-loop
    matrix = matrix + torch.eye(matrix.size(0))
    # Calculate row sum
    row_sum = matrix.sum(1)
    # Calculate D^(-1/2)
    d_inv_sqrt = torch.pow(row_sum, -0.5).flatten()
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    # Calculate normalized Laplacian
    normalized_laplacian = (
        matrix.matmul(d_mat_inv_sqrt).transpose(0, 1).matmul(d_mat_inv_sqrt)
    )
    return normalized_laplacian


class TGCNGraphConvolution(nn.Module):
    """
    Temporal Graph Convolutional Layer
    """

    def __init__(self, adj, num_gru_units: int, output_dim: int, bias: float = 0.0):
        super(TGCNGraphConvolution, self).__init__()
        self._num_gru_units = num_gru_units
        self._output_dim = output_dim
        self._bias_init_value = bias
        self.register_buffer(
            "laplacian", calculate_laplacian_with_self_loop(torch.FloatTensor(adj))
        )
        self.weights = nn.Parameter(
            torch.FloatTensor(self._num_gru_units + 1, self._output_dim)
        )
        self.biases = nn.Parameter(torch.FloatTensor(self._output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights)
        nn.init.constant_(self.biases, self._bias_init_value)

    def forward(self, inputs, hidden_state):
        batch_size, num_nodes = inputs.shape
        # inputs (batch_size, num_nodes) -> (batch_size, num_nodes, 1)
        inputs = inputs.reshape((batch_size, num_nodes, 1))
        # hidden_state (batch_size, num_nodes, num_gru_units)
        hidden_state = hidden_state.reshape(
            (batch_size, num_nodes, self._num_gru_units)
        )
        # [x, h] (batch_size, num_nodes, num_gru_units + 1)
        concatenation = torch.cat((inputs, hidden_state), dim=2)
        # [x, h] (num_nodes, num_gru_units + 1, batch_size)
        concatenation = concatenation.transpose(0, 1).transpose(1, 2)
        # [x, h] (num_nodes, (num_gru_units + 1) * batch_size)
        concatenation = concatenation.reshape(
            (num_nodes, (self._num_gru_units + 1) * batch_size)
        )
        # A[x, h] (num_nodes, (num_gru_units + 1) * batch_size)
        a_times_concat = self.laplacian @ concatenation
        # A[x, h] (num_nodes, num_gru_units + 1, batch_size)
        a_times_concat = a_times_concat.reshape(
            (num_nodes, self._num_gru_units + 1, batch_size)
        )
        # A[x, h] (batch_size, num_nodes, num_gru_units + 1)
        a_times_concat = a_times_concat.transpose(0, 2).transpose(1, 2)
        # A[x, h] (batch_size * num_nodes, num_gru_units + 1)
        a_times_concat = a_times_concat.reshape(
            (batch_size * num_nodes, self._num_gru_units + 1)
        )
        # A[x, h]W + b (batch_size * num_nodes, output_dim)
        outputs = a_times_concat @ self.weights + self.biases
        # A[x, h]W + b (batch_size, num_nodes, output_dim)
        outputs = outputs.reshape((batch_size, num_nodes, self._output_dim))
        # A[x, h]W + b (batch_size, num_nodes * output_dim)
        outputs = outputs.reshape((batch_size, num_nodes * self._output_dim))
        return outputs


class TGCNCell(nn.Module):
    """
    Temporal Graph Convolutional Network Cell
    """

    def __init__(self, adj, input_dim: int, hidden_dim: int):
        super(TGCNCell, self).__init__()
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self.register_buffer("adj", torch.FloatTensor(adj))
        self.graph_conv1 = TGCNGraphConvolution(
            self.adj, self._hidden_dim, self._hidden_dim * 2, bias=1.0
        )
        self.graph_conv2 = TGCNGraphConvolution(
            self.adj, self._hidden_dim, self._hidden_dim
        )

    def forward(self, inputs, hidden_state):
        # [r, u] = sigmoid(A[x, h]W + b)
        # [r, u] (batch_size, num_nodes * (2 * num_gru_units))
        concatenation = torch.sigmoid(self.graph_conv1(inputs, hidden_state))
        # r (batch_size, num_nodes, num_gru_units)
        # u (batch_size, num_nodes, num_gru_units)
        r, u = torch.chunk(concatenation, chunks=2, dim=1)
        # c = tanh(A[x, (r * h)W + b])
        # c (batch_size, num_nodes * num_gru_units)
        c = torch.tanh(self.graph_conv2(inputs, r * hidden_state))
        # h := u * h + (1 - u) * c
        # h (batch_size, num_nodes * num_gru_units)
        new_hidden_state = u * hidden_state + (1.0 - u) * c
        return new_hidden_state, new_hidden_state


class STGCN_TGCN(nn.Module):
    """
    Spatio-Temporal Graph Convolutional Network using TGCN
    Designed for spatio-temporal prediction tasks
    """

    def __init__(self, gso, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int = 1, dropout: float = 0.0):
        """
        Initialize STGCN-TGCN model

        Args:
            gso: Graph shift operator (adjacency matrix)
            input_dim: Input feature dimension (C in [B, L, N, C])
            hidden_dim: Hidden dimension for TGCN
            output_dim: Output feature dimension
            num_layers: Number of TGCN layers
            dropout: Dropout rate
        """
        super(STGCN_TGCN, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_nodes = gso.shape[0]

        # Register adjacency matrix as buffer
        self.register_buffer("gso", torch.FloatTensor(gso))

        # Input projection layer to handle multi-dimensional features
        if input_dim > 1:
            self.input_proj = nn.Linear(input_dim, 1)
        else:
            self.input_proj = nn.Identity()

        # TGCN layers
        self.tgcn_layers = nn.ModuleList()
        for i in range(num_layers):
            self.tgcn_layers.append(
                TGCNCell(gso, self.num_nodes, hidden_dim)
            )

        # Output projection layer
        self.output_proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor = None,
                batch_seen: int = None, epoch: int = None, train: bool = True, **kwargs) -> torch.Tensor:
        """
        Forward function of STGCN-TGCN.

        Args:
            history_data (torch.Tensor): historical data with shape [B, L, N, C]
            future_data (torch.Tensor): future data (not used in this implementation)
            batch_seen (int): number of batches seen (for training statistics)
            epoch (int): current epoch (for training statistics)
            train (bool): training mode flag

        Returns:
            torch.Tensor: prediction with shape [B, L, N, C]
        """
        B, L, N, C = history_data.shape

        # Input projection: [B, L, N, C] -> [B, L, N, 1] if C > 1
        if C > 1:
            x = history_data.view(B * L * N, C)
            x = self.input_proj(x)  # [B*L*N, 1]
            x = x.view(B, L, N, 1).squeeze(-1)  # [B, L, N]
        else:
            x = history_data.squeeze(-1)  # [B, L, N]

        # Initialize hidden states for all layers
        hidden_states = []
        for _ in range(self.num_layers):
            hidden_state = torch.zeros(B, N * self.hidden_dim,
                                       device=x.device, dtype=x.dtype)
            hidden_states.append(hidden_state)

        # Process sequence step by step
        outputs = []
        for t in range(L):
            layer_input = x[:, t, :]  # [B, N]

            # Pass through TGCN layers
            for layer_idx, tgcn_layer in enumerate(self.tgcn_layers):
                layer_output, hidden_states[layer_idx] = tgcn_layer(
                    layer_input, hidden_states[layer_idx]
                )
                # Reshape for next layer: [B, N, hidden_dim]
                layer_input = layer_output.view(B, N, self.hidden_dim)
                # Apply layer normalization
                layer_input = self.layer_norm(layer_input)
                # Reshape back for graph convolution: [B, N] (only for first layer)
                if layer_idx < len(self.tgcn_layers) - 1:
                    # For intermediate layers, we need to aggregate the hidden dimensions
                    layer_input = layer_input.mean(dim=-1)  # [B, N]

            # Final layer output: [B, N, hidden_dim]
            step_output = layer_output.view(B, N, self.hidden_dim)
            outputs.append(step_output)

        # Stack outputs: [B, L, N, hidden_dim]
        sequence_output = torch.stack(outputs, dim=1)

        # Apply output projection: [B, L, N, hidden_dim] -> [B, L, N, output_dim]
        output = self.output_proj(sequence_output)

        return output

    @staticmethod
    def add_model_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--hidden_dim", type=int, default=64)
        parser.add_argument("--num_layers", type=int, default=1)
        parser.add_argument("--dropout", type=float, default=0.0)
        return parser

    @property
    def hyperparameters(self):
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "num_layers": self.num_layers,
            "num_nodes": self.num_nodes
        }


class TGCN(nn.Module):
    """
    TGCN for spatio-temporal forecasting
    """

    def __init__(self, gso, input_dim: int, hidden_dim: int, output_dim: int,
                 seq_len: int, pred_len: int, num_layers: int = 1, dropout: float = 0.0):
        """
        Initialize Multi-step TGCN model

        Args:
            gso: Graph shift operator (adjacency matrix)
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension for TGCN
            output_dim: Output feature dimension
            seq_len: Input sequence length
            pred_len: Prediction sequence length
            num_layers: Number of TGCN layers
            dropout: Dropout rate
        """
        super(TGCN, self).__init__()

        self.seq_len = seq_len
        self.pred_len = pred_len

        # Encoder TGCN
        self.encoder = STGCN_TGCN(gso, input_dim, hidden_dim, hidden_dim,
                                  num_layers, dropout)

        # Decoder for multi-step prediction
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * seq_len, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim * pred_len)
        )

        self.num_nodes = gso.shape[0]
        self.output_dim = output_dim

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor = None,
                batch_seen: int = None, epoch: int = None, train: bool = True, **kwargs) -> torch.Tensor:
        """
        Forward function for multi-step prediction.

        Args:
            history_data (torch.Tensor): historical data with shape [B, L, N, C]

        Returns:
            torch.Tensor: prediction with shape [B, pred_len, N, C]
        """
        B, L, N, C = history_data.shape

        # Encode the input sequence
        encoded = self.encoder(history_data)  # [B, L, N, hidden_dim]

        # Flatten temporal and feature dimensions for each node
        encoded_flat = encoded.view(B, N, -1)  # [B, N, L * hidden_dim]

        # Decode to multi-step predictions
        decoded = self.decoder(encoded_flat)  # [B, N, output_dim * pred_len]

        # Reshape to final output format
        output = decoded.view(B, N, self.pred_len, self.output_dim)
        output = output.permute(0, 2, 1, 3)  # [B, pred_len, N, output_dim]

        return output



# Example of how to use the model
if __name__ == "__main__":
    # Model parameters
    num_nodes = 12
    input_dim = 2
    hidden_dim = 64
    output_dim = 2
    seq_len = 10
    pred_len = 5

