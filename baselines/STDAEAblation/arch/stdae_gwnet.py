import torch
from torch import nn

from .stdae import STDAE
from .graphwavenet import GraphWaveNet

class STDAEGWNET(nn.Module):

    def __init__(self, dataset_name, pre_trained_tae_path, pre_trained_sae_path, stdae_args, gwnet_args,
                 short_term_len, is_use_tae = True, is_use_sae = True):
        super().__init__()
        self.dataset_name = dataset_name
        self.pre_trained_tae_path = pre_trained_tae_path
        self.pre_trained_sae_path = pre_trained_sae_path
        self.is_use_tae = is_use_tae
        self.is_use_sae = is_use_sae

        # iniitalize
        self.tae = STDAE(**stdae_args)
        self.sae = STDAE(**stdae_args)

        self.backend = GraphWaveNet(**gwnet_args)

        self.short_term_len = short_term_len

        # load pre-trained model
        if is_use_tae or is_use_sae:
            # load pre-trained model
            self.load_pre_trained_model()

    def load_pre_trained_model(self):
        """Load pre-trained model"""

        # load parameters
        checkpoint_dict = torch.load(self.pre_trained_tae_path)
        self.tae.load_state_dict(checkpoint_dict["model_state_dict"])

        checkpoint_dict = torch.load(self.pre_trained_sae_path)
        self.sae.load_state_dict(checkpoint_dict["model_state_dict"])

        # freeze parameters
        for param in self.tae.parameters():
            param.requires_grad = False
        for param in self.sae.parameters():
            param.requires_grad = False

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int,
                **kwargs) -> torch.Tensor:
        """Feed forward of STDAE-LSTM.

        Args:
            history_data (torch.Tensor): Short-term historical data. shape: [B, L, N, D]
            long_history_data (torch.Tensor): Long-term historical data. shape: [B, L * P, N, C = 4]

        Returns:
            torch.Tensor: prediction with shape [B, N, L].
        """
        # reshape
        long_history_data = history_data[:, :, :, :4]  # [B, L, N, 1]

        short_term_history = history_data[:, -self.short_term_len:, :, :]

        batch_size, _, num_nodes, _ = history_data.shape

        hidden_states_t = None
        hidden_states_s = None

        # enhance
        out_len = 1

        if self.is_use_tae:
            hidden_states_t = self.tae(long_history_data)
            hidden_states_t = hidden_states_t[:, :, -out_len, :]

        if self.is_use_sae:
            hidden_states_s = self.sae(long_history_data)
            hidden_states_s = hidden_states_s[:, :, -out_len, :]

        y_hat = self.backend(short_term_history, hidden_states_t=hidden_states_t, hidden_states_s=hidden_states_s)
        return y_hat


