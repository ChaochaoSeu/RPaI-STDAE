import torch
from torch import nn

from .mask_stdae import MaskSTDAE
from .graphwavenet import GraphWaveNet

class MaskSTDAEGWNET(nn.Module):

    def __init__(self, dataset_name, pre_trained_masktae_path, pre_trained_masksae_path, maskstdae_args, gwnet_args,
                 short_term_len, is_use_mask_tae = True, is_use_mask_sae = True, mask_dir_list = None, mask_time_list = None):
        super().__init__()
        self.dataset_name = dataset_name
        self.pre_trained_masktae_path = pre_trained_masktae_path
        self.pre_trained_masksae_path = pre_trained_masksae_path

        self.is_use_mask_tae = is_use_mask_tae
        self.is_use_mask_sae = is_use_mask_sae

        # iniitalize
        if is_use_mask_tae:
            self.masktae = MaskSTDAE(**maskstdae_args)
        if is_use_mask_sae:
            self.masksae = MaskSTDAE(**maskstdae_args)

        self.backend = GraphWaveNet(**gwnet_args)

        self.short_term_len = short_term_len

        if is_use_mask_tae or is_use_mask_sae:
            # load pre-trained model
            self.load_pre_trained_model()

        # mask dir
        self.mask_dir_list = mask_dir_list if mask_dir_list is not None else []

        self.is_dir_mask = False
        if self.mask_dir_list:
            self.is_dir_mask = True
            self.rules = []
            if "East" in self.mask_dir_list:
                self.rules.extend([
                    {"nodes": [0, 1, 2], "dims": [0, 1]},
                    {"nodes": [4, 6, 11], "dims": [2, 3]}
                ])
            if "West" in self.mask_dir_list:
                self.rules.extend([
                    {"nodes": [3, 4, 5], "dims": [0, 1]},
                    {"nodes": [1, 8, 9], "dims": [2, 3]}
                ])
            if "South" in self.mask_dir_list:
                self.rules.extend([
                    {"nodes": [6, 7, 8], "dims": [0, 1]},
                    {"nodes": [2, 3, 10], "dims": [2, 3]}
                ])
            if "North" in self.mask_dir_list:
                self.rules.extend([
                    {"nodes": [9, 10, 11], "dims": [0, 1]},
                    {"nodes": [0, 5, 7], "dims": [2, 3]}
                ])
        # mask timestep
        self.mask_time_list = mask_time_list if mask_time_list is not None else []
        self.is_time_mask = False
        if self.mask_time_list:
            self.is_time_mask = True

    def load_pre_trained_model(self):
        """Load pre-trained model"""

        if self.is_use_mask_tae:
            # load parameters
            checkpoint_dict = torch.load(self.pre_trained_masktae_path)
            self.masktae.load_state_dict(checkpoint_dict["model_state_dict"])

            # freeze parameters
            for param in self.masktae.parameters():
                param.requires_grad = False

        if self.is_use_mask_sae:
            checkpoint_dict = torch.load(self.pre_trained_masksae_path)
            self.masksae.load_state_dict(checkpoint_dict["model_state_dict"])

            for param in self.masksae.parameters():
                param.requires_grad = False

    def apply_mask(self, x, rules, mask_time_list):
        """
        Apply masking rules to the input tensor.

        Args:
            x: torch.Tensor, shape = [B, T, N, D]
            dir_mask_rules: list of dict, masking rules
                Example: [
                    {"nodes": [0, 1, 2], "dims": [0, 1]},
                    {"nodes": [5, 7, 12], "dims": [2, 3]}
                ]
            mask_time_list: list of dict, time-step masks
        Returns:
            Masked tensor (masked_x)
        """
        mask_dir = torch.ones_like(x)
        mask_time = torch.ones_like(x)
        if self.is_dir_mask:
            for rule in rules:
                nodes = rule.get("nodes", [])
                dims = rule.get("dims", [])
                if len(nodes) > 0 and len(dims) > 0:
                    # Convert to tensor
                    nodes_tensor = torch.tensor(nodes, device=x.device, dtype=torch.long)
                    dims_tensor = torch.tensor(dims, device=x.device, dtype=torch.long)
                    # Use broadcasting for indexing
                    # nodes_tensor[:, None] shape = [len(nodes), 1]
                    # dims_tensor[None, :] shape = [1, len(dims)]
                    mask_dir[:, :, nodes_tensor[:, None], dims_tensor[None, :]] = 0
        # Time-step masking
        if self.is_time_mask:
            mask_time[:, mask_time_list, :, 0:4] = 0

        return x * mask_dir * mask_time

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

        short_term_history = self.apply_mask(short_term_history, self.rules, self.mask_time_list)

        batch_size, _, num_nodes, _ = history_data.shape

        out_len = 1

        hidden_states_t = None

        hidden_states_s = None

        if self.is_use_mask_tae:
            hidden_states_t = self.masktae(long_history_data)
            hidden_states_t = hidden_states_t[:, :, -out_len, :]

        if self.is_use_mask_sae:
            hidden_states_s = self.masksae(long_history_data)
            hidden_states_s = hidden_states_s[:, :, -out_len, :]

        y_hat = self.backend(short_term_history, hidden_states_t=hidden_states_t, hidden_states_s=hidden_states_s)

        return y_hat


