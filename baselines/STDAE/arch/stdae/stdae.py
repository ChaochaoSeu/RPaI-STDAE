from inspect import Parameter
import torch
import torch.nn as nn

from .stpositionencoding import SpatioTemporalPositionalEncoding
from .patchembedding import PatchEmbedding
from .transformer_layers import TransformerLayers

# 定义STDAE模型
class STDAE(nn.Module):
    def __init__(self, patch_size, in_channel, embed_dim, num_heads, mlp_ratio,
                 dropout, encoder_depth, decoder_depth, spatial=False, mode="pre-train"):
        super().__init__()
        assert mode in ["pre-train", "forecasting"], "Error mode."
        self.patch_size = patch_size
        self.in_channel = in_channel
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.encoder_depth = encoder_depth
        self.mode = mode
        self.mlp_ratio = mlp_ratio
        self.spatial = spatial

        self.selected_feature = 0

        # norm layers
        self.encoder_norm = nn.LayerNorm(embed_dim)
        self.decoder_norm = nn.LayerNorm(embed_dim)

        # patchify & embedding
        self.patch_embedding = PatchEmbedding(patch_size, in_channel, embed_dim, norm_layer=None)
        # positional encoding
        self.positional_encoding = SpatioTemporalPositionalEncoding(embed_dim)
        # encoder
        self.encoder = TransformerLayers(embed_dim, encoder_depth, mlp_ratio, num_heads, dropout)
        # decoder
        self.decoder = TransformerLayers(embed_dim, decoder_depth, mlp_ratio, num_heads, dropout)

        # transform layer
        self.enc_2_dec_emb = nn.Linear(embed_dim, embed_dim, bias=True)
        # prediction layer
        self.output_layer = nn.Linear(embed_dim, patch_size)

    def encoding(self, long_term_history_data):

        batch_size, num_nodes, _, _ = long_term_history_data.shape
        # patchify and embed input
        patches = self.patch_embedding(long_term_history_data)     # B, N, d, P
        patches = patches.transpose(-1, -2)         # B, N, P, d
        # positional embedding
        patches, self.pos_mat = self.positional_encoding(patches) # B, N, P, d

        encoder_input = patches # B, N, P, d

        if self.spatial:
            encoder_input = encoder_input.transpose(-2, -3)  # B, P, N, d

        hidden_state = self.encoder(encoder_input)# s: B, P, N, d  /# t: B, N, P, d

        if self.spatial:
            hidden_state = hidden_state.transpose(-2, -3)# B, N, P, d

        hidden_state = self.encoder_norm(hidden_state).view(batch_size, num_nodes, -1, self.embed_dim)  # B, N, P, d
        return hidden_state

    def decoding(self, hidden_states):
        # encoder 2 decoder layer
        hidden_states = self.enc_2_dec_emb(hidden_states)  # B, N, P, d
        batch_size, num_nodes, num_time, _ = hidden_states.shape
        if self.spatial:
            hidden_states = hidden_states.transpose(-2, -3)  # B, P, N, d
            decoder_out = self.decoder(hidden_states)  # B, P, N, d
            decoder_out = self.decoder_norm(decoder_out) # B, P, N, d
            # prediction (reconstruction)
            reconstruction_full = self.output_layer(decoder_out)  # B, P, N, L
            reconstruction_full = reconstruction_full.transpose(-2, -3 )  # B, N, P, L

        else:
            decoder_out = self.decoder(hidden_states)  # B, N, P, d
            decoder_out = self.decoder_norm(decoder_out)  # B, N, P, d
            # prediction (reconstruction)
            reconstruction_full = self.output_layer(decoder_out)  # B, N, P, L
        return reconstruction_full

    def get_reconsturcted_tokens(self, reconstruction_full, future_data):

        batch_size, num_nodes, num_time, _ = reconstruction_full.shape
        reconstruction_masked_tokens = reconstruction_full.view(batch_size, num_nodes, -1).transpose(1, 2)  # B, P*L, N

        label_full = future_data.squeeze(-1)

        return reconstruction_masked_tokens, label_full



    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor = None, batch_seen: int = None,
                epoch: int = None, **kwargs) -> torch.Tensor:
        """
        参数:
        history_data: [B, T_long, N, C] - 原始输入时间序列
        future_data: [B, T_long, N, 1] - 转向交通流量时间序列

        返回:
        hidden_states: 时间、空间重构结果
        """
        # reshape
        history_data = history_data.permute(0, 2, 3, 1)     # B, N, C, L * P

        # feed forward
        if self.mode == "pre-train":
            # encoding
            hidden_states = self.encoding(history_data)
            # decoding
            reconstruction_full = self.decoding(hidden_states)
            # for subsequent loss computing
            batch_size, num_nodes, num_time, _ = reconstruction_full.shape
            reconstruction_tokens = reconstruction_full.reshape(batch_size, num_nodes, -1).transpose(1, 2)  # B, P*L, N
            reconstruction = reconstruction_tokens.unsqueeze(-1)
            return reconstruction

        else:
            hidden_states = self.encoding(history_data)
            return hidden_states


