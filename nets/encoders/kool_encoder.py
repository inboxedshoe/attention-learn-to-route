from typing import Optional
import torch
import torch.nn as nn
from torch import LongTensor
import torch_geometric.nn as gnn

from nets.encoders.attn_network import AttentionBlock
from nets.encoders.base_encoder import BaseEncoder

class AttentionEncoder(BaseEncoder):
    """Attention encoder model for node embeddings."""

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: int = 128,
                 ff_hidden_dim: int = 512,
                 n_heads: int = 8,
                 add_bias: bool = False,
                 dropout: float = 0,
                 attention_activation: str = "softmax",
                 skip: bool = False,
                 norm_type: Optional[str] = None,
                 attention_neighborhood: int = 0,
                 num_layers: int = 2,
                 **kwargs):
        """

        Args:
            input_dim: dimension of node features
            output_dim: embedding dimension to output, 0 if no reembed is required
            hidden_dim: dimension of hidden layers in the MHA
            ff_hidden_dim: dimension of hidden layers in the FF network
                            (0 if a single FF layer is desired instead)
            attention_activation: activation function to use: softmax(default),
                                entmax, or sparsemax
            skip: flag to use skip (residual) connections
            add_bias: add bias to layers in the MHA, care if use with normalization
            num_layers: number of attention blocks required where one block is an MHA layer,
                        and FF network/layer
            norm_type: type of norm to use
            dropout: dropout to be used in the MHA network
            n_heads: number of heads in the MHA network
            attention_neighborhood: size of node neighborhood to consider for node attention
        """
        super(AttentionEncoder, self).__init__(input_dim, output_dim)

        self.attention_activation = {
            'softmax': 0,
            'entmax': 1,
            'sparse': 2
        }.get(attention_activation, "softmax")
        ##TODO: implement attention neighborhood
        self.attention_neighborhood = attention_neighborhood
        self.n_heads = n_heads

        self.skip = skip
        self.norm_type = norm_type
        self.dropout = dropout
        self.add_bias = add_bias
        self.ff_hidden_dim = ff_hidden_dim

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.input_proj = None
        self.output_proj = None
        self.num_layers = num_layers
        self.attention_blocks = None

        self.create_layers(**kwargs)
        self.reset_parameters()

    def reset_parameters(self):
        self.input_proj.reset_parameters()
        if self.output_dim != 0:
            self.output_proj.reset_parameters()

    @staticmethod
    def _reset_module_list(mod_list):
        """Resets all eligible modules in provided list."""
        if mod_list is not None:
            for m in mod_list:
                m.reset_parameters()

    def create_layers(self, **kwargs):
        """Create the specified model layers."""
        # conv = getattr(gnn, self.conv_type)
        # input projection layer
        self.input_proj = nn.Linear(self.input_dim, self.hidden_dim)

        def AN():
            # creates an attention module with specified parameters
            # all modules are initialized globally with the call to
            # reset_parameters()
            # it should be noted that every head will have embed_dim/n_heads
            # dimensions
            return AttentionBlock(
                embed_dim=self.hidden_dim,
                n_heads=self.n_heads,
                ff_hidden_dim=self.ff_hidden_dim,
                norm_type=self.norm_type,
                dropout=self.dropout,
                add_bias=self.add_bias,
                skip=self.skip,
                attention_activation=self.attention_activation,
                **kwargs
            )

        self.attention_blocks = nn.ModuleList()
        for _ in range(self.num_layers):
            self.attention_blocks.append(AN())

        if self.output_dim > 0:
            self.output_proj = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, embedding, attn_mask=None, key_padding_mask=None):

        #project input onto embedding space
        #h = self.input_proj(features.view(-1, features.size(-1))).view(*features.size()[:2], -1)
        h = self.input_proj(embedding)

        # pass through each of the attention blocks
        for block in self.attention_blocks:
            h, weights = block(h, attn_mask=attn_mask, key_padding_mask=key_padding_mask)

        #project to output dimensions
        if self.output_dim > 0:
            h = self.output_proj(h)

        return h, weights
