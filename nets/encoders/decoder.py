from entmax.activations import Entmax15
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
import math
from typing import NamedTuple
from utils.tensor_functions import compute_in_batches

from nets.graph_encoder import GraphAttentionEncoder
from torch.nn import DataParallel
from utils.beam_search import CachedLookup
from utils.functions import sample_many
from entmax import entmax15, Entmax15Loss
from nets.encoders.kool_encoder import AttentionEncoder

from typing import Optional



class kool_decoder (nn.Module):
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