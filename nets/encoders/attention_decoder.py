from typing import Union, Tuple, Optional, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import math


class AttnDecoder(BaseDecoder):
    """
    Attention decoder model based on

        Kool, W., Van Hoof, H., & Welling, M. (2018).
        Attention, learn to solve routing problems!.
        arXiv preprint arXiv:1803.08475.

    """
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: int = 128,
                 num_heads: int = 2,
                 clip_tanh: Union[int, float] = 10.,
                 bias: bool = False,
                 **kwargs):
        super(AttnDecoder, self).__init__(input_dim, output_dim, hidden_dim)

        self.num_heads = num_heads
        self.clip_tanh = clip_tanh
        self.bias = bias

        head_dim = hidden_dim // num_heads
        assert head_dim * num_heads == hidden_dim, "<hidden_dim> must be divisible by <num_heads>!"
        self.head_dim = head_dim

        self.dk_get_locp = torch.tensor(hidden_dim, dtype=torch.float32)
        self.dk_mha_decoder = torch.tensor(head_dim, dtype=torch.float32)

        self.wq_context = nn.Linear(self.input_dim, self.hidden_dim)
        self.wq_step_context = nn.Linear(self.input_dim, self.hidden_dim)

        self.wk_context = nn.Linear(self.input_dim, self.hidden_dim)
        self.wk_tanh = nn.Linear(self.input_dim, self.hidden_dim)

        self.wv = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.w_out = nn.Linear(self.hidden_dim, self.output_dim)

        # scaling factors for scaled product attention
        self.u_norm = lambda x: x * (float(head_dim) ** -0.5)
        self.nc_norm = lambda x: x * (float(hidden_dim) ** -0.5)

    def set_decode_type(self, decode_type):
        self.decode_type = decode_type


    def _make_heads(self, x: Tensor):
        """Makes attention heads for the provided glimpses (BS, N, emb_dim)"""
        return (
            x.contiguous()
            .view(x.size(0), x.size(1), self.num_heads, -1)  # emb_dim --> head_dim * n_heads
            .permute(0, 2, 1, 3)  # (BS, n_heads, N, head_dim)
        )

    def _select_node(self, probs, mask):

        assert (probs == probs).all(), "Probs should not contain any nans"

        if self.decode_type == "greedy":
            _, selected = probs.max(1)

        elif self.decode_type == "sampling":
            selected = probs.multinomial(1).squeeze(1)

            # Check if sampling went OK, can go wrong due to bug on GPU
            # See https://discuss.pytorch.org/t/bad-behavior-of-multinomial-function/10232
            while mask.gather(1, selected.unsqueeze(-1)).data.any():
                print('Sampled bad values, resampling!')
                selected = probs.multinomial(1).squeeze(1)

        else:
            assert False, "Unknown decode type"
        return selected

    def get_step_context(self, state, embeddings):
        """Takes a state and graph embeddings,
           Returns a part [h_N, D] of context vector [h_c, h_N, D],
           that is related to RL Agent last step.
        """
        # index of previous node
        current_node = state.get_current_node()
        batch_size, num_steps = current_node.size()

        # from embeddings=(batch_size, n_nodes, input_dim) select embeddings of previous nodes
        #TODO: check if replace current_node with prev_a (check dimension values)
        cur_embedded_node = torch.gather(embeddings, 1,
                            current_node.contiguous()
                                .view(batch_size, num_steps, 1)
                                .expand(batch_size, num_steps, embeddings.size(-1))
                        ).view(batch_size, num_steps, embeddings.size(-1))

        # add remaining capacity
        step_context = torch.cat((cur_embedded_node, self.problem.VEHICLE_CAPACITY - state.used_capacity[:, :, None]), -1)

        return step_context  # (batch_size, 1, input_dim + 1)


    def decoder_mha(self, Q, K, V, mask=None):
        """ Computes Multi-Head Attention part of decoder
        Basically, its a part of MHA sublayer, but we cant construct a layer since Q changes in a decoding loop.

        Args:
            mask: a mask for visited nodes,
                has shape (batch_size, seq_len_q, seq_len_k), seq_len_q = 1 for context vector attention in decoder
            Q: query (context vector for decoder)
                    has shape (..., seq_len_q, head_depth) with seq_len_q = 1 for context_vector attention in decoder
            K, V: key, value (projections of nodes embeddings)
                have shape (..., seq_len_k, head_depth), (..., seq_len_v, head_depth),
                                                                with seq_len_k = seq_len_v = n_nodes for decoder
        """

        # (batch_size, num_heads, seq_len_q, seq_len_k)
        compatibility = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(self.dk_mha_decoder)

        if mask is not None:
            compatibility[mask[None, :, :, None, :].expand_as(compatibility)] = -math.inf

        # apply softmax
        compatibility = F.softmax(compatibility, axis=-1)  # (batch_size, num_heads, seq_len_q, seq_len_k)

        #get attention
        attention = torch.matmul(compatibility, V)  # (batch_size, num_heads, seq_len_q, head_depth)

        output = self.w_out(attention)  # (batch_size, seq_len_q, output_dim), seq_len_q = 1 for context att in decoder

        return output

    def get_log_p(self, Q, K, mask=None):
        """Single-Head attention sublayer in decoder,
        computes log-probabilities for node selection.

        Args:
            mask: mask for nodes
            Q: query (output of mha layer)
                    has shape (batch_size, seq_len_q, output_dim), seq_len_q = 1 for context attention in decoder
            K: key (projection of node embeddings)
                    has shape  (batch_size, seq_len_k, output_dim), seq_len_k = n_nodes for decoder
        """

        compatibility = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(self.dk_get_loc_p)

        #tanh clipping
        if self.clip_tanh:
            x = torch.tanh(compatibility) * self.clip_tanh

        if mask is not None:

            # we dont need to reshape mask like we did in multi-head version:
            # (batch_size, seq_len_q, seq_len_k) --> (batch_size, num_heads, seq_len_q, seq_len_k)
            # since we dont have multiple heads

            # compatibility = tf.where(
            #                     tf.broadcast_to(mask, compatibility.shape), tf.ones_like(compatibility) * (-np.inf),
            #                     compatibility
            #                      )

            if mask is not None:
                compatibility[mask[None, :, :, None, :].expand_as(compatibility)] = -math.inf

        log_p = torch.log_softmax(compatibility, dim=-1)  # (batch_size, seq_len_q, seq_len_k)

        return log_p

    def _calc_log_likelihood(self, _log_p, a, mask=None):

        # Get log_p corresponding to selected actions
        log_p = _log_p.gather(2, a.unsqueeze(-1)).squeeze(-1)

        # Optional: mask out actions irrelevant to objective so they do not get reinforced
        if mask is not None:
            log_p[mask] = 0

        assert (log_p > -1000).data.all(), "Logprobs should not be -inf, check sampling procedure!"

        # Calculate log_likelihood
        return log_p.sum(1)

    def get_projections(self, embeddings, context_vectors):

        # we compute some projections (common for each policy step) before decoding loop for efficiency
        K = self.wk(embeddings)  # (batch_size, n_nodes, output_dim)
        K_tanh = self.wk_tanh(embeddings)  # (batch_size, n_nodes, output_dim)
        V = self.wv(embeddings)  # (batch_size, n_nodes, output_dim)
        Q_context = self.wq_context(context_vectors[:, None, :])  # (batch_size, 1, output_dim)

        # we dont need to split K_tanh since there is only 1 head; Q will be split in decoding loop
        K = self._make_heads(K)  # (batch_size, num_heads, n_nodes, head_depth)
        V = self._make_heads(V)  # (batch_size, num_heads, n_nodes, head_depth)

        return K_tanh, Q_context, K, V

    def forward(self, inputs, return_pi=False):

        embeddings, mean_graph_emb = self.embedder(inputs)

        self.batch_size = embeddings.shape[0]

        outputs = []
        sequences = []

        state = self.problem(inputs)

        K_tanh, Q_context, K, V = self.get_projections(embeddings, mean_graph_emb)

        # Perform decoding steps
        i = 0
        inner_i = 0

        while not state.all_finished():

            if i > 0:
                state.i = torch.zeros(1, dtype=torch.int64)
                att_mask, cur_num_nodes = state.get_att_mask()
                embeddings, context_vectors = self.embedder(inputs, att_mask, cur_num_nodes)
                K_tanh, Q_context, K, V = self.get_projections(embeddings, context_vectors)

            inner_i = 0
            while not state.partial_finished():

                step_context = self.get_step_context(state, embeddings)  # (batch_size, 1), (batch_size, 1, input_dim + 1)
                Q_step_context = self.wq_step_context(step_context)  # (batch_size, 1, output_dim)
                Q = Q_context + Q_step_context

                # split heads for Q
                Q = self._make_heads(Q)  # (batch_size, num_heads, 1, head_depth)

                # get current mask
                mask = state.get_mask()  # (batch_size, 1, n_nodes) with True/False indicating where agent can go

                # compute MHA decoder vectors for current mask
                mha = self.decoder_mha(Q, K, V, mask)  # (batch_size, 1, output_dim)

                # compute probabilities
                log_p = self.get_log_p(mha, K_tanh, mask)  # (batch_size, 1, n_nodes)

                # next step is to select node
                selected = self._select_node(log_p, mask=None)

                state.step(selected)

                outputs.append(log_p[:, 0, :])
                sequences.append(selected)

                inner_i += 1

            i += 1

        _log_p, pi = torch.stack(outputs, 1), torch.stack(sequences, 1)

        cost = self.problem.get_costs(inputs, pi)

        ll = self.get_log_likelihood(_log_p, pi)

        if return_pi:
            return cost, ll, pi

        return cost, ll



#
# ============= #
# ### TEST #### #
# ============= #
def _test(
        bs: int = 5,
        n: int = 10,
        cuda=False,
        seed=1
):
    import torch
    device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    I = 32
    O = 10
    nf = torch.randn(bs, n, I).to(device)
    obs = RPObs(node_features=nf, current_tour=None, best_tour=None)
    emb = RPEmb(
        node_feature_emb=nf,
        aggregated_emb=torch.randn(bs, I).to(device),
        option_set_emb=nf
    )

    d = AttnDecoder(I, O).to(device)
    logits, _ = d(obs, emb)
    assert logits.size() == torch.empty((bs, O)).size()

