from typing import Union, Tuple, Optional, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import math

from nets.encoders.kool_encoder import AttentionEncoder
from nets.encoders.kool_encoder import DynamicAttentionEncoder
from problems.vrp.environment import AgentVRP

def set_decode_type(model, decode_type):
    model.set_decode_type(decode_type)


class AttentionModel(nn.Module):

    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 problem,
                 n_encode_layers=2,
                 tanh_clipping=10.,
                 mask_inner=True,
                 mask_logits=True,
                 normalization='batch',
                 n_heads=8,
                 checkpoint_encoder=False,
                 shrink_size=None,
                 attention_type="full",
                 attention_neighborhood=0,
                 encode_freq=0,
                 encoder_knn=0,
                 reencode_partial=False):
        super(AttentionModel, self).__init__()

        # MHA attributes
        self.embedding_dim = embedding_dim
        self.n_encode_layers = n_encode_layers
        self.decode_type = "greedy"

        # VRP attributes
        #self.problem = problem
        self.problem = AgentVRP
        self.num_heads = n_heads

        # encoder
        #TODO: try no batch, with tanh
        self.embedder = DynamicAttentionEncoder(
            input_dim=embedding_dim,
            output_dim=0,  # optional for final output FF layer
            n_heads=n_heads,
            attention_activation="softmax",
            num_layers=self.n_encode_layers,
            #norm_type=normalization,
            norm_type=None,
            dropout=0,
            skip=True,
            tanh_activation=True
        )

        # decoder

        self.hidden_dim = self.embedding_dim
        head_dim = hidden_dim // n_heads
        assert head_dim * n_heads == hidden_dim, "<hidden_dim> must be divisible by <num_heads>!"
        self.head_dim = head_dim

        self.dk_get_loc_p = torch.tensor(hidden_dim, dtype=torch.float32)
        self.dk_mha_decoder = torch.tensor(head_dim, dtype=torch.float32)

        self.clip_tanh = tanh_clipping

        self.wq_context = nn.Linear(self.embedding_dim, self.hidden_dim)
        self.wq_step_context = nn.Linear(self.embedding_dim+1, self.hidden_dim)

        self.wk = nn.Linear(self.embedding_dim, self.hidden_dim)
        self.wk_tanh = nn.Linear(self.embedding_dim, self.hidden_dim)

        self.wv = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.w_out = nn.Linear(self.hidden_dim, self.embedding_dim)

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
        #probs[probs.isnan()] = 0
        assert (probs == probs).all(), "Probs should not contain any nans"

        if self.decode_type == "greedy":
            _, selected = probs.max(2)

        elif self.decode_type == "sampling":
            selected = probs.squeeze(1).multinomial(1)

            # # Check if sampling went OK, can go wrong due to bug on GPU
            # # See https://discuss.pytorch.org/t/bad-behavior-of-multinomial-function/10232
            # while mask.gather(1, selected.unsqueeze(-1)).data.any():
            #     print('Sampled bad values, resampling!')
            #     selected = probs.squeeze(1).multinomial(1)

        else:
            assert False, "Unknown decode type"
        return selected.reshape(-1)

    def get_step_context(self, state, embeddings):
        """Takes a state and graph embeddings,
           Returns a part [h_N, D] of context vector [h_c, h_N, D],
           that is related to RL Agent last step.
        """
        # index of previous node
        current_node = state.prev_a
        batch_size, num_steps = current_node.size()

        # from embeddings=(batch_size, n_nodes, input_dim) select embeddings of previous nodes
        #TODO: check if replace current_node with prev_a (check dimension values)
        cur_embedded_node = torch.gather(embeddings, 1,
                            current_node.to(torch.int64).contiguous()
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
            compatibility[mask[:, None, :, :].expand_as(compatibility)] = -math.inf
            # compatibility = torch.where(mask[:, None, :, :].expand_as(compatibility),
            #                          torch.ones_like(compatibility) * (-np.inf),
            #                          compatibility)

        # apply softmax
        compatibility = F.softmax(compatibility, dim=-1)  # (batch_size, num_heads, seq_len_q, seq_len_k)
        if mask is not None:
            compatibility[mask[:, None, :, :].expand_as(compatibility)] = 0
        # get attention
        attention = torch.matmul(compatibility, V)  # (batch_size, num_heads, seq_len_q, head_depth)

        # fix attention size
        attention = attention.permute(0, 2, 1, 3)
        attention = attention.reshape(self.batch_size, -1, self.embedding_dim)

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

        # tanh clipping
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

             compatibility[mask] = -math.inf
            # compatibility = torch.where(mask,
            #                             torch.ones_like(compatibility) * (-np.inf),
            #                             compatibility)

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

        state = self.problem(inputs, device=inputs["demand"].get_device())
        embeddings, mean_graph_emb = self.embedder(inputs)

        self.batch_size = embeddings.shape[0]

        outputs = []
        sequences = []

        #state = self.problem.make_state(inputs)

        K_tanh, Q_context, K, V = self.get_projections(embeddings, mean_graph_emb)

        # Perform decoding steps
        i = 0
        inner_i = 0

        while not state.all_finished():

            if i > 0:
                state.i = torch.zeros(1, dtype=torch.int64, device=inputs["demand"].get_device())
                att_mask, cur_num_nodes = state.get_att_mask()

                #####################################TRIAL#################################################
                # mask = state.get_mask()  # (batch_size, 1, n_nodes) with True/False indicating where agent can go
                # att_mask = mask.repeat(8, mask.shape[2], 1)  # make (batch_size*n_heads, n_nodes, n_nodes)
                # att_mask[:, :, 0] = False  # depots always available
                #
                # cur_num_nodes = (mask == False).sum(2).int()
                ###########################################################################################
                att_mask = att_mask.repeat(8, 1, 1)  # make (batch_size*n_heads, n_nodes, n_nodes)
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
                selected = self._select_node(log_p, mask=mask)

                state.step(selected)

                outputs.append(log_p[:, 0, :])
                sequences.append(selected)

                inner_i += 1

            i += 1

        _log_p, pi = torch.stack(outputs, 1), torch.stack(sequences, 1)

        cost, _ = self.problem.get_costs(inputs, pi)

        ll = self._calc_log_likelihood(_log_p, pi)

        if return_pi:
            return cost, ll, pi

        return cost, ll, None



