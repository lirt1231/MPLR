#!/usr/bin/env python
# -*-coding:utf-8 -*-
# @file    :   rule_miner.py
# @brief   :   Rule miner model.
# @author  :   Haotian Li
# @email   :   lcyxlihaotian@126.com

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class RuleMiner(nn.Module):
    def __init__(self,
                 rank: int,
                 num_steps: int,
                 num_entities: int,
                 num_embeddings: int,
                 num_operators: int,
                 query_embedding_dim: int = 128,
                 num_rnn_layers: int = 1,
                 hidden_size: int = 128
                 ) -> None:
        """
        Args:
            `rank`: rank of esimators
            `num_steps': number of RNN input time step
            `num_entities`: number of entities in KG
            `num_embeddings`: number of embedding vectors (number of relations)
            `num_operators`: number of operator matrices
            `query_embedding_dim`: dimension of query embedding vectors
            `num_rnn_layers`: number of RNN layers
            `hidden_size`: RNN hidden state size
        """
        super(RuleMiner, self).__init__()
        self.rank = rank
        self.num_entities = num_entities
        self.num_steps = num_steps
        self.num_operators = num_operators
        self.num_rnn_layers = num_rnn_layers
        self.num_embeddings = num_embeddings
        self.query_embedding_dim = query_embedding_dim
        self.hidden_size = hidden_size

        weights = self._random_uniform_unit(
            self.num_embeddings,
            self.query_embedding_dim
        )

        self.query_embedding = nn.Embedding(
            self.num_embeddings,
            self.query_embedding_dim,
            _weight=torch.from_numpy(weights).float()
        )
        self.rnns = nn.ModuleList([
            nn.LSTM(
                input_size=self.query_embedding_dim,
                hidden_size=self.hidden_size,
                num_layers=self.num_rnn_layers,
                bidirectional=True
            ) for _ in range(self.rank)
        ])

        self.W_0 = nn.parameter.Parameter(torch.tensor(
            np.random.randn(
                self.hidden_size * 2,
                self.num_operators + 1
            ),
            dtype=torch.float,
        ))
        self.b_0 = nn.parameter.Parameter(torch.tensor(
            np.zeros((1, self.num_operators + 1)),
            dtype=torch.float,
        ))

    def forward(self,
                queries: torch.LongTensor,
                heads: torch.LongTensor,
                tails: np.ndarray,
                adjacency_matrices: dict) -> torch.Tensor:
        """Forward calculation.

        Args:
            `queries`: query relations (batch_size, )
            `heads`: head entities (batch_size, )
            `tails`: `batch_size` lists of tails
            `adjacency_matrices`: adjacency matrices for each relation

        Returns:
            torch.Tensor(batch_size, num_entities).
        """
        device = "cuda" if heads.is_cuda else "cpu"

        query_embed = queries.view(1, -1)
        # (num_steps, batch_size).
        query_embed = torch.cat(
            [
                query_embed
                for _ in range(self.num_steps)
            ],
            dim=0
        )
        # (num_steps, batch_size, query_embedding_dim).
        query_embed = self.query_embedding(query_embed)

        # Do attention.
        self.attention_operator_list = []
        for rnn in self.rnns:
            # `rnn_output`: (num_steps, batch_size, hidden_size*2).
            rnn_output, (_, _) = rnn(query_embed)

            # (num_steps, batch_size, num_operators+1).
            trans_output = torch.matmul(rnn_output, self.W_0) + self.b_0
            # (num_steps, num_operators+1, batch_size).
            attn_output = F.softmax(trans_output, -1).transpose(-2, -1)

            self.attention_operator_list.append(attn_output.unsqueeze(-1))

        operators_matrices = {
            rel: torch.sparse.FloatTensor(
                torch.LongTensor(adjacency_matrices[rel][0]).t(),
                torch.FloatTensor(adjacency_matrices[rel][1]),
                adjacency_matrices[rel][2],
            ).to(device)
            for rel in adjacency_matrices.keys()
        }

        # A list of `rank` tensors,
        # each tensor: (batch_size, step, num_entities).
        memory_list = [
            F.one_hot(heads, self.num_entities)
             .float().unsqueeze(1).to(device)
            for _ in range(self.rank)
        ]

        # (batch_size, num_entities).
        logits = 0.0
        for r in range(self.rank):
            if self.training:
                # (num_operators+1, batch_size, 1).
                first_hop_attns = self.attention_operator_list[r][0]
                # (batch_size, T, t, num_entities)
                sub_memory_list = [
                    # (T, t, num_entities)
                    F.one_hot(torch.LongTensor(tails[bid]), self.num_entities)
                    .float().unsqueeze(1).to(device) * first_hop_attns[queries[bid], bid, 0]
                    for bid in range(len(tails))
                ]

            for t in range(self.num_steps):
                # (batch_size, num_entities).
                memory = memory_list[r][:, -1, :]
                # (num_operators+1, batch_size, 1).
                attn_ops = self.attention_operator_list[r][t]
                # (batch_size, num_entities).
                added_matrix_result = 0.0

                if self.training:
                    # (batch_size, T)
                    added_sub_memory_result = [0. for _ in range(len(tails))]

                for op in range(self.num_operators):
                    # `op_matrix`: (num_entities, num_entities).
                    # `op_attn`: (batch_size, 1).
                    op_matrix = operators_matrices[op]
                    op_attn = attn_ops[op]
                    # (batch_size, num_entities).
                    product = torch.matmul(op_matrix.t(), memory.t()).t()
                    # (batch_size, num_entities).
                    added_matrix_result += product * op_attn

                    if self.training and t > 0:
                        for bid, sub_memory in enumerate(sub_memory_list):
                            # (T, num_entities)
                            sub_memory = sub_memory[:, -1, :]
                            product = torch.matmul(op_matrix.t(), sub_memory.t()).t()
                            added_sub_memory_result[bid] += product * op_attn[bid]
                if self.training and t > 0:
                    for bid, sub_memory in enumerate(sub_memory_list):
                        # (T, num_entities)
                        sub_memory = sub_memory[:, -1, :]
                        # k=0 operator.
                        added_sub_memory_result[bid] += sub_memory * attn_ops[-1][bid]
                        # Transmit from head node.
                        T = tails[bid]
                        Tids = range(len(T))
                        tmp_attn = self.attention_operator_list[r][t, queries[bid], bid, 0]
                        added_sub_memory_result[bid][Tids, T] +=\
                            memory[bid, heads[bid]] * tmp_attn

                        sub_memory_list[bid] = torch.cat(
                            [sub_memory_list[bid],
                             added_sub_memory_result[bid].unsqueeze(1)],
                            dim=1
                        )

                added_matrix_result += memory * attn_ops[-1]
                # Each tensor: (batch_size, step, num_entities).
                memory_list[r] = torch.cat(
                    [memory_list[r],
                     added_matrix_result.unsqueeze(1)],
                    dim=1
                )
            # (batch_size, num_entities).
            last_memory = memory_list[r][:, -1, :]
            if self.training:
                for bid, T in enumerate(tails):
                    T = tails[bid]
                    Tids = range(len(T))
                    sub_memory = sub_memory_list[bid][Tids, -1, T]
                    last_memory[bid, T] -= sub_memory

            norm = torch.maximum(torch.tensor(1e-20).to(device),
                                 torch.sum(last_memory, dim=1, keepdim=True))
            last_memory /= norm
            logits += last_memory

        return logits

    def _random_uniform_unit(self,
                             rows: int,
                             cols: int
                             ) -> np.ndarray:
        bound = 6. / np.sqrt(cols)
        matrix = np.random.uniform(-bound, bound, (rows, cols))
        for row in range(matrix.shape[0]):
            matrix[row] = matrix[row] / np.linalg.norm(matrix[row])
        return matrix
