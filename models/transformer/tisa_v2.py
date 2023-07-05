# Copyright 2023 Ulme Wennberg, Inc. All Rights Reserved.

import torch
from torch import nn
from typing import List
import math

class TisaV2(nn.Module):
    def __init__(self, 
                num_attention_heads,
                tisa_num_kernels,
                tisa_dropout_prob,
                num_position_agnostic_heads,
                max_field_view,
                min_field_view,
                p_eps = 1e-8):
                
        super().__init__()

        self.num_attention_heads = num_attention_heads
        self.num_kernels = tisa_num_kernels
        self.tisa_dropout_prob = tisa_dropout_prob
        self.num_position_agnostic_heads = num_position_agnostic_heads
        self.max_field_view = max_field_view
        self.min_field_view = min_field_view
        self.p_eps = p_eps

        self.eps = 1e-8

        self.offsets = nn.Parameter(
            torch.zeros(1, self.num_kernels, self.num_attention_heads, 1, 1)
        )
        self.amplitudes = nn.Parameter(
            torch.zeros(1, self.num_kernels, self.num_attention_heads, 1, 1)
        )
        self.sharpness = nn.Parameter(
            torch.zeros(1, self.num_kernels, self.num_attention_heads, 1, 1)
        )

        self.bias = nn.Parameter(torch.zeros(1, self.num_attention_heads, 1, 1))

        self.dropout = nn.Dropout(self.tisa_dropout_prob)

        self.num_position_agnostic_heads = self.num_position_agnostic_heads
        self.num_position_aware_heads = (
            self.num_attention_heads - self.num_position_agnostic_heads
        )
        self.position_agnostic_heads = torch.arange(
            self.num_attention_heads - self.num_position_agnostic_heads,
            self.num_attention_heads,
        )

        assert 0 < self.p_eps < 1

        """
        exp ( - field_view * m) = p_eps
        - log(p_eps) = field_view * m
        m = - log(p_eps) / field_view
        """

        self.one_side_min_field_view = self.min_field_view / 2
        self.one_side_max_field_view = self.max_field_view / 2

        self.first_slope = -math.log(self.p_eps) / self.one_side_min_field_view
        self.last_slope = -math.log(self.p_eps) / self.one_side_max_field_view
        self.slopes = nn.Parameter(
            (
                self.first_slope
                * (self.last_slope / self.first_slope)
                ** (
                    torch.arange(self.num_attention_heads)
                    / (self.num_position_aware_heads - 1)
                )
            ).reshape(self.num_attention_heads, 1, 1),
            requires_grad=False,
        )

        # Disable exponential decay for position agnostic heads
        self.slopes[self.position_agnostic_heads, 0, 0] = 0.0

    def create_relative_offsets(self, seq_len):
        """Creates offsets for all the relative distances between
        -seq_len + 1 to seq_len - 1."""
        return (
            torch.arange(-seq_len, seq_len + 1, device=self.offsets.device)
            .unsqueeze(0)
            .unsqueeze(0)
        )

    def forward(self, dim=-1, skip_apply_dropout=False):
        """Computes the translation-invariant positional contribution to the
        attention matrix in the self-attention module of transformer models."""
        
        indices_from = torch.arange(dim, device=self.offsets.device).unsqueeze(0).unsqueeze(0)
        indices_to = indices_from

        if not self.num_kernels:
            return torch.zeros(
                (self.num_attention_heads, indices_from.shape[-1], indices_to.shape[-1])
            )
        params = (indices_to.unsqueeze(-1) - indices_from.unsqueeze(-2)).unsqueeze(-4)
        exponential_decay_arguments = -params.abs() * self.slopes
        params = params.unsqueeze(-4) - self.offsets
        params = params / self.sharpness
        params = self.amplitudes.abs() * torch.sigmoid(self.amplitudes.sign() * params)

        if self.training and not skip_apply_dropout:
            params = self.dropout(params)
        params = params.sum(dim=-4)

        # Make final dimensions completely position agnostic
        params[:, :, self.position_agnostic_heads] = 0.0

        params += self.eps + self.bias.abs()
        params = torch.log(params)
        params = params + exponential_decay_arguments
        return params.squeeze(0)

    def _init_weights(self):
        """Initialize the weights"""
        torch.nn.init.normal_(self.offsets, mean=0.0, std=15.0)
        torch.nn.init.normal_(self.amplitudes, mean=0.0, std=0.01)

        self.sharpness.data.fill_(5.0)
        self.bias.data.fill_(1.0)
