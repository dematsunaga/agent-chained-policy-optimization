# MIT License
# Copyright (c) 2023 Replicable-MARL

import torch
import torch.nn as nn


class BeliefNetwork(nn.Module):
    """Shared belief network for predecessor action prediction."""

    def __init__(self, obs_dim: int, n_actions: int, n_agents: int,
                 hidden_sizes=(64, 64)):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.n_agents = n_agents

        input_dim = obs_dim + 2 * n_agents
        layers = []
        prev_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, n_actions))
        self.mlp = nn.Sequential(*layers)

    def _make_one_hot(self, idx, B, device, dtype):
        if not isinstance(idx, torch.Tensor):
            oh = torch.zeros(B, self.n_agents, device=device, dtype=dtype)
            oh[:, int(idx)] = 1.0
            return oh
        return torch.nn.functional.one_hot(
            idx.long(), self.n_agents).to(dtype=dtype, device=device)

    def forward(self, obs: torch.Tensor, i, j) -> torch.Tensor:
        B = obs.shape[0]
        oh_i = self._make_one_hot(i, B, obs.device, obs.dtype)
        oh_j = self._make_one_hot(j, B, obs.device, obs.dtype)
        x = torch.cat([obs, oh_i, oh_j], dim=-1)
        return self.mlp(x)
