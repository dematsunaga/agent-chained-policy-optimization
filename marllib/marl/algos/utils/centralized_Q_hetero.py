# MIT License
# Copyright (c) 2023 Replicable-MARL
#
# before_learn_on_batch implementations.
#

import numpy as np
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_ops import convert_to_torch_tensor
from ray.rllib.utils.numpy import convert_to_numpy

from marllib.marl.algos.utils.centralized_Q import before_learn_on_batch, get_dim

torch, nn = try_import_torch()


def _add_target_noise(multi_agent_batch, policies):
    """Add TD3 target smoothing noise, scaled by 1/sqrt(N)."""
    sorted_pids = sorted(policies.keys(), key=lambda pid: int(pid.split("_")[-1]))
    n_agents = len(sorted_pids)
    noise_scale_factor = 1.0 / np.sqrt(n_agents)
    for pid in sorted_pids:
        policy = policies[pid]
        custom_config = policy.config["model"]["custom_model_config"]
        target_noise = custom_config.get("target_noise", 0.2) * noise_scale_factor
        target_noise_clip = custom_config.get("target_noise_clip", 0.5) * noise_scale_factor
        act_low = policy.action_space.low
        act_high = policy.action_space.high
        act_scale = (act_high - act_low) / 2.0

        opp_actions = multi_agent_batch.policy_batches[pid]["next_opponent_actions"]
        noise = np.random.randn(*opp_actions.shape).astype(np.float32)
        noise = noise * target_noise * act_scale
        noise = np.clip(noise, -target_noise_clip * act_scale,
                        target_noise_clip * act_scale)
        noisy = np.clip(opp_actions + noise, act_low, act_high)

        multi_agent_batch.policy_batches[pid]["next_opponent_actions"] = noisy

