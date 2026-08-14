# MIT License

# Copyright (c) 2023 Replicable-MARL

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Implement ACPPO algorithm based on Rlib original PPO.
"""

from typing import List, Type, Union
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.policy.policy import Policy
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils.torch_ops import explained_variance
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import TensorType
from ray.rllib.agents.ppo.ppo import PPOTrainer
from ray.rllib.utils.torch_ops import apply_grad_clipping
from ray.rllib.policy.torch_policy import LearningRateSchedule, EntropyCoeffSchedule
from marllib.marl.algos.utils.setup_utils import setup_torch_mixins
from marllib.marl.algos.utils.centralized_critic_hetero import (
    add_all_agents_gae,
)
from ray.rllib.examples.centralized_critic import CentralizedValueMixin
from marllib.marl.algos.utils.setup_utils import get_device
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy, KLCoeffMixin, ppo_surrogate_loss, kl_and_loss_stats
import torch
from marllib.marl.algos.utils.heterogeneous_updateing import update_m_advantage, get_each_agent_train
from marllib.marl.algos.utils.agent_chained_critic import acpo_postprocessing
from ray.rllib.agents.ppo.ppo import PPOTrainer, DEFAULT_CONFIG as PPO_CONFIG
from ray.rllib.utils.torch_ops import apply_grad_clipping
from ray.rllib.models.action_dist import ActionDistribution

from ray.rllib.utils.torch_ops import sequence_mask

from torch.distributions.kl import kl_divergence
from torch.distributions.normal import Normal
import numpy as np
from gym.spaces import Box, Discrete
from marllib.marl.models.zoo.mlp.belief_network import BeliefNetwork


class ACPPOPrevAgentPolicyMixin:
    """Lazily wires predecessor policy models for prev policy inference."""

    def _ensure_initialized(self):
        if hasattr(self, "_predecessor_policies_wired"):
            return
        self._predecessor_policy_models = []
        self._predecessor_policies_wired = False
        custom_config = self.config["model"]["custom_model_config"]
        self._prev_agent_policy_access = custom_config.get(
            "prev_agent_policy_access", False)

    def _wire_predecessor_policies(self, all_policies):
        self._ensure_initialized()
        if self._predecessor_policies_wired:
            return
        sorted_pids = sorted(all_policies.keys(),
                             key=lambda pid: int(pid.split("_")[-1]))
        my_pid = None
        for pid, pol in all_policies.items():
            if pol is self:
                my_pid = pid
                break
        assert my_pid is not None, "Could not find self in all_policies"
        my_idx = sorted_pids.index(my_pid)
        self._predecessor_policy_models = [
            all_policies[sorted_pids[k]].model
            for k in range(my_idx)
        ]
        self._predecessor_policies_wired = True

    def _infer_predecessor_policies(self, obs_tensor):
        """Run predecessor policies to get their distributions."""
        custom_config = self.config["model"]["custom_model_config"]
        n_agents = custom_config["num_agents"]
        act_space = custom_config["space_act"]
        if isinstance(act_space, Box):
            action_dim = act_space.shape[0]
            policy_dim_per_agent = action_dim * 2
        elif isinstance(act_space, Discrete):
            action_dim = act_space.n
            policy_dim_per_agent = action_dim
        else:
            raise NotImplementedError

        variable_width = custom_config.get("ac_agent_id", None) is not None
        n_pred = len(self._predecessor_policy_models)
        B = obs_tensor.shape[0]

        if variable_width:
            total_dim = n_pred * policy_dim_per_agent
        else:
            total_dim = (n_agents - 1) * policy_dim_per_agent
        result = torch.zeros(B, total_dim, device=obs_tensor.device,
                             dtype=obs_tensor.dtype)

        for k, pred_model in enumerate(self._predecessor_policy_models):
            input_dict = {"obs": {"obs": obs_tensor}}
            if variable_width:
                # Predecessor k expects (B, k*pdim) --> slice from result.
                input_dict["prev_agent_policies"] = result[:, :k * policy_dim_per_agent].clone()
            else:
                input_dict["prev_agent_policies"] = result.clone()
            agent_index = torch.full((B,), k, dtype=torch.long,
                                     device=obs_tensor.device)
            input_dict["agent_index"] = agent_index

            init_states = pred_model.get_initial_state()
            s_in = [torch.zeros(B, s.shape[0], device=obs_tensor.device)
                    for s in init_states]
            seq = torch.ones(B, dtype=torch.int32, device=obs_tensor.device)

            with torch.no_grad():
                logits, _ = pred_model.forward(input_dict, s_in, seq)

            offset = k * policy_dim_per_agent
            if isinstance(act_space, Box):
                result[:, offset:offset + policy_dim_per_agent] = logits[:, :policy_dim_per_agent]
            else:
                result[:, offset:offset + action_dim] = torch.softmax(logits, dim=-1)

        return result


class BeliefNetworkMixin:
    """Lazily creates and manages a shared BeliefNetwork for opponent modeling."""

    def _ensure_belief_net(self):
        if hasattr(self, "_belief_net_initialized"):
            return
        self._belief_net_initialized = True
        custom_config = self.config["model"]["custom_model_config"]
        use_bn = custom_config.get("use_belief_networks", False)
        self._use_belief_networks = use_bn
        if not use_bn:
            self.belief_net = None
            self.belief_optimizer = None
            return
        act_space = custom_config["space_act"]
        obs_dim = custom_config["space_obs"]["obs"].shape[0]
        if isinstance(act_space, Discrete):
            output_dim = act_space.n
        elif isinstance(act_space, Box):
            output_dim = act_space.shape[0] * 2  # mean + logstd
        else:
            raise NotImplementedError(f"BeliefNetwork: unsupported action space {act_space}")
        n_agents = custom_config["num_agents"]
        belief_lr = custom_config.get("belief_lr", 1e-3)
        belief_hidden_size = custom_config.get("belief_hidden_size", 64)
        self.belief_net = BeliefNetwork(obs_dim, output_dim, n_agents,
                                        hidden_sizes=(belief_hidden_size, belief_hidden_size)).to(self.device)
        self.belief_optimizer = torch.optim.Adam(
            self.belief_net.parameters(), lr=belief_lr)

    _BELIEF_PREFIX = "_belief_net."

    def get_weights(self):
        weights = super().get_weights()
        self._ensure_belief_net()
        if getattr(self, "belief_net", None) is not None:
            for k, v in self.belief_net.state_dict().items():
                weights[self._BELIEF_PREFIX + k] = v.cpu().detach().numpy()
        return weights

    def set_weights(self, weights):
        belief_weights = {}
        model_weights = {}
        for k, v in weights.items():
            if k.startswith(self._BELIEF_PREFIX):
                belief_weights[k[len(self._BELIEF_PREFIX):]] = v
            else:
                model_weights[k] = v
        super().set_weights(model_weights)
        if belief_weights:
            self._ensure_belief_net()
            if getattr(self, "belief_net", None) is not None:
                from ray.rllib.utils.torch_ops import convert_to_torch_tensor
                belief_weights = convert_to_torch_tensor(
                    belief_weights, device=self.device)
                self.belief_net.load_state_dict(belief_weights)


def _try_eager_wire(policy):
    """Wire predecessor policies eagerly for eval workers."""
    try:
        from ray.rllib.evaluation.rollout_worker import get_global_worker
        worker = get_global_worker()
        if worker is None:
            return
        policy_map = worker.policy_map
        if policy_map is None or len(policy_map) <= 1:
            return
        all_policies = {}
        for pid, pol in policy_map.items():
            all_policies[pid] = pol
        policy._wire_predecessor_policies(all_policies)
    except Exception:
        pass


def action_distribution_fn_acppo(policy, model, input_dict, *,
                                  is_training=False,
                                  state_batches=None, seq_lens=None,
                                  **_kwargs):
    """Inject predecessor policy distributions into input_dict."""
    if hasattr(policy, "_ensure_initialized"):
        policy._ensure_initialized()
    use_prev = getattr(policy, "_prev_agent_policy_access", False)

    if use_prev and not is_training:
        wired = getattr(policy, "_predecessor_policies_wired", False)
        if not wired:
            _try_eager_wire(policy)
            wired = getattr(policy, "_predecessor_policies_wired", False)
        if wired and len(policy._predecessor_policy_models) > 0:
            obs_flat = input_dict[SampleBatch.OBS]
            obs_dim = policy.config["model"]["custom_model_config"][
                "space_obs"]["obs"].shape[0]
            obs = obs_flat[:, :obs_dim]
            pred_policies = policy._infer_predecessor_policies(obs)
            input_dict["prev_agent_policies"] = pred_policies

    elif not is_training:
        # Belief network inference during rollout
        if hasattr(policy, "_ensure_belief_net"):
            policy._ensure_belief_net()
        use_bn = getattr(policy, "_use_belief_networks", False)
        if use_bn and policy.belief_net is not None:
            custom_config = policy.config["model"]["custom_model_config"]
            act_space = custom_config["space_act"]
            if isinstance(act_space, Discrete):
                n_agents = custom_config["num_agents"]
                n_actions = act_space.n
                obs_dim = custom_config["space_obs"]["obs"].shape[0]
                mask_flag = custom_config.get("mask_flag", False)
                obs_flat = input_dict[SampleBatch.OBS]
                B = obs_flat.shape[0]
                if mask_flag:
                    action_mask_dim = n_actions
                    obs = obs_flat[:, action_mask_dim:action_mask_dim + obs_dim].float()
                else:
                    obs = obs_flat[:, :obs_dim].float()
                agent_ids = input_dict["agent_index"].long()
                prev_agent_policies = torch.zeros(
                    B, (n_agents - 1) * n_actions,
                    device=obs.device, dtype=obs.dtype)
                with torch.no_grad():
                    for j in range(n_agents):
                        # Only predict predecessor j for agents with id > j
                        mask = agent_ids > j
                        if mask.sum() == 0:
                            continue
                        obs_j = obs[mask]
                        ids_j = agent_ids[mask]
                        logits_j = policy.belief_net(obs_j, ids_j, j)
                        if mask_flag:
                            # Use predecessor j's action mask, not the
                            # observing agents' masks.  During rollout all
                            # agents' obs are batched together so row j
                            # holds predecessor j's observation.
                            pred_j_mask = obs_flat[j, :action_mask_dim]
                            inf_mask_j = torch.clamp(
                                torch.log(pred_j_mask.float()), min=-1e10)
                            logits_j = logits_j + inf_mask_j
                        probs_j = torch.softmax(logits_j, dim=-1)
                        offset = j * n_actions
                        prev_agent_policies[mask, offset:offset + n_actions] = probs_j
                input_dict["prev_agent_policies"] = prev_agent_policies
            elif isinstance(act_space, Box):
                n_agents = custom_config["num_agents"]
                action_dim = act_space.shape[0]
                policy_dim = action_dim * 2  # mean + logstd
                obs_dim = custom_config["space_obs"]["obs"].shape[0]
                obs_flat = input_dict[SampleBatch.OBS]
                B = obs_flat.shape[0]
                obs = obs_flat[:, :obs_dim].float()
                agent_ids = input_dict["agent_index"].long()
                prev_agent_policies = torch.zeros(
                    B, (n_agents - 1) * policy_dim,
                    device=obs.device, dtype=obs.dtype)
                with torch.no_grad():
                    for j in range(n_agents):
                        mask = agent_ids > j
                        if mask.sum() == 0:
                            continue
                        obs_j = obs[mask]
                        ids_j = agent_ids[mask]
                        pred_j = policy.belief_net(obs_j, ids_j, j)
                        offset = j * policy_dim
                        prev_agent_policies[mask, offset:offset + policy_dim] = pred_j
                input_dict["prev_agent_policies"] = prev_agent_policies

    dist_inputs, state_out = model(input_dict, state_batches or [], seq_lens)
    return dist_inputs, policy.dist_class, state_out


def gaussian_kl(mu1, sigma1, mu2, sigma2):
    """KL divergence between two diagonal Gaussians."""
    p = Normal(loc=mu1, scale=torch.exp(sigma1))
    q = Normal(loc=mu2, scale=torch.exp(sigma2))

    return kl_divergence(p, q).sum(dim=-1)
    

def acppo_loss(
        policy: Policy, model: ModelV2,
        dist_class: ActionDistribution,
        train_batch: SampleBatch) -> Union[TensorType, List[TensorType]]:
    """ACPPO loss with belief network training."""

    CentralizedValueMixin.__init__(policy)
    
    opp_action_in_cc = policy.config["model"]["custom_model_config"]["opp_action_in_cc"]
    vf_use_prev_agent_policies = policy.config["model"]["custom_model_config"]["vf_use_prev_agent_policies"]

    func = ppo_surrogate_loss

    vf_saved = model.value_function
    
    
    prev_agent_inputs = None

    if vf_use_prev_agent_policies:
        prev_agent_inputs = train_batch["prev_agent_policies"]
    # Append agent_id one-hot for uniform-width (shared policy).
    # Variable-width separate policies skip this (constant input).
    variable_width = policy.config["model"]["custom_model_config"].get(
        "ac_agent_id", None) is not None
    if prev_agent_inputs is not None and not variable_width:
        agent_id_inputs = torch.nn.functional.one_hot(
            train_batch["agent_index"].long(),
            num_classes=policy.config["model"]["custom_model_config"]["num_agents"]
        ).to(get_device())
        prev_agent_inputs = torch.cat(
            [prev_agent_inputs, agent_id_inputs], dim=-1)
        
    model.value_function = lambda: policy.model.central_value_function(train_batch["state"],
                                                                       train_batch[
                                                                           "opponent_actions"] if opp_action_in_cc else None,
                                                                       prev_agent_inputs)
    policy._central_value_out = model.value_function()
    loss = func(policy, model, dist_class, train_batch)

    model.value_function = vf_saved

    # Belief network loss (cross-entropy against true predecessor logits).
    # Stepped separately so gradients don't flow into the PPO model.
    if hasattr(policy, "_ensure_belief_net"):
        policy._ensure_belief_net()
    if getattr(policy, "_use_belief_networks", False):
        if "predecessor_true_logits" in train_batch:
            custom_config = policy.config["model"]["custom_model_config"]
            act_space = custom_config["space_act"]
            n_agents = custom_config["num_agents"]
            obs_dim = custom_config["space_obs"]["obs"].shape[0]
            mask_flag = custom_config.get("mask_flag", False)

            is_discrete = isinstance(act_space, Discrete)
            if is_discrete:
                policy_dim = act_space.n
            else:
                policy_dim = act_space.shape[0] * 2  # mean + logstd

            if mask_flag and is_discrete:
                action_mask_dim = act_space.n
                all_obs = train_batch[SampleBatch.OBS][:, action_mask_dim:action_mask_dim + obs_dim].float()
            else:
                all_obs = train_batch[SampleBatch.OBS][:, :obs_dim].float()
            true_logits_all = train_batch["predecessor_true_logits"].float()
            agent_ids = train_batch["agent_index"].long()

            # For RNN models, build a mask to exclude zero-padded timesteps.
            # Padded positions have garbage obs/logits and would train the
            # belief net on noise (zero obs -> uniform target).
            seq_lens = train_batch.get(SampleBatch.SEQ_LENS, None)
            if seq_lens is not None and len(seq_lens) > 0:
                max_seq_len = all_obs.shape[0] // len(seq_lens)
                seq_valid = sequence_mask(
                    seq_lens, max_seq_len,
                    time_major=model.is_time_major()
                ).reshape(-1)
            else:
                seq_valid = torch.ones(all_obs.shape[0], dtype=torch.bool,
                                       device=all_obs.device)

            belief_loss_acc = torch.tensor(0.0, device=all_obs.device)
            n_terms = 0
            for j in range(n_agents - 1):
                # Only samples from agents with id > j have valid true logits
                # for predecessor j (agent 0 has no predecessors, agent 1 has
                # predecessor 0, etc.)
                # Also exclude RNN-padded timesteps.
                mask = (agent_ids > j) & seq_valid
                if mask.sum() == 0:
                    continue
                obs_j = all_obs[mask]
                self_ids_j = agent_ids[mask]
                true_j = true_logits_all[mask, j * policy_dim:(j + 1) * policy_dim]
                hat_j = policy.belief_net(obs_j, self_ids_j, j)
                if is_discrete:
                    target_probs = torch.softmax(true_j.detach(), dim=-1)
                    belief_loss_acc = belief_loss_acc + (
                        -(target_probs * torch.log_softmax(hat_j, dim=-1))
                        .sum(-1).mean()
                    )
                else:
                    action_dim = act_space.shape[0]
                    true_mu = true_j[:, :action_dim].detach()
                    true_logstd = true_j[:, action_dim:].detach()
                    hat_mu = hat_j[:, :action_dim]
                    hat_logstd = hat_j[:, action_dim:]
                    belief_loss_acc = belief_loss_acc + gaussian_kl(
                        true_mu, true_logstd, hat_mu, hat_logstd).mean()
                n_terms += 1
            if n_terms > 0:
                belief_loss_val = belief_loss_acc / n_terms
                policy.belief_optimizer.zero_grad()
                belief_loss_val.backward()
                policy.belief_optimizer.step()
                model.tower_stats["belief_loss"] = belief_loss_val.detach()
            else:
                model.tower_stats["belief_loss"] = torch.tensor(0.0)
        else:
            model.tower_stats["belief_loss"] = torch.tensor(0.0)
    else:
        model.tower_stats["belief_loss"] = torch.tensor(0.0)

    return loss



def acppo_stats(policy, train_batch):
    stats = kl_and_loss_stats(policy, train_batch)
    belief_vals = policy.get_tower_stats("belief_loss")
    if belief_vals:
        stats["belief_loss"] = torch.mean(torch.stack(belief_vals))
    return stats


ACPPOTorchPolicy =  PPOTorchPolicy.with_updates(
    name="ACPPOTorchPolicy",
    get_default_config=lambda: PPO_CONFIG,
    postprocess_fn=acpo_postprocessing,
    loss_fn=acppo_loss,
    stats_fn=acppo_stats,
    before_init=setup_torch_mixins,
    extra_grad_process_fn=apply_grad_clipping,
    action_distribution_fn=action_distribution_fn_acppo,
    mixins=[
        LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin,
        CentralizedValueMixin,
        ACPPOPrevAgentPolicyMixin,
        BeliefNetworkMixin,
    ])


def get_policy_class_acppo(config_):
    if config_["framework"] == "torch":
        return ACPPOTorchPolicy


ACPPOTrainer = PPOTrainer.with_updates(
    name="ACPPOTrainer",
    default_policy=None,
    get_policy_class=get_policy_class_acppo,
)
