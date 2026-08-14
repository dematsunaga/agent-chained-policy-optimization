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

import numpy as np
from gym.spaces import Box, MultiDiscrete
from functools import reduce
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from ray.rllib.models.torch.misc import SlimFC, normc_initializer
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import TensorType
from marllib.marl.models.zoo.mlp.base_mlp import BaseMLP
from marllib.marl.models.zoo.encoder.ac_encoder import AgentChainedEncoder
from marllib.marl.models.zoo.encoder.cc_encoder import CentralizedEncoder
from torch.optim import Adam

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from marllib.marl.models.zoo.encoder.base_encoder import BaseEncoder
from ray.rllib.utils.typing import Dict, TensorType, List
from ray.rllib.utils.torch_ops import FLOAT_MIN

from marllib.marl.algos.utils.setup_utils import get_device


LOG_STD_MAX = 2
LOG_STD_MIN = -20

torch, nn = try_import_torch()

class LogStdModule(nn.Module):
    def __init__(self, initial_value=0.0, size=1):
        super().__init__()
        self.log_std = nn.Parameter(torch.ones(size) * initial_value)
        
    def forward(self):
        return self.log_std
    
class AgentChainedMLP(BaseMLP):

    def __init__(
            self,
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name,
            **kwargs,
    ):

        super().__init__(obs_space, action_space, num_outputs, model_config,
                         name, **kwargs)
        self.custom_config = model_config["custom_model_config"]
        self.vf_use_prev_agent_policies = self.custom_config["vf_use_prev_agent_policies"]
        self.use_prev_agent_policies = self.custom_config["use_prev_agent_policies"]
        self.global_state_flag = self.custom_config["global_state_flag"]
        
        # encoder
        self.p_encoder = AgentChainedEncoder(model_config, self.full_obs_space, self.action_space, use_policy_encoder=True)
        if isinstance(self.custom_config["space_act"], Box):  # continuous
            self.action_dim = self.action_space.shape[0]
            self._policy_dim_per_agent = self.action_dim * 2  # mean and logstd
        else:
            self.action_dim = self.action_space.n
            self._policy_dim_per_agent = self.action_dim

        # Variable-width: when ac_agent_id is set, use agent_id predecessors
        # instead of (N-1) and drop agent_id one-hot.
        self._ac_agent_id = self.custom_config.get("ac_agent_id", None)
        self._variable_width = (
            self._ac_agent_id is not None
            and self.use_prev_agent_policies
        )
        if self._variable_width:
            n_pred = self._ac_agent_id
        else:
            n_pred = self.n_agents - 1
        self.other_agent_policy_size = n_pred * self._policy_dim_per_agent

        #########
        # from cc_mlp.py
        #########
        if self.global_state_flag:
            input_dim = self.full_obs_space['state'].shape[0]
        else:
            input_dim = self.n_agents * self.full_obs_space['obs'].shape[0]
        full_obs_space = self.full_obs_space
        if self.vf_use_prev_agent_policies:
            input_dim += self.other_agent_policy_size
            # agent ID one-hot (skip for variable-width separate policies)
            if not self._variable_width:
                input_dim += self.n_agents

        # encoder for centralized VF
        self.cc_vf_encoder = CentralizedEncoder(model_config, full_obs_space, add_obs=False, input_dim=input_dim)

        # Central VF
        if self.custom_config["opp_action_in_cc"]:
            if isinstance(self.custom_config["space_act"], Box):  # continuous
                input_size = self.cc_vf_encoder.output_dim + num_outputs * (self.n_agents - 1) // 2
            else:
                input_size = self.cc_vf_encoder.output_dim + num_outputs * (self.n_agents - 1)
        else:
            input_size = self.cc_vf_encoder.output_dim

        self.cc_vf_branch = SlimFC(
            in_size=input_size,
            out_size=1,
            initializer=normc_initializer(0.01),
            activation_fn=None)
        
            
    def central_value_function(self, state, opponent_actions=None, prev_agent_inputs=None) -> TensorType:
        assert self._features is not None, "must call forward() first"
        B = state.shape[0]
        if prev_agent_inputs is not None and self.vf_use_prev_agent_policies:
            inputs = torch.cat([state.reshape(B, -1), prev_agent_inputs], dim=-1)
            x = self.cc_vf_encoder(inputs)
        else:
            x = self.cc_vf_encoder(state)

        if opponent_actions is None:
            x = torch.cat([x.reshape(B, -1)], 1)
        else:
            if isinstance(self.custom_config["space_act"], Box):  # continuous
                opponent_actions_ls = [opponent_actions[:, i, :]
                                       for i in
                                       range(self.n_agents - 1)]
            elif isinstance(self.custom_config["space_act"], MultiDiscrete):
                opponent_actions_ls = []
                action_space_ls = [single_action_space.n for single_action_space in self.action_space]
                for i in range(self.n_agents - 1):
                    opponent_action_ls = []
                    for single_action_index, single_action_space in enumerate(action_space_ls):
                        opponent_action = torch.nn.functional.one_hot(
                            opponent_actions[:, i, single_action_index].long(), single_action_space).float()
                        opponent_action_ls.append(opponent_action)
                    opponent_actions_ls.append(torch.cat(opponent_action_ls, axis=1))

            else:
                opponent_actions_ls = [
                    torch.nn.functional.one_hot(opponent_actions[:, i].long(), self.num_outputs).float()
                    for i in
                    range(self.n_agents - 1)]

            x = torch.cat([x.reshape(B, -1)] + opponent_actions_ls, 1)

        return torch.reshape(self.cc_vf_branch(x), [-1])
    @override(BaseMLP)
    def actor_parameters(self):

        return reduce(lambda x, y: x + y, map(lambda p: list(p.parameters()), self.actors))
    @override(BaseMLP)
    def critic_parameters(self):
        critics = [self.cc_vf_encoder, self.cc_vf_branch, ]
        return reduce(lambda x, y: x + y, map(lambda p: list(p.parameters()), critics))

    def link_other_agent_policy(self, agent_id, policy):
        if agent_id in self.other_policies:
            if self.other_policies[agent_id] != policy:
                raise ValueError('the policy is not same with the two time look up')
        else:
            self.other_policies[agent_id] = policy

    def update_actor(self, loss, lr, grad_clip):
        AgentChainedMLP.update_use_torch_adam(
            loss=(-1 * loss),
            optimizer=self.actor_optimizer,
            parameters=self.parameters(),
            grad_clip=grad_clip
        )
    def forward_om(self, input_dict: Dict[str, TensorType]) -> (TensorType, List[TensorType]):
        flat_inputs = input_dict["obs"].float()
        prev_agent_policies = input_dict["prev_agent_policies"].float()

        if self._variable_width:
            # Variable-width: encoder has no agent_id one-hot.
            # prev_agent_policies is already sized (B, self.other_agent_policy_size)
            # from the caller (my_id * policy_dim_per_agent).
            if self.use_prev_agent_policies:
                inputs = torch.cat([flat_inputs, prev_agent_policies], dim=-1)
            else:
                inputs = flat_inputs
        else:
            # Uniform-width: include agent_id one-hot.
            agent_id_inputs = torch.nn.functional.one_hot(
                    input_dict["agent_index"].long(),
                    num_classes=self.n_agents).float()
            if self.use_prev_agent_policies:
                inputs = torch.cat([flat_inputs, prev_agent_policies, agent_id_inputs], dim=-1)
            else:
                inputs = torch.cat([flat_inputs, agent_id_inputs], dim=-1)

        _features = self.p_encoder(inputs)

        output = self.p_branch(_features)
        return output
    ## methods from base_mlp.py
    @override(BaseMLP)
    def forward(self, input_dict: Dict[str, TensorType],
                state: List[TensorType],
                seq_lens: TensorType) -> (TensorType, List[TensorType]):

        flat_inputs = input_dict["obs"]["obs"].float()
        
        if self.custom_config["global_state_flag"] or self.custom_config["mask_flag"]:
            # Convert action_mask into a [0.0 || -inf]-type mask.
            # TODO: other agent masks
            if self.custom_config["mask_flag"]:
                action_mask = input_dict["obs"]["action_mask"]
                inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)

        self.inputs = flat_inputs


        if flat_inputs.ndim == 2 and flat_inputs.shape[0] != self.n_agents:
            # (batch_size, input_dim) for initial call to forward or training 
             # use itself to compute the "previous agent" actions
            my_ids = list(set(input_dict["agent_index"].cpu().numpy()))
            
            B = flat_inputs.shape[0]
            if self.use_prev_agent_policies:
                prev_agent_policies = torch.zeros(
                    (B, self.other_agent_policy_size),
                    device=get_device(),
                    dtype=flat_inputs.dtype,
                )
            if len(my_ids) == 1:
                if self.use_prev_agent_policies and "prev_agent_policies" in input_dict:
                    prev_agent_policies = input_dict["prev_agent_policies"].float()
            elif len(my_ids) > 1: # training contains batches that mix data for different agents
                if self.use_prev_agent_policies and "prev_agent_policies" in input_dict:
                    prev_agent_policies = input_dict["prev_agent_policies"].float()
            else:
                raise NotImplementedError

            if self._variable_width:
                # Variable-width: no agent_id one-hot, prev_agent_policies
                # already has the right size from postprocessing.
                if self.use_prev_agent_policies:
                    inputs = torch.cat([flat_inputs, prev_agent_policies], dim=-1)
                else:
                    inputs = flat_inputs
            else:
                # Uniform-width: include agent_id one-hot.
                agent_id_inputs = torch.nn.functional.one_hot(
                    input_dict["agent_index"].long(),
                    num_classes=self.n_agents).float()
                if self.use_prev_agent_policies:
                    inputs = torch.cat([flat_inputs, prev_agent_policies, agent_id_inputs], dim=-1)
                else:
                    inputs = torch.cat([flat_inputs, agent_id_inputs], dim=-1)
            self._features = self.p_encoder(inputs)
            
            output = self.p_branch(self._features)

            if self.custom_config["mask_flag"]:
                output = output + inf_mask
            return output, state
        
        # (n_agents, input_dim)
        assert flat_inputs.ndim == 2 and flat_inputs.shape[0] == self.n_agents

            
        features = []
        logits = []
        logits_logp = []
        mus = []
        log_stds = []
        if self.use_prev_agent_policies:
            use_belief = self.custom_config.get("use_belief_networks", False)
            # When belief predictions were injected by action_distribution_fn
            # (stored in input_dict["prev_agent_policies"]), use them directly
            # so rollout is consistent with training.  Fall back to the
            # autoregressive self-prediction loop only when no belief
            # predictions are available.
            if use_belief and "prev_agent_policies" in input_dict:
                belief_pap = input_dict["prev_agent_policies"].float()
                for agent_id in range(self.n_agents):
                    agent_inputs = flat_inputs[agent_id].reshape(1, -1)
                    pap_i = belief_pap[agent_id].reshape(1, -1)
                    agent_id_inputs = torch.zeros(
                        (1, self.n_agents), dtype=agent_inputs.dtype,
                        device=get_device())
                    agent_id_inputs[:, agent_id] = 1.0
                    inputs_i = torch.cat(
                        [agent_inputs, pap_i, agent_id_inputs], dim=-1)
                    feature = self.p_encoder(inputs_i)
                    logit = self.p_branch(feature)
                    if self.custom_config["mask_flag"]:
                        logit = logit + inf_mask[agent_id]
                    features.append(feature)
                    logits.append(logit)
            else:
                # opponent modelling: each agent infers what the other agent policies are given its own input
                for agent_id in range(self.n_agents):
                    prev_agent_policies = torch.zeros(
                        (1, self.other_agent_policy_size),
                        dtype=flat_inputs.dtype,
                        device=get_device())

                    for j in range(agent_id + 1):
                        # get the inputs from current agent
                        agent_inputs = flat_inputs[agent_id].reshape(1, -1)

                        agent_id_inputs = torch.zeros((agent_inputs.shape[0], self.n_agents), dtype=agent_inputs.dtype, device=get_device())
                        agent_id_inputs[:, j] = 1.0

                        agent_inputs = torch.cat([agent_inputs, prev_agent_policies, agent_id_inputs], dim=-1)
                        feature = self.p_encoder(agent_inputs)

                        logit = self.p_branch(feature)
                        if self.custom_config["mask_flag"]:
                            logit = logit + inf_mask[j]

                        if j == agent_id:
                            features.append(feature)
                            logits.append(logit)
                            break
                        if isinstance(self.custom_config["space_act"], Box):  # continuous
                            offset = j * self.action_dim * 2
                            prev_agent_policies[:, offset:offset + self.action_dim * 2] = logit
                        else:
                            offset = j * self.action_dim
                            agent_i_policies = torch.nn.functional.softmax(logit, dim=-1)
                            prev_agent_policies[:, offset:offset + self.action_dim] = agent_i_policies
        else:
            for agent_id in range(self.n_agents):
                # agent_inputs = torch.index_select(flat_inputs, dim=-2, index=torch.IntTensor([agent_id]))
                
                agent_inputs = flat_inputs[agent_id].reshape(1, -1)
        
                agent_id_inputs = torch.zeros((agent_inputs.shape[0], self.n_agents), dtype=agent_inputs.dtype, device=get_device())
                agent_id_inputs[:, agent_id] = 1.0

                agent_inputs = torch.cat([agent_inputs, agent_id_inputs], dim=-1)
                
                feature = self.p_encoder(agent_inputs)
                
                logit = self.p_branch(feature)
                if self.custom_config["mask_flag"]:
                    logit = logit + inf_mask[agent_id]
                features.append(feature)
                logits.append(logit)
        self._features = torch.cat(features, dim=-2)
        output = torch.cat(logits, dim=-2)
        return output, state

    @staticmethod
    def update_use_torch_adam(loss, parameters, optimizer, grad_clip):
        optimizer.zero_grad()
        loss.backward()
        # total_norm = torch.norm(torch.stack([torch.norm(p.grad) for p in parameters if p.grad is not None]))
        torch.nn.utils.clip_grad_norm_(parameters, grad_clip)
        optimizer.step()
