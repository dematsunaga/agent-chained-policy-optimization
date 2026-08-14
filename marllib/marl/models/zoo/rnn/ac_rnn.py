import numpy as np
from gym.spaces import Box, MultiDiscrete
from functools import reduce
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from ray.rllib.models.torch.misc import SlimFC, normc_initializer
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.typing import Dict, TensorType, List
from ray.rllib.utils.torch_ops import FLOAT_MIN

from marllib.marl.models.zoo.rnn.base_rnn import BaseRNN
from marllib.marl.models.zoo.encoder.ac_encoder import AgentChainedEncoder
from marllib.marl.models.zoo.encoder.cc_encoder import CentralizedEncoder
from marllib.marl.models.zoo.encoder.base_encoder import BaseEncoder
from marllib.marl.models.zoo.rnn.base_rnn import add_time_dimension
from marllib.marl.algos.utils.setup_utils import get_device
from torch.optim import Adam
from torch.distributions import Categorical, Normal
import copy
import torch.nn as nn

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()

LOG_STD_MAX = 2
LOG_STD_MIN = -20

class LogStdModule(nn.Module):
    def __init__(self, initial_value=0.0, size=1):
        super().__init__()
        self.log_std = nn.Parameter(torch.ones(size) * initial_value)
    def forward(self):
        return self.log_std

class AgentChainedRNN(BaseRNN):
    """
    RNN version of AgentChainedMLP.
    - Actor: AgentChainedEncoder (+ agent-id + optional prev-agent policies/actions) -> RNN -> policy head
    - Critic: CentralizedEncoder over (state or concat obs) (+ optional prev-agent policies/actions) -> VF head
    """
    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 **kwargs):
        super().__init__(obs_space, action_space, num_outputs, model_config, name, **kwargs)

        self.custom_config = model_config["custom_model_config"]
        self.vf_use_prev_agent_policies = self.custom_config["vf_use_prev_agent_policies"]
        self.use_prev_agent_policies = self.custom_config["use_prev_agent_policies"]
        self.global_state_flag = self.custom_config["global_state_flag"]

        # action dimensions for prev-agent buffers
        if isinstance(self.custom_config["space_act"], Box):
            self.action_dim = self.action_space.shape[0]
            # For policies, continuous stores mean and logstd
            if self.vf_use_prev_agent_policies:
                self.other_agent_policy_size = (self.n_agents - 1) * self.action_dim * 2
            else:
                self.other_agent_policy_size = 0
        else:
            # discrete or MultiDiscrete uses flat-one-hot for policies/actions
            if isinstance(self.custom_config["space_act"], MultiDiscrete):
                self.action_dim = sum([sp.n for sp in self.action_space])
            else:
                self.action_dim = self.action_space.n
            self.other_agent_policy_size = (self.n_agents - 1) * self.action_dim

        if self.use_prev_agent_policies:
            # action branch should change to include previous agent policies and agent ID
            self.p_branch = SlimFC(
                in_size=self.hidden_state_size + self.other_agent_policy_size + self.n_agents,
                out_size=num_outputs,
                initializer=normc_initializer(0.01),
                activation_fn=None)
        
        # Centralized critic encoder input dim handling (match ac_mlp logic)
        if self.global_state_flag:
            input_dim = self.full_obs_space['state'].shape[0]
        else:
            input_dim = self.n_agents * self.full_obs_space['obs'].shape[0]

        full_obs_space = self.full_obs_space
        if self.vf_use_prev_agent_policies:
            input_dim += self.other_agent_policy_size
            # plus agent id one-hot if you also condition VF on which agent (mirroring ac_mlp forward usage)
            input_dim += self.n_agents

        # Centralized VF encoder
        self.cc_vf_encoder = CentralizedEncoder(model_config, full_obs_space, add_obs=False, input_dim=input_dim)

        # Centralized value head size (optionally concat opponent actions to encoder output)
        if self.custom_config["opp_action_in_cc"]:
            if isinstance(self.custom_config["space_act"], Box):
                vf_in = self.cc_vf_encoder.output_dim + num_outputs * (self.n_agents - 1) // 2
            else:
                vf_in = self.cc_vf_encoder.output_dim + num_outputs * (self.n_agents - 1)
        else:
            vf_in = self.cc_vf_encoder.output_dim

        self.cc_vf_branch = SlimFC(
            in_size=vf_in,
            out_size=1,
            initializer=normc_initializer(0.01),
            activation_fn=None
        )

    @override(BaseRNN)
    def get_initial_state(self) -> List[TensorType]:
        return super().get_initial_state()

    
    def central_value_function(self, state, opponent_actions=None, prev_agent_inputs=None) -> TensorType:
        """
        Centralized critic over either global state or concatenated local obs.
        state: [B, state_dim] (global) or [B, n_agents*obs_dim]
        opponent_actions: None or tensor of shape:
          - continuous: [B, n_agents-1, action_dim]
          - MultiDiscrete: [B, n_agents-1, k] with per-dim categories
          - discrete: [B, n_agents-1]
        prev_agent_inputs: optional prev-agent policies/actions + agent-id one-hot for VF
        """
        assert self._features is not None, "must call forward() first"

        B = state.shape[0]
        if prev_agent_inputs is not None and self.vf_use_prev_agent_policies:
            x = self.cc_vf_encoder(torch.cat([state.reshape(B, -1), prev_agent_inputs], dim=-1))
        else:
            x = self.cc_vf_encoder(state)

        if opponent_actions is not None:
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
        else:
            x = torch.cat([x.reshape(B, -1)], 1)

        if self.q_flag:
            return torch.reshape(self.cc_vf_branch(x), [-1, self.num_outputs])
        else:
            return torch.reshape(self.cc_vf_branch(x), [-1])

    def forward_om(self, input_dict: Dict[str, TensorType]) -> (TensorType, List[TensorType]):
        flat_inputs = input_dict["obs"].float()
        prev_agent_policies = input_dict["prev_agent_policies"].float()
        agent_id_inputs = torch.nn.functional.one_hot(
                input_dict["agent_index"].long(),
                num_classes=self.n_agents).float()
        
        if self.use_prev_agent_policies:
            hidden_state = input_dict["hidden_state"].float()
            inputs = torch.cat([hidden_state, prev_agent_policies, agent_id_inputs], dim=-1)
        else:
            _features = self.p_encoder(flat_inputs)
            inputs = _features# torch.cat([_features, agent_id_inputs], dim=-1)
          
        output = self.p_branch(inputs)
        return output
    
    @override(BaseRNN)
    def forward(self, input_dict: Dict[str, TensorType],
                hidden_state: List[TensorType],
                seq_lens: TensorType) -> (TensorType, List[TensorType]):
        """
        Adds time dimension to batch before sending inputs to forward_rnn()
        """
        if self.custom_config["global_state_flag"] or self.custom_config["mask_flag"]:
            flat_inputs = input_dict["obs"]["obs"].float()
            # Convert action_mask into a [0.0 || -inf]-type mask.
            if self.custom_config["mask_flag"]:
                action_mask = input_dict["obs"]["action_mask"]
                inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        else:
            flat_inputs = input_dict["obs"]["obs"].float()

        if isinstance(seq_lens, np.ndarray):
            seq_lens = torch.Tensor(seq_lens).int()
        max_seq_len = flat_inputs.shape[0] // seq_lens.shape[0]

        self.time_major = self.model_config.get("_time_major", False)
        inputs = add_time_dimension(
            flat_inputs,
            max_seq_len=max_seq_len,
            framework="torch",
            time_major=self.time_major,
        )
        output, hidden_state = self.forward_rnn(input_dict, inputs, hidden_state, seq_lens, inf_mask)
        output = torch.reshape(output, [-1, self.num_outputs])

        if self.custom_config["mask_flag"]:
            output = output + inf_mask

        return output, hidden_state

    @override(BaseRNN)
    def forward_rnn(self, input_dict, inputs, hidden_state, seq_lens, inf_mask):
        self.inputs = inputs

        x = self.p_encoder(self.inputs)

        if self.custom_config["model_arch_args"]["core_arch"] == "gru":
            self._features, h = self.rnn(x, torch.unsqueeze(hidden_state[0], 0))
            #============================================================
            # specific to acppo: other agent policies
            if not self.use_prev_agent_policies:
                logits = self.p_branch(self._features)
            else:
                # Case 1: initialization: ()
                if input_dict["eps_id"].sum() == 0 and inputs.sum() == 0 and inputs.shape[0] != self.n_agents:
                    logits = torch.zeros((inputs.shape[0] * inputs.shape[1], self.num_outputs), device=get_device(), dtype=inputs.dtype)
                else:
                    agent_id_inputs = torch.nn.functional.one_hot(
                        input_dict["agent_index"].long(),
                        num_classes=self.n_agents).float().unsqueeze(1).repeat(1, inputs.shape[1], 1)
                    
                    my_ids = list(set(input_dict["agent_index"].cpu().numpy()))
                    # Case 2: training time
                    if "prev_agent_policies" in input_dict and len(my_ids) > 0:
                        agent_id_inputs = torch.nn.functional.one_hot(
                            input_dict["agent_index"].long(),
                            num_classes=self.n_agents).float()
                    
                        prev_agent_policies = input_dict["prev_agent_policies"]
                        joint_inputs = torch.cat([self._features.reshape(-1, self._features.shape[-1]), prev_agent_policies, agent_id_inputs], dim=-1)
                        logits = self.p_branch(joint_inputs)

                    # Case 3: inference
                    else:
                        logits = []
                        for agent_id in range(self.n_agents):
                            prev_agent_policies = torch.zeros(
                                (1, self.other_agent_policy_size),
                                dtype=inputs.dtype,
                                device=get_device())
                            
                            
                            for j in range(agent_id + 1):                      
                                agent_id_inputs = torch.zeros((1, self.n_agents), dtype=inputs.dtype, device=get_device())
                                agent_id_inputs[:, j] = 1.0

                                # get the inputs from current agent
                                joint_inputs = torch.cat([self._features[agent_id], prev_agent_policies, agent_id_inputs], dim=-1)
                                logit = self.p_branch(joint_inputs)
                                # if self.custom_config["mask_flag"]:
                                #     logit = logit + inf_mask[j]
                                
                                if j == agent_id:
                                    logits.append(logit)
                                    break
                                if isinstance(self.custom_config["space_act"], Box):  # continuous
                                    offset = j * self.action_dim * 2
                                    prev_agent_policies[:, offset:offset + self.action_dim * 2] = logit
                                else:
                                    offset = j * self.action_dim
                                    agent_i_policies = torch.nn.functional.softmax(logit, dim=-1)
                                    prev_agent_policies[:, offset:offset + self.action_dim] = agent_i_policies
                        logits = torch.cat(logits, dim=0)
            #============================================================
            return logits, [torch.squeeze(h, 0)]

        elif self.custom_config["model_arch_args"]["core_arch"] == "lstm":
            # TODO: not implemented
            self._features, [h, c] = self.rnn(
                x, [torch.unsqueeze(hidden_state[0], 0),
                    torch.unsqueeze(hidden_state[1], 0)])
            logits = self.p_branch(self._features)
            return logits, [torch.squeeze(h, 0), torch.squeeze(c, 0)]

        else:
            raise ValueError("rnn core_arch wrong: {}".format(self.custom_config["model_arch_args"]["core_arch"]))
        
    @override(BaseRNN)
    def actor_parameters(self):
        # Include encoder, RNN, and policy head
        return reduce(lambda x, y: x + y, map(lambda p: list(p.parameters()), [self.p_encoder, self.rnn, self.p_branch]))

    @override(BaseRNN)
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
        AgentChainedRNN.update_use_torch_adam(
            loss=(-1 * loss),
            optimizer=getattr(self, "actor_optimizer", Adam(params=self.parameters(), lr=lr)),
            parameters=self.parameters(),
            grad_clip=grad_clip
        )

    @staticmethod
    def update_use_torch_adam(loss, parameters, optimizer, grad_clip):
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, grad_clip)
        optimizer.step()
