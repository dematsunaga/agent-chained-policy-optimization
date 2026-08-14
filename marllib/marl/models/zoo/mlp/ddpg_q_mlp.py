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

from typing import Dict, List
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch, TensorType

torch, nn = try_import_torch()


class CentralizedQMLP(TorchModelV2, nn.Module):
    """Q-network for DDPG-family algorithms.

    Architecture: concat(raw_inputs) -> Linear -> ReLU -> Linear -> ReLU -> Linear -> Q
    No pre-encoding of observations or state -- raw values are concatenated with
    actions and fed directly into the MLP. 

    Input composition by algorithm:
      MADDPG (global_state):  concat(state, own_action, opp_actions)
      MADDPG (no global):     concat(all_agent_obs, own_action, opp_actions)
      IDDPG:                         concat(obs, own_action)
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name,
                 **kwargs):
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        custom_config = model_config["custom_model_config"]
        self.algorithm = custom_config["algorithm"]
        self.global_state_flag = custom_config.get("global_state_flag", False)
        self.n_agents = custom_config["num_agents"]

        # Hidden layer sizes from encode_layer config (e.g. "256-256").
        encode_layer = custom_config["model_arch_args"]["encode_layer"]
        hidden_sizes = [int(x) for x in encode_layer.split("-")]

        # Dummy state size for get_initial_state (RLlib recurrent interface).
        self.hidden_state_size = custom_config["model_arch_args"]["hidden_state_size"]

        # Compute input dimension based on algorithm variant.
        space_obs = custom_config["space_obs"]
        space_act = custom_config["space_act"]
        action_dim = space_act.shape[0]

        self.agent_id_dim = 0

        if self.algorithm in ["maddpg"]:
            n_action_slots = self.n_agents
            if self.global_state_flag:
                state_dim = space_obs["state"].shape[0]
                input_dim = state_dim + n_action_slots * action_dim + self.agent_id_dim
            else:
                obs_dim = space_obs["obs"].shape[0]
                input_dim = obs_dim * self.n_agents + n_action_slots * action_dim + self.agent_id_dim
        else:  # IDDPG
            obs_dim = space_obs["obs"].shape[0]
            input_dim = obs_dim + action_dim

        self.input_dim = input_dim

        # Build 2-layer MLP with ReLU 
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_sizes:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(prev_dim, num_outputs)

        self._features = None

    @override(ModelV2)
    def get_initial_state(self):
        # Dummy state so RLlib treats this as "recurrent" (required by the
        # existing IDDPGTorchModel / action_distribution_fn interface).
        return self.out.weight.new(1, self.hidden_state_size).zero_().squeeze(0),

    @override(ModelV2)
    def forward(self, input_dict: Dict[str, TensorType],
                state: List[TensorType],
                seq_lens: TensorType) -> (TensorType, List[TensorType]):

        obs_inputs = input_dict["obs"]["obs"]
        state_inputs = input_dict.get("state")
        action_inputs = input_dict.get("actions")
        opp_action_inputs = input_dict.get("opponent_actions")

        B = obs_inputs.shape[0]

        if self.algorithm in ["maddpg"]:
            if action_inputs is not None:
                if self.global_state_flag:
                    x = torch.cat((state_inputs,
                                   action_inputs,
                                   opp_action_inputs.reshape(B, -1)), -1)
                else:
                    # state_inputs is (B, n_agents, obs_dim) -- flatten.
                    x = torch.cat((state_inputs.reshape(B, -1),
                                   action_inputs,
                                   opp_action_inputs.reshape(B, -1)), -1)
            else:
                x = torch.zeros(B, self.input_dim, device=obs_inputs.device)
        else:  # IDDPG
            if action_inputs is not None:
                x = torch.cat((obs_inputs, action_inputs), -1)
            else:
                x = torch.zeros(B, self.input_dim, device=obs_inputs.device)

        self._features = self.mlp(x)
        output = self.out(self._features)
        return output.reshape(-1, self.num_outputs), state
