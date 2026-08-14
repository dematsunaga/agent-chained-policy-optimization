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

from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gym.spaces import Dict as GymDict, Discrete, Box as GymBox
# from gymnasium.spaces import Box as GymnasiumBox, Dict as GymnasiumDict
# from gymnasium_robotics.envs.multiagent_mujoco import MultiAgentMujocoEnv
from gymnasium_robotics.envs.multiagent_mujoco.mamujoco_v1 import parallel_env, get_parts_and_edges

import numpy as np
import time

# Gymnasium-Robotics based MAMuJoCo example, you can add / customize your own
# referring to https://robotics.farama.org/envs/MaMuJoCo/
AGENT_OBSK = 0
env_args_dict = {
    "2AgentAnt": {"scenario": "Ant",  # "Ant-v4"
                  "agent_conf": "2x4",
                  "agent_obsk": AGENT_OBSK,
                  "episode_limit": 1000},
    "4AgentAnt": {"scenario": "Ant",
                  "agent_conf": "4x2",
                  "agent_obsk": AGENT_OBSK,
                  "episode_limit": 1000},
    "2AgentHalfCheetah": {"scenario": "HalfCheetah",
                          "agent_conf": "2x3",
                          "agent_obsk": AGENT_OBSK,
                          "episode_limit": 1000},
    "6AgentHalfCheetah": {"scenario": "HalfCheetah",
                          "agent_conf": "6x1",
                          "agent_obsk": 1,
                          "episode_limit": 1000},
    "2AgentWalker2d": {"scenario": "Walker2d",
                          "agent_conf": "2x3",
                          "agent_obsk": AGENT_OBSK,
                          "episode_limit": 1000},
    "2AgentHumanoidStandup": {"scenario": "HumanoidStandup",
                              "agent_conf": "9|8",
                              "agent_obsk": AGENT_OBSK,
                              "episode_limit": 1000},
    "2AgentHumanoid": {"scenario": "Humanoid",
                       "agent_conf": "9|8",
                       "agent_obsk": AGENT_OBSK,
                       "episode_limit": 1000},
}

policy_mapping_dict = {
    "all_scenario": {
        "description": "mamujoco all scenarios",
        "team_prefix": ("agent_",),
        "all_agents_one_policy": True,
        "one_agent_one_policy": True,
    },
}


# maintained version of https://github.com/Farama-Foundation/Gymnasium-Robotics
class RLlibGymnasiumRoboticsMAMujoco(MultiAgentEnv):

    def __init__(self, env_config):
        self.env_config = env_args_dict[env_config["map_name"]]
        self.random_seed = env_config.get("seed", 0)
        map_name = env_config["map_name"]
        scenario_name = env_args_dict[map_name]["scenario"]
        partition = env_args_dict[map_name]["agent_conf"]
        agent_obsk = env_args_dict[map_name]["agent_obsk"]
        self.episode_limit = env_args_dict[map_name]["episode_limit"]

        self.render_mode = env_config.get("render_mode", None)
        self.env = parallel_env(scenario_name, agent_conf=partition, agent_obsk=agent_obsk,
                                render_mode=self.render_mode)
        
        agent0_space = self.env.action_spaces["agent_0"]
        self.action_space = GymBox(
            agent0_space.low[0], agent0_space.high[0],
            shape=(agent0_space.shape[0],), dtype=np.float32)
        self.state_space = self.env.single_agent_env.observation_space
        self.state_dim = self.state_space.shape[0]
        self.max_obs_dim = max([self.env.observation_spaces[agent].shape[0] for agent in self.env.agents])
        self.observation_space = GymDict({
            # "obs": Box(-10000, 10000.0, shape=(self.env.observation_spaces["agent_0"].shape[0],), dtype=np.float32),
            # "obs": Box(-10000, 10000.0, shape=(self.max_obs_dim,), dtype=np.float32),
            "obs": GymBox(-np.inf, np.inf, shape=(self.state_space.shape[0],), dtype=np.float32),
            "state": GymBox(-np.inf, np.inf, shape=(self.state_space.shape[0],), dtype=np.float32),
        })

        if "|" in self.env_config["agent_conf"]:
            self.num_agents = len(self.env_config["agent_conf"].split("|"))
        else:
            self.num_agents = int(self.env_config["agent_conf"].split("x")[0])

        self.agents = ["agent_{}".format(i) for i in range(self.num_agents)]
        self.step_count = 0

    def reset(self):
        self.step_count = 0
        o = self.env.reset(seed=self.random_seed)
        s = self.env.state().copy()
        obs = {}
        for agent_index, agent_name in enumerate(self.agents):
            o_ = np.zeros(self.max_obs_dim, dtype=np.float32)
            o_[:self.env.observation_spaces[agent_name].shape[0]] = np.float32(o[0][agent_name])
            obs[agent_name] = {
                # "obs": o_,
                "obs": np.float32(s), # following the env used in happo
                # "obs": np.float32(o[0][agent_name]),
                "state": np.float32(s),
            }
        return obs

    def step(self, action_dict):
        self.step_count += 1
        o, r, terminated, truncated, info = self.env.step(action_dict)
        s = self.env.state().copy()
        rewards = {}
        obs = {}
        for agent_index, agent_name in enumerate(self.agents):
            rewards[agent_name] = r[agent_name]
            o_ = np.zeros(self.max_obs_dim, dtype=np.float32)
            o_[:self.env.observation_spaces[agent_name].shape[0]] = np.float32(o[agent_name])
            obs[agent_name] = {
                # "obs": o_,
                "obs": np.float32(s), # following the env used in happo
                # "obs": np.float32(o[agent_name]),
                "state": np.float32(s),
            }
            if agent_name not in info:
                info[agent_name] = {}
            # Signal truncation (time limit) so the value function bootstraps
            # instead of treating it as a true terminal state with value 0.
            is_truncated = truncated.get(agent_name, False)
            is_terminated = terminated.get(agent_name, False)
            info[agent_name]["TimeLimit.truncated"] = is_truncated and not is_terminated

        any_terminated = any(terminated.values())
        any_truncated = any(truncated.values())
        dones = {"__all__": any_terminated or any_truncated}
        return obs, rewards, dones, info

    def close(self):
        self.env.close()

    def render(self, mode=None):
        frame = self.env.render()
        if self.render_mode == "rgb_array":
            return frame
        time.sleep(0.05)
        return True

    def get_env_info(self):
        env_info = {
            "space_obs": self.observation_space,
            "space_act": self.action_space,
            "num_agents": self.num_agents,
            "episode_limit": self.episode_limit,
            "policy_mapping_info": policy_mapping_dict,
        }
        return env_info
