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

from ray import tune
from ray.tune.utils import merge_dicts
from ray.tune import CLIReporter
from ray.rllib.models import ModelCatalog
from marllib.marl.algos.core.CC.maddpg import MADDPGTrainer
from marllib.marl.algos.utils.setup_utils import AlgVar
from marllib.marl.algos.utils.log_dir_util import available_local_dir
from marllib.marl.algos.scripts.coma import restore_model
import json
from typing import Any, Dict
from ray.tune.analysis import ExperimentAnalysis

from ray.tune.integration.wandb import WandbLoggerCallback
from ray.tune.logger import JsonLogger, CSVLogger
from marllib.marl.algos.utils.video_render_callback import make_video_render_callback

def run_maddpg(model: Any, exp: Dict, run: Dict, env: Dict,
             stop: Dict, restore: Dict) -> ExperimentAnalysis:
    """ This script runs the Multi-Agent Deep Deterministic Policy Gradient (MADDPG) algorithm using Ray RLlib.
    Args:
        :params model (str): The name of the model class to register.
        :params exp (dict): A dictionary containing all the learning settings.
        :params run (dict): A dictionary containing all the environment-related settings.
        :params env (dict): A dictionary specifying the condition for stopping the training.
        :params restore (bool): A flag indicating whether to restore training/rendering or not.

    Returns:
        ExperimentAnalysis: Object for experiment analysis.

    Raises:
        TuneError: Any trials failed and `raise_on_failed_trial` is True.
    """

    ModelCatalog.register_custom_model(
        "DDPG_Model", model)

    _param = AlgVar(exp)

    episode_limit = env["episode_limit"]
    train_batch_size = _param["batch_episode"]
    learning_starts = _param["learning_starts_episode"] * episode_limit
    buffer_size = _param["buffer_size_episode"] * episode_limit
    twin_q = _param["twin_q"]
    prioritized_replay = _param["prioritized_replay"]
    smooth_target_policy = _param["smooth_target_policy"]
    n_step = _param["n_step"]
    critic_lr = _param["critic_lr"]
    actor_lr = _param["actor_lr"]
    # Soft-update targets every training step.
    target_network_update_freq = 1
    tau = _param["tau"]
    batch_mode = _param["batch_mode"]
    back_up_config = merge_dicts(exp, env)
    back_up_config.pop("algo_args")  # clean for grid_search
    back_up_config["reward_centering"] = _param.get("reward_centering", False)
    back_up_config["reward_centering_rate"] = _param.get("reward_centering_rate", 0.001)

    arch = exp["model_arch_args"]["core_arch"]
    max_seq_len = 1 if arch == "mlp" else episode_limit

    config = {
        "seed": back_up_config["seed"],
        "batch_mode": batch_mode,
        "buffer_size": buffer_size,
        "train_batch_size": train_batch_size,
        "critic_lr": critic_lr if restore is None else 1e-10,
        "actor_lr": actor_lr if restore is None else 1e-10,
        "twin_q": twin_q,
        "prioritized_replay": prioritized_replay,
        "smooth_target_policy": smooth_target_policy,
        "tau": tau,
        "target_network_update_freq": target_network_update_freq,
        "learning_starts": learning_starts,
        "n_step": n_step,
        "model": {
            "max_seq_len": max_seq_len,
            "custom_model_config": back_up_config,
        },
        "zero_init_states": True,
        "use_huber": False,
        "train_interval": _param.get("train_interval", 50),
        "update_per_train": _param.get("update_per_train", 25),
        # Collect train_interval steps per store_op call so round_robin_weights=[1, update_per_train]
        # makes 1 store call (not train_interval calls) before each batch of gradient updates.
        "rollout_fragment_length": _param.get("train_interval", 50),
        "callbacks": make_video_render_callback(
            render_interval=10_000_000, render_episodes=3),
        # Fixed stddev=0.1 for all envs. Action-range scaling (0.1*scale)
        # under-explores on Humanoid [-0.4,0.4] (stddev=0.04 is too small).
        "exploration_config": {
            "type": "GaussianNoise",
            "random_timesteps": 10000,
            "stddev": 0.1,
            "initial_scale": 1.0,
            "final_scale": 1.0,
        },
    }
    config.update(run)

    config["timesteps_per_iteration"] = 100000
    config["evaluation_interval"] = 5

    algorithm = exp["algorithm"]
    map_name = exp["env_args"]["map_name"]
    RUNNING_NAME = back_up_config["exp_name"]+ '_' + '_'.join([algorithm, arch, map_name])
    model_path = restore_model(restore, exp)
    wandb_callback = WandbLoggerCallback(
        project = back_up_config["env"],
        entity = back_up_config["wandb_entity"], #wandb_entity,
        name = RUNNING_NAME,
        dir = available_local_dir if exp["local_dir"] == "" else exp["local_dir"],
        job_type = "training",
        resume = "allow",
        id = None
    )
    results = tune.run(MADDPGTrainer,
                       name=RUNNING_NAME,
                       checkpoint_at_end=exp['checkpoint_end'],
                       checkpoint_freq=exp['checkpoint_freq'],
                       restore=model_path,
                       stop=stop,
                       config=config,
                       verbose=0,
                       progress_reporter=CLIReporter(max_report_frequency=999999),
                       callbacks=[wandb_callback],
                       loggers=(JsonLogger,),
                       local_dir=available_local_dir if exp["local_dir"] == "" else exp["local_dir"])

    return results
