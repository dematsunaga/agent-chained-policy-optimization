"""Periodic video rendering callback for RLlib trainers.

Creates a separate render environment (render_mode="rgb_array"), runs
deterministic evaluation episodes at fixed timestep intervals, and logs
the resulting videos to the active wandb run.

WandbLoggerCallback runs wandb in a subprocess, so wandb.log() from the
trainer process goes nowhere.  Instead we inject wandb.Video objects into
the RLlib result dict -- they flow through the queue to the wandb process
which calls wandb.log() on them.
"""

import numpy as np
from ray.rllib.agents.callbacks import DefaultCallbacks


def make_video_render_callback(render_interval=2_000_000, render_episodes=3):
    """Factory that returns a DefaultCallbacks subclass configured for video rendering.

    Args:
        render_interval: log videos every this many env timesteps.
        render_episodes: number of episodes to record per render event.
    """

    class VideoRenderCallback(DefaultCallbacks):

        def __init__(self):
            super().__init__()
            self._render_env = None
            self._last_render_step = 0
            self._render_interval = render_interval
            self._render_episodes = render_episodes

        def on_train_result(self, *, trainer, result, **kwargs):
            if self._render_interval <= 0:
                return
            timesteps = result.get("timesteps_total", 0)
            if timesteps - self._last_render_step < self._render_interval:
                return
            self._last_render_step = timesteps
            self._inject_videos(trainer, result, timesteps)

        def _inject_videos(self, trainer, result, cur_step):
            """Render episodes and inject wandb.Video objects into result dict.

            The result dict flows to WandbLoggerCallback which sends it
            through a multiprocessing queue to the wandb subprocess.
            Ray's _is_allowed_type accepts wandb.data_types.Video.
            """
            try:
                import wandb
            except ImportError:
                return

            print(f"[VideoRenderCallback] rendering {self._render_episodes} "
                  f"episodes at step {cur_step}")

            env = self._get_or_create_render_env(trainer)
            if env is None:
                return

            policy_mapping_fn = trainer.config["multiagent"]["policy_mapping_fn"]

            try:
                video_data = self._collect_videos(
                    trainer, env, policy_mapping_fn, cur_step)
                result.update(video_data)
            except RuntimeError as e:
                if "OpenGL" in str(e) or "Failed to initialize" in str(e):
                    print(f"[VideoRenderCallback] rendering skipped: {e}")
                else:
                    raise

        def _get_or_create_render_env(self, trainer):
            if self._render_env is not None:
                return self._render_env

            custom_config = trainer.config["model"]["custom_model_config"]
            env_name = custom_config.get("env", "")
            if "gymnasium_mamujoco" not in env_name:
                print(f"[VideoRenderCallback] unsupported env: {env_name}")
                return None

            from marllib.envs.base_env.gymnasium_mamujoco import (
                RLlibGymnasiumRoboticsMAMujoco,
            )

            map_name = custom_config["env_args"]["map_name"]
            env_args = {"map_name": map_name, "render_mode": "rgb_array"}
            try:
                self._render_env = RLlibGymnasiumRoboticsMAMujoco(env_args)
            except Exception as e:
                print(f"[VideoRenderCallback] failed to create render env: {e}")
                return None
            return self._render_env

        def _collect_videos(self, trainer, env, policy_mapping_fn, cur_step):
            """Run episodes, return dict of wandb.Video objects to merge into result."""
            import wandb

            video_data = {}
            for ep in range(self._render_episodes):
                obs = env.reset()
                done = {"__all__": False}
                frames = []
                total_reward = 0.0

                # MLP DDPG models have a dummy get_initial_state() that makes
                # RLlib treat them as recurrent.  compute_single_action with
                # state=None passes empty state_batches, crashing the model.
                # We must pass initial states explicitly and track them.
                states = {}
                for agent_id in env.agents:
                    policy_id = policy_mapping_fn(agent_id)
                    policy = trainer.get_policy(policy_id)
                    states[agent_id] = policy.get_initial_state()

                while not done["__all__"]:
                    actions = {}
                    for agent_id in env.agents:
                        policy_id = policy_mapping_fn(agent_id)
                        action, state, _ = trainer.compute_single_action(
                            obs[agent_id], state=states[agent_id],
                            policy_id=policy_id, explore=False,
                            full_fetch=True)
                        actions[agent_id] = action
                        states[agent_id] = state

                    obs, rewards, done, info = env.step(actions)
                    total_reward += sum(rewards.values())

                    frame = env.render()
                    if frame is not None and isinstance(frame, np.ndarray):
                        frames.append(frame)

                if len(frames) > 0:
                    # wandb.Video expects (T, C, H, W)
                    frames_array = np.array(frames)
                    frames_array = np.transpose(frames_array, (0, 3, 1, 2))
                    video_data[f"video/episode_{ep}"] = wandb.Video(
                        frames_array, fps=30, format="mp4")
                    video_data[f"video/episode_reward_{ep}"] = total_reward
                    video_data[f"video/step"] = cur_step

                print(f"[VideoRenderCallback] ep {ep}: "
                      f"reward={total_reward:.1f}, frames={len(frames)}")

            return video_data

    return VideoRenderCallback
