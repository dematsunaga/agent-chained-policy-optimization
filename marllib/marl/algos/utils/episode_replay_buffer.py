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

import random
from ray.rllib.execution.replay_buffer import *
from ray.rllib.policy.sample_batch import SampleBatch


class EpisodeBasedReplayBuffer(LocalReplayBuffer):

    def __init__(
            self,
            num_shards: int = 1,
            learning_starts: int = 1000,
            capacity: int = 10000,
            replay_batch_size: int = 32,
            prioritized_replay_alpha: float = 0.6,
            prioritized_replay_beta: float = 0.4,
            prioritized_replay_eps: float = 1e-6,
            replay_mode: str = "independent",
            replay_sequence_length: int = 1,
            replay_burn_in: int = 0,
            replay_zero_init_states: bool = True,
            buffer_size=DEPRECATED_VALUE,
            n_step: int = 1,
            gamma: float = 0.99,
    ):
        LocalReplayBuffer.__init__(self, num_shards, learning_starts, capacity, replay_batch_size,
                                   prioritized_replay_alpha, prioritized_replay_beta,
                                   prioritized_replay_eps, replay_mode, replay_sequence_length, replay_burn_in,
                                   replay_zero_init_states,
                                   buffer_size)

        self.replay_batch_size = replay_batch_size
        self.n_step = n_step
        self.gamma = gamma
        self._gamma_powers = np.array([gamma ** k for k in range(n_step + 1)])

        # Episode tracking for n-step: a parallel array of episode IDs per
        # policy buffer slot.  Prevents n-step look-ahead from crossing episode
        # boundaries that lack a done flag (e.g. interleaved workers).
        self._ep_ids = {}       # {policy_id: list[int]}
        self._ep_counter = {}   # {policy_id: int}

    @override(LocalReplayBuffer)
    def add_batch(self, batch: SampleBatchType) -> None:
        # With num_workers=0 (local inference), no Plasma object store is
        # used, so there is no risk of pinning shared memory.  Skip the
        # deep-copy to avoid ~12 numpy array copies per env step.
        # Handle everything as if multiagent
        if isinstance(batch, SampleBatch):
            batch = MultiAgentBatch({DEFAULT_POLICY_ID: batch}, batch.count)

        with self.add_batch_timer:
            for policy_id, sample_batch in batch.policy_batches.items():
                # Slice into individual 1-step entries so the buffer stores
                # one transition per slot. Required when rollout_fragment_length>1.
                T = sample_batch.count
                slices = [sample_batch] if T == 1 else [sample_batch[i:i+1] for i in range(T)]
                for s in slices:
                    self.replay_buffers[policy_id].add(s, weight=None)
        self.num_added += batch.count

    @override(LocalReplayBuffer)
    def replay(self):
        if self.num_added < self.replay_starts:
            return None

        # We always sample the SAME random indices for every policy buffer
        # so that multi-agent data stays temporally aligned.  Without this,
        # each policy would sample independently, and before_learn_on_batch
        # would construct next_opponent_actions from mismatched timesteps
        # (different env steps for different agents).
        #
        # The parent's "lockstep" replay_mode uses a single _ALL_POLICIES
        # buffer which is incompatible with our per-policy add_batch, so
        # we implement index sharing here instead.
        with self.replay_timer:
            samples = {}
            shared_indices = None

            for policy_id, replay_buffer in self.replay_buffers.items():
                storage = replay_buffer._storage
                cur_size = len(storage)
                if cur_size == 0:
                    continue

                batch_size = min(self.replay_batch_size, cur_size)

                # First policy picks the indices; all others reuse them.
                if shared_indices is None:
                    shared_indices = random.sample(range(cur_size), batch_size)
                base_indices = shared_indices

                replay_buffer._num_timesteps_sampled += batch_size

                if self.n_step <= 1:
                    # Fast path: stack per key instead of 256-way
                    # concat_samples (avoids O(keys * batch_size) tiny
                    # np.concatenate calls).
                    first = storage[base_indices[0]]
                    batch_data = {}
                    for key in first.keys():
                        if key == SampleBatch.SEQ_LENS:
                            continue
                        if key == "infos":
                            # Infos are dicts; can't np.stack them.
                            # Collect as object array for QMIX group_rewards.
                            infos = np.empty(batch_size, dtype=object)
                            for i, idx in enumerate(base_indices):
                                infos[i] = storage[idx][key][0]
                            batch_data[key] = infos
                            continue
                        batch_data[key] = np.stack(
                            [storage[idx][key][0] for idx in base_indices])
                    batch_data[SampleBatch.SEQ_LENS] = np.ones(
                        batch_size, dtype=np.int32)
                    samples[policy_id] = SampleBatch(batch_data)
                    continue

                # N-step sampling: vectorised over the batch.
                # Pre-extract rewards/dones for all needed look-ahead
                # indices, then do a forward scan with numpy masks
                # instead of a Python loop over batch_size.
                eviction_started = replay_buffer._eviction_started
                write_ptr = replay_buffer._next_idx
                base_arr = np.array(base_indices)

                # Shared-policy agent stride: when share_policy="all",
                # N agents' transitions are interleaved in one buffer:
                #   a0_t0, a1_t0, ..., a(N-1)_t0, a0_t1, a1_t1, ...
                # The n-step look-ahead must stride by N to stay within
                # the same agent's trajectory. With individual policies
                # (one buffer per agent), stride=1 is correct.
                _stride = getattr(self, '_agent_stride', None)
                if _stride is None:
                    _has_ai = ("agent_index" in storage[0])
                    if _has_ai and cur_size > 1:
                        # Detect stride: starting from index 0, walk forward
                        # until we find the same agent_index again.
                        ref_aid = storage[0]["agent_index"][0]
                        _stride = 1
                        for _s in range(1, min(cur_size, 20)):
                            if storage[_s]["agent_index"][0] == ref_aid:
                                _stride = _s
                                break
                    else:
                        _stride = 1
                    self._agent_stride = _stride

                actual_n = np.ones(batch_size, dtype=np.int32)
                stopped = np.zeros(batch_size, dtype=bool)
                n_rewards = np.array(
                    [storage[i][SampleBatch.REWARDS][0]
                     for i in base_arr])

                for step in range(1, self.n_step):
                    # With agent stride, look-ahead offset = step * stride.
                    offset = step * _stride
                    # Frontier = base + (step-1)*stride (check done here).
                    frontier_offset = (step - 1) * _stride
                    if eviction_started:
                        frontier = (base_arr + frontier_offset) % cur_size
                    else:
                        frontier = base_arr + frontier_offset

                    # Extract dones at frontier for the whole batch.
                    frontier_dones = np.array(
                        [storage[f][SampleBatch.DONES][0]
                         if f < cur_size else True
                         for f in frontier], dtype=bool)
                    stopped |= frontier_dones

                    # Next index = base + step * stride.
                    if eviction_started:
                        nxt = (base_arr + offset) % cur_size
                        # With stride > 1, nxt could skip over write_ptr.
                        # Check if any index in [frontier+1..nxt] crosses
                        # write_ptr by comparing distances from base.
                        dist_to_wp = (write_ptr - base_arr) % cur_size
                        stopped |= (offset >= dist_to_wp) & (dist_to_wp > 0)
                    else:
                        nxt = base_arr + offset
                        stopped |= (nxt >= cur_size)

                    # Only accumulate for non-stopped samples.
                    active = ~stopped
                    if not np.any(active):
                        break
                    nxt_rewards = np.array(
                        [storage[n][SampleBatch.REWARDS][0]
                         if n < cur_size else 0.0
                         for n in nxt])
                    n_rewards[active] += (
                        self._gamma_powers[step] * nxt_rewards[active])
                    actual_n[active] += 1

                # last_indices[i] = base + actual_n[i] * stride - stride.
                # (actual_n includes the base step, so the last look-ahead
                # offset is (actual_n - 1) * stride.)
                last_offset = (actual_n - 1) * _stride
                if eviction_started:
                    last_indices = (base_arr + last_offset) % cur_size
                else:
                    last_indices = base_arr + last_offset

                # Gather all keys from storage into batch_data.
                first_item = storage[base_arr[0]]
                last_keys = {SampleBatch.NEXT_OBS, SampleBatch.DONES,
                             "new_state", "truncated"}
                batch_data = {}
                for key in first_item.keys():
                    if key == SampleBatch.SEQ_LENS or key == "infos":
                        continue
                    if key == SampleBatch.REWARDS:
                        continue
                    if key in last_keys:
                        batch_data[key] = np.stack(
                            [storage[last_indices[i]][key][0]
                             for i in range(batch_size)])
                    else:
                        batch_data[key] = np.stack(
                            [storage[base_arr[i]][key][0]
                             for i in range(batch_size)])
                batch_data[SampleBatch.REWARDS] = n_rewards
                batch_data["sp_gamma"] = self._gamma_powers[actual_n]
                batch_data[SampleBatch.SEQ_LENS] = np.ones(
                    batch_size, dtype=np.int32)
                samples[policy_id] = SampleBatch(batch_data)

        return MultiAgentBatch(
            samples, sum(s.count for s in samples.values()))
