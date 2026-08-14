import torch
from torch.distributions import Categorical, Normal
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch

import numpy as np
import scipy.signal


from ray.rllib.evaluation.postprocessing import compute_advantages
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_ops import convert_to_torch_tensor
from ray.rllib.policy.sample_batch import SampleBatch
from marllib.marl.algos.utils.centralized_Q import get_dim
from marllib.marl.algos.utils.mixing_Q import align_batch
from gym.spaces import Discrete, MultiDiscrete, Box
from ray.rllib.utils.torch_ops import FLOAT_MIN

def acpo_postprocessing(policy: Policy,
                      sample_batch: SampleBatch,
                      other_agent_batches: dict = None,
                      episode=None) -> SampleBatch:
    """PPO postprocessing with per-agent predecessor beliefs.

    Builds the belief over predecessors' actions, computes the centralised value,
    then runs the agent-chained GAE.

    Env config:
                             | smacv2   | rware    | gym_mamujoco
        mask_flag            | True     | False    | False
        global_state_flag    | True     | False    | True
        opp_action_in_cc     | False    | False    | False
        joint_obs_flag       | False    | False    | False
        use_belief_networks  | True     | True     | False

    Belief source, first match wins:
        prev_agent_policy_access -> read the predecessors' nets directly
        use_belief_networks      -> learned belief net
        otherwise                -> model.forward_om

    MAMuJoCo runs use gymnasium_mamujoco; legacy mamujoco.yaml defines no
    joint_obs_flag and would KeyError below.
    """
    custom_config = policy.config["model"]["custom_model_config"]
    pytorch = custom_config["framework"] == "torch"
    obs_dim = get_dim(custom_config["space_obs"]["obs"].shape)


    opp_action_in_cc = custom_config["opp_action_in_cc"] # False for rware/smacv2
    global_state_flag = custom_config["global_state_flag"] # True for smacv2, False for rware
    mask_flag = custom_config["mask_flag"] # True for smacv2, False for rware/mamj
    joint_obs_flag = custom_config["joint_obs_flag"] # False for rware/smacv2/mamj

    vf_use_prev_agent_policies = custom_config["vf_use_prev_agent_policies"]
    use_agent_chained_gae = custom_config["use_agent_chained_gae"]
    use_prev_agent_policies = custom_config["use_prev_agent_policies"]

    prev_agent_policy_access = custom_config.get("prev_agent_policy_access", False)
    use_belief_networks = custom_config.get("use_belief_networks", False)

    # width of the action-mask prefix in obs; 0 when there is no mask
    if mask_flag:
        action_mask_dim = custom_config["space_act"].n
    else:
        action_mask_dim = 0

    n_agents = custom_config["num_agents"]
    opponent_agents_num = n_agents - 1
    # separate per-agent policies (share_policy=individual): agent i gets i
    # predecessor slots instead of a padded N-1, and no agent-id one-hot
    ac_agent_id = custom_config.get("ac_agent_id", None)
    variable_width = (
        ac_agent_id is not None
        and (use_prev_agent_policies or vf_use_prev_agent_policies )
    )

    # policy built yet?  if not, RLlib is tracing and we just need zeros of the
    # right shape.  condition kept consistent with mappo postprocessing
    if (pytorch and hasattr(policy, "compute_central_vf")) or \
            (not pytorch and policy.loss_initialized()):
        initialized = True
        my_id = sample_batch["agent_index"][0]
        # hand this agent direct refs to its predecessors' nets
        if prev_agent_policy_access and other_agent_batches and hasattr(policy, '_ensure_initialized'):
            policy._ensure_initialized()
            if not getattr(policy, '_predecessor_policies_wired', False):
                all_policies = {}
                for agent_name_key, (other_pol, _) in other_agent_batches.items():
                    idx = custom_config['agent_name_ls'].index(agent_name_key)
                    all_policies["policy_{}".format(idx)] = other_pol
                all_policies["policy_{}".format(my_id)] = policy
                policy._wire_predecessor_policies(all_policies)
        # build the belief over predecessors.  all four flags off = no chaining
        if use_prev_agent_policies or vf_use_prev_agent_policies:

            # assert my_id == sample_batch["agent_index"][-1]
            # assert my_id == sample_batch["agent_index"].mean()
            space_act = custom_config["space_act"]
            if isinstance(space_act, Box):
                action_dim = space_act.shape[0] 
            elif isinstance(space_act, Discrete):
                action_dim = space_act.n

            n_pred = my_id if variable_width else (n_agents - 1)

            # slot stride: mean + logstd for Box, one prob per action for Discrete
            if isinstance(custom_config["space_act"], Box):
                policy_dim_per_agent = action_dim * 2
            else:
                policy_dim_per_agent = action_dim

            # the belief itself; slots for predecessors >= my_id stay zero
            if use_prev_agent_policies or vf_use_prev_agent_policies:
                prev_agent_policies = np.zeros(
                    (sample_batch["obs"].shape[0], n_pred * policy_dim_per_agent),
                    dtype=sample_batch["obs"].dtype,
                )
                # training targets for the belief net, used in acppo_loss
                if use_belief_networks:
                    predecessor_true_logits = np.zeros(
                        (sample_batch["obs"].shape[0], (n_agents - 1) * policy_dim_per_agent),
                        dtype=np.float32,
                    )
            # one slot per predecessor; agent 0 has none
            for i in range(my_id):
                action_offset = i * action_dim
                policy_offset = i * policy_dim_per_agent
                agent_name = custom_config['agent_name_ls'][i]

                # where predecessor i's distribution comes from
                if use_prev_agent_policies or vf_use_prev_agent_policies:
                    # predecessor's actual rollout output
                    if prev_agent_policy_access:
                        agent_k_batch = align_batch(
                            other_agent_batches[agent_name][1], sample_batch)
                        agent_i_logits = agent_k_batch["action_dist_inputs"]
                    # predict i's distribution from my own obs, and record i's
                    # true logits as the training target
                    elif use_belief_networks:
                        if not hasattr(policy, "_belief_net_initialized"):
                            policy._ensure_belief_net()
                        agent_k_batch = align_batch(
                            other_agent_batches[agent_name][1], sample_batch)
                        # my own obs out of the packed vector
                        if mask_flag:
                            obs_i_t = torch.tensor(
                                sample_batch['obs'][:, action_mask_dim:action_mask_dim + obs_dim],
                                dtype=torch.float32, device=policy.device)
                        else:
                            obs_i_t = torch.tensor(
                                sample_batch['obs'][:, :obs_dim] if global_state_flag else sample_batch["obs"],
                                dtype=torch.float32, device=policy.device)
                        with torch.no_grad():
                            agent_i_logits = policy.belief_net(obs_i_t, my_id, i)
                        # restrict to i's legal actions, taken from i's obs
                        if mask_flag and isinstance(custom_config["space_act"], Discrete):
                            pred_j_action_mask = torch.tensor(
                                agent_k_batch["obs"][:, :action_mask_dim],
                                dtype=torch.float32, device=policy.device)
                            pred_j_inf_mask = torch.clamp(torch.log(pred_j_action_mask), min=FLOAT_MIN)
                            agent_i_logits = agent_i_logits + pred_j_inf_mask
                        true_logits = agent_k_batch["action_dist_inputs"]
                        predecessor_true_logits[:, policy_offset:policy_offset + policy_dim_per_agent] = true_logits[:, :policy_dim_per_agent]
                    else:
                        # re-query the actor with predecessor i's id.  only
                        # meaningful under a shared policy -- with
                        # share_policy=individual forward_om drops agent_index
                        agent_i_batch = {}
                        if custom_config["model_arch_args"]["core_arch"] in ["gru", "lstm"]:
                            agent_i_batch["hidden_state"] = torch.tensor(sample_batch['state_out_0'], device=policy.device)
                        agent_i_batch["obs"] = torch.tensor( sample_batch['obs'][:, :obs_dim] if global_state_flag else sample_batch["obs"] , device=policy.device)
                        agent_i_batch["agent_index"] = torch.tensor(other_agent_batches[agent_name][1]["agent_index"], device=policy.device)
                        agent_i_batch["prev_agent_policies"] = torch.tensor(prev_agent_policies, device=policy.device)
                        agent_i_logits = policy.model.forward_om(agent_i_batch)

                    if isinstance(custom_config["space_act"], Discrete):
                        agent_i_policies = torch.nn.functional.softmax(torch.tensor(agent_i_logits), dim=-1)
                        prev_agent_policies[:, policy_offset:policy_offset + action_dim] = agent_i_policies.detach().cpu().numpy()
                    elif isinstance(custom_config["space_act"], MultiDiscrete):
                        raise NotImplementedError
                    elif isinstance(custom_config["space_act"], Box):
                        if prev_agent_policy_access:
                            prev_agent_policies[:, policy_offset:policy_offset + action_dim * 2] = agent_i_logits[:, :action_dim * 2]
                        else:
                            agent_i_policies = torch.tensor(agent_i_logits)
                            prev_agent_policies[:, policy_offset:policy_offset + action_dim * 2] = agent_i_policies.detach().cpu().numpy()
            
            if use_prev_agent_policies or vf_use_prev_agent_policies:
                sample_batch["prev_agent_policies"] = prev_agent_policies
                if use_belief_networks:
                    sample_batch["predecessor_true_logits"] = predecessor_true_logits

                # set 'obs', 'actions', 'rewards', 'dones', 'infos', 'eps_id', 'unroll_id', 'agent_index', 'vf_preds', 'action_dist_inputs', 'action_logp'
        # belief input for the critic, shared by both branches below
        prev_agent_inputs = None
        if vf_use_prev_agent_policies:
            prev_agent_inputs = convert_to_torch_tensor(
                sample_batch["prev_agent_policies"], policy.device)
        # agent-id one-hot; constant under separate policies, so skipped there
        if prev_agent_inputs is not None and not variable_width:
            agent_id_inputs = torch.nn.functional.one_hot(
                torch.tensor(sample_batch["agent_index"]).long(),
                num_classes=n_agents).to(policy.device)
            prev_agent_inputs = torch.cat(
                [prev_agent_inputs, agent_id_inputs], dim=-1)

        # build the state and run the centralised critic.  smacv2/gym_mamujoco
        # have a global state in obs and need no opponent actions
        if not opp_action_in_cc and global_state_flag:
            # obs is [mask | obs | state]; drop the mask and our own obs
            sample_batch["state"] = sample_batch['obs'][:, action_mask_dim + obs_dim:]
            sample_batch[SampleBatch.VF_PREDS] = policy.compute_central_vf(
                convert_to_torch_tensor(
                    sample_batch["state"], policy.device),
                None, # no opponent actions
                prev_agent_inputs
            ).cpu().detach().numpy()
        else:  # need opponent info
            # reached when opp_action_in_cc, or when there is no global state and
            # the state has to be stacked from every agent's obs (rware)
            assert other_agent_batches is not None
            opponent_batch_list = []
            for opp_agent_id in range(n_agents):
                if opp_agent_id != my_id:
                    agent_name = custom_config['agent_name_ls'][opp_agent_id]
                    opponent_batch_list.append(other_agent_batches[agent_name])
            
            # opponent_batch_list = list(other_agent_batches.values())
            raw_opponent_batch = [opponent_batch_list[i][1] for i in range(opponent_agents_num)]
            opponent_batch = []
            for one_opponent_batch in raw_opponent_batch:
                one_opponent_batch = align_batch(one_opponent_batch, sample_batch)
                opponent_batch.append(one_opponent_batch)

            # only opp_action_in_cc envs get here with a global state
            if global_state_flag:  # include self obs and global state
                if mask_flag:
                    sample_batch["state"] = sample_batch['obs'][:, action_mask_dim + obs_dim:]
                else:
                    # BUG: action_mask_dim is 0 here, so this slices to empty.
                    # should be obs[:, obs_dim:].  unreachable for our envs
                    sample_batch["state"] = sample_batch['obs'][:, obs_dim:-action_mask_dim]
            else:
                if joint_obs_flag:
                    # self obs is already joint obs
                    sample_batch["state"] = sample_batch['obs'][:, action_mask_dim:action_mask_dim + obs_dim]
                # rware: stack every agent's obs
                else:
                    # must stack in order for the consistency
                    # Build name->index mapping matching opponent_batch order
                    # (numerical agent order, skipping my_id).
                    opp_name_to_idx = {}
                    idx = 0
                    for oid in range(n_agents):
                        if oid != my_id:
                            opp_name_to_idx[custom_config['agent_name_ls'][oid]] = idx
                            idx += 1
                    state_batch_list = []
                    for agent_name in custom_config['agent_name_ls']:
                        if agent_name in other_agent_batches:
                            index = opp_name_to_idx[agent_name]
                            state_batch_list.append(
                                opponent_batch[index]["obs"][:, action_mask_dim:action_mask_dim + obs_dim])
                        else:
                            state_batch_list.append(sample_batch['obs'][:, action_mask_dim:action_mask_dim + obs_dim])
                    sample_batch["state"] = np.stack(state_batch_list, 1)

            # stored unconditionally but only read when opp_action_in_cc
            sample_batch["opponent_actions"] = np.stack(
                [opponent_batch[i]["actions"] for i in range(opponent_agents_num)],
                1)

            sample_batch[SampleBatch.VF_PREDS] = policy.compute_central_vf(
                convert_to_torch_tensor(
                    sample_batch["state"], policy.device),
                convert_to_torch_tensor(
                    sample_batch["opponent_actions"], policy.device) if opp_action_in_cc else None,
                prev_agent_inputs
            ) \
                .cpu().detach().numpy()

    else:
        # tracing batch: no real values yet, just zeros of the right shape.
        # also disables the chained GAE below
        initialized = False
        _fill_dummy_sample_batch(
            sample_batch, custom_config, obs_dim, n_agents,
            opponent_agents_num, variable_width)

    completed = sample_batch["dones"][-1]
    # bootstrap from the last state if the episode was cut by the time limit
    # rather than genuinely terminating
    truncated = sample_batch[SampleBatch.INFOS][-1].get("TimeLimit.truncated", False) if completed else False
    if completed and not truncated:
        last_r = 0.0
    else:
        last_r = sample_batch[SampleBatch.VF_PREDS][-1]

    # else falls back to plain per-agent PPO GAE, which is also what the tracing
    # batch needs since it has no real vf_preds to chain
    if use_agent_chained_gae and initialized:
        if "lambda" in policy.config:
            train_batch = compute_serialized_advantages(
                sample_batch,
                other_agent_batches,
                last_r,
                policy.config["gamma"],
                policy.config["lambda"],
                use_gae=policy.config["use_gae"],
                agent_name_ls=custom_config["agent_name_ls"]
            )
        else:
            train_batch = compute_serialized_advantages(
                rollout=sample_batch,
                other_agent_batches=other_agent_batches,
                last_r=0.0,
                gamma=policy.config["gamma"],
                use_gae=False,
                use_critic=False,
                agent_name_ls=custom_config["agent_name_ls"]
            )
    else:
        
        if "lambda" in policy.config:
            train_batch = compute_advantages(
                sample_batch,
                last_r,
                policy.config["gamma"],
                policy.config["lambda"],
                use_gae=policy.config["use_gae"])
        else:
            train_batch = compute_advantages(
                rollout=sample_batch,
                last_r=0.0,
                gamma=policy.config["gamma"],
                use_gae=False,
                use_critic=False)
            
    return train_batch


def _fill_dummy_sample_batch(sample_batch, custom_config, obs_dim, n_agents,
                             opponent_agents_num, variable_width):
    """Zero-fill sample_batch for RLlib's tracing pass, where only shapes matter.

    Mutates sample_batch in place; the caller owns `initialized`.
    """
    global_state_flag = custom_config["global_state_flag"]
    joint_obs_flag = custom_config["joint_obs_flag"]
    ac_agent_id = custom_config.get("ac_agent_id", None)
    use_prev_agent_policies = custom_config["use_prev_agent_policies"]
    vf_use_prev_agent_policies = custom_config["vf_use_prev_agent_policies"]
    use_belief_networks = custom_config.get("use_belief_networks", False)

    o = sample_batch[SampleBatch.CUR_OBS]
    # flat state vector, or stacked per-agent obs when there is no global state
    if global_state_flag:
        sample_batch["state"] = np.zeros(
            (o.shape[0], get_dim(custom_config["space_obs"]["state"].shape)),
            dtype=sample_batch[SampleBatch.CUR_OBS].dtype)
    else:
        if joint_obs_flag:
            sample_batch["state"] = np.zeros(
                (o.shape[0], obs_dim),
                dtype=sample_batch[SampleBatch.CUR_OBS].dtype)
        else:
            sample_batch["state"] = np.zeros(
                (o.shape[0], n_agents, obs_dim),
                dtype=sample_batch[SampleBatch.CUR_OBS].dtype)

    sample_batch["vf_preds"] = np.zeros_like(
        sample_batch[SampleBatch.REWARDS], dtype=np.float32)
    sample_batch["opponent_actions"] = np.stack(
        [np.zeros_like(sample_batch["actions"], dtype=sample_batch["actions"].dtype)
         for _ in range(opponent_agents_num)], axis=1)
    if isinstance(custom_config["space_act"], Discrete):
        action_dim = sample_batch["action_dist_inputs"].shape[1]
    elif isinstance(custom_config["space_act"], Box):
        action_dim = custom_config["space_act"].shape[0]

    n_pred_init = ac_agent_id if variable_width else (n_agents - 1)
    if isinstance(custom_config["space_act"], Box):
        prev_policy_dim = n_pred_init * action_dim * 2
    else:
        prev_policy_dim = n_pred_init * action_dim

    if use_prev_agent_policies or vf_use_prev_agent_policies:
        sample_batch["prev_agent_policies"] = np.zeros(
            (sample_batch["obs"].shape[0], prev_policy_dim),
            dtype=sample_batch["obs"].dtype,
        )
        if use_belief_networks:
            sample_batch["predecessor_true_logits"] = np.zeros(
                (sample_batch["obs"].shape[0], prev_policy_dim),
                dtype=np.float32,
            )



def compute_serialized_advantages(rollout: SampleBatch,
                        other_agent_batches: dict,
                       last_r: float,
                       gamma: float = 0.9,
                       lambda_: float = 1.0,
                       use_gae: bool = True,
                       use_critic: bool = True,
                       agent_name_ls: list = None):
    """
    Given a rollout, compute its value targets and the advantages.

    Args:
        rollout (SampleBatch): SampleBatch of a single trajectory.
        last_r (float): Value estimation for last observation.
        gamma (float): Discount factor.
        lambda_ (float): Parameter for GAE.
        use_gae (bool): Using Generalized Advantage Estimation.
        use_critic (bool): Whether to use critic (value estimates). Setting
            this to False will use 0 as baseline.

    Returns:
        SampleBatch (SampleBatch): Object with experience from rollout and
            processed rewards.
    """

    assert SampleBatch.VF_PREDS in rollout or not use_critic, \
        "use_critic=True but values not found"
    assert use_critic or not use_gae, \
        "Can't use gae without using a value function"
    agent_id = rollout["agent_index"][0]

    n_agents = len(other_agent_batches) + 1
    # only the last agent does the work, since it needs every agent's vf_preds.
    # the earlier agents get their advantages via the write-back below
    if agent_id < n_agents - 1:
        # only compute advantages after all agents have finished computing values
        return rollout
    
    # list of size N-1, each element is size (H,)
    # other_agent_vf_preds = [batch[1][SampleBatch.VF_PREDS] for batch in other_agent_batches.values()]
    other_agent_vf_preds = [] # to prevent weird ordering of agents e.g. 10 agents: [0, 1, 10, 2,...]

    for opp_agent_id in range(n_agents):
        if opp_agent_id != agent_id:
            agent_name = agent_name_ls[opp_agent_id]
            other_agent_vf_preds.append(other_agent_batches[agent_name][1][SampleBatch.VF_PREDS])

    # list of is size N, each element is size (H,)
    vf_preds_serialized = np.stack(other_agent_vf_preds + [rollout[SampleBatch.VF_PREDS]], axis=1).flatten()
    
    # rewards are 0s for micro-steps and the Nth agent gets the actual rewards
    rewards_serialized = np.zeros_like(vf_preds_serialized, dtype=np.float32)
    rewards_serialized[n_agents - 1::n_agents] = rollout[SampleBatch.REWARDS]
    # append last_r
    vf_preds_serialized = np.concatenate(
        [vf_preds_serialized, np.array([last_r], dtype=np.float32)])
    
    # N micro-steps must compound to one env step of discounting
    gamma_serialized = gamma ** (1 / n_agents)
    lambda_serialized = lambda_

    # note: shadows the outer agent_id, which is not used again
    for agent_id in range(n_agents):
        if use_gae:
            delta_t = (
                    rewards_serialized[agent_id:] + gamma_serialized * vf_preds_serialized[agent_id + 1:] - vf_preds_serialized[agent_id:-1])
            advantage = discount_cumsum(
                    delta_t, gamma_serialized * lambda_serialized)
            if agent_id == n_agents - 1:
                rollout[Postprocessing.ADVANTAGES] = advantage[::n_agents]
                
                rollout[Postprocessing.VALUE_TARGETS] = (
                    rollout[Postprocessing.ADVANTAGES] +
                    rollout[SampleBatch.VF_PREDS]).astype(np.float32)
                rollout[Postprocessing.ADVANTAGES] = rollout[Postprocessing.ADVANTAGES].astype(np.float32)
            else:
                # write back into the predecessors' own batches.
                # assumes agents are named 'agent_i'; agent_name_ls[agent_id]
                # would be the consistent lookup
                agent_id_str = 'agent_' + str(agent_id)
                other_agent_batches[agent_id_str][1][Postprocessing.ADVANTAGES] = advantage[::n_agents]
                
                other_agent_batches[agent_id_str][1][Postprocessing.VALUE_TARGETS] = (
                    other_agent_batches[agent_id_str][1][Postprocessing.ADVANTAGES] +
                    other_agent_batches[agent_id_str][1][SampleBatch.VF_PREDS]).astype(np.float32)
                
                other_agent_batches[agent_id_str][1][Postprocessing.ADVANTAGES] = other_agent_batches[agent_id_str][1][Postprocessing.ADVANTAGES].astype(np.float32)
    # guard: without gae we would return a rollout with no advantages set
    if not use_gae:
        raise NotImplementedError("TODO")

    return rollout
def discount_cumsum(x: np.ndarray, gamma: float) -> np.ndarray:
    """Calculates the discounted cumulative sum over a reward sequence `x`.

    y[t] - discount*y[t+1] = x[t]
    reversed(y)[t] - discount*reversed(y)[t-1] = reversed(x)[t]

    Args:
        gamma (float): The discount factor gamma.

    Returns:
        np.ndarray: The sequence containing the discounted cumulative sums
            for each individual reward in `x` till the end of the trajectory.

    Examples:
        >>> x = np.array([0.0, 1.0, 2.0, 3.0])
        >>> gamma = 0.9
        >>> discount_cumsum(x, gamma)
        ... array([0.0 + 0.9*1.0 + 0.9^2*2.0 + 0.9^3*3.0,
        ...        1.0 + 0.9*2.0 + 0.9^2*3.0,
        ...        2.0 + 0.9*3.0,
        ...        3.0])
    """
    return scipy.signal.lfilter([1], [1, float(-gamma)], x[::-1], axis=0)[::-1]