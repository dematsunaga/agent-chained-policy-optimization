from marllib import marl
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cid", help="condor cluster id", default=0, type=int)
    parser.add_argument("--pid", help="condor process id", default=0, type=int)
    parser.add_argument("--machine", help="machine", default='', type=str)
    parser.add_argument("--seed", help="random seed", default=0, type=int)
    parser.add_argument("--env", help="environment name", default='overcooked', type=str)
    parser.add_argument("--map", help="map name", default='asymmetric_advantages', type=str)
    parser.add_argument("--algo", help="algorithm name", default='acppo', type=str)
    parser.add_argument("--hyperparam_source", help="test/$ENV/common", default='common', type=str)
    parser.add_argument("--local_mode", help="run in local mode", action='store_true', default=False)
    parser.add_argument("--local_dir", help="local directory", default='', type=str)
    parser.add_argument("--exp_name", help="experiment name", default='test', type=str)
    parser.add_argument("--checkpoint_freq", help="checkpoint frequency", default=1000, type=int)
    parser.add_argument("--timesteps_total", help="total timesteps", default=10000000, type=int)
    parser.add_argument("--wandb_entity", help="wandb entity", default='acpo', type=str)
    parser.add_argument("--share_policy", help="share policy", default='group', type=str)
    parser.add_argument("--policy_type", help="policy type: gru or mlp", default='mlp', type=str)

    # acppo specific
    parser.add_argument("--use_agent_chained_gae", help="use agent chained GAE", action='store_true', default=False)

    parser.add_argument("--vf_use_prev_agent_policies", help="use previous agent actions for the value function", action='store_true', default=False)
    parser.add_argument("--use_prev_agent_policies", help="use previous agent actions", action='store_true', default=False)

    parser.add_argument("--encode_layer", help="encode layers", default="128-256", type=str)
    parser.add_argument("--rnn_hidden_size", help="RNN hidden size", default=64, type=int)

    args = parser.parse_args()
    
    # prepare env
    if args.env == 'rware':
        map = args.map.split("-")
        map_size = map[0]
        n_agents = int(map[1])
        difficulty = map[2]
        env = marl.make_env(environment_name=args.env, map_name=args.map, force_coop=True,
                            map_size=map_size, n_agents=n_agents, difficulty=difficulty)

    else:
        env = marl.make_env(environment_name=args.env, map_name=args.map, force_coop=True)

    # initialize algorithm with appointed hyper-parameters
    if args.algo == 'acppo':
        algo = marl.algos.acppo(hyperparam_source=args.hyperparam_source,
                                use_agent_chained_gae=True,
                                vf_use_prev_agent_policies=True,
                                use_prev_agent_policies=True)
    elif args.algo == 'phi_mappo':
        algo = marl.algos.acppo(hyperparam_source=args.hyperparam_source,
                                use_agent_chained_gae=False,
                                vf_use_prev_agent_policies=False,
                                use_prev_agent_policies=True)
    elif args.algo == 'happo':
        algo = marl.algos.happo(hyperparam_source=args.hyperparam_source)
    elif args.algo == 'mappo':
        algo = marl.algos.mappo(hyperparam_source=args.hyperparam_source)
    elif args.algo == 'qmix':
        algo = marl.algos.qmix(hyperparam_source=args.hyperparam_source)
    elif args.algo == 'hatrpo':
        algo = marl.algos.hatrpo(hyperparam_source=args.hyperparam_source)

    
    if args.env == 'smacv2':
        encode_layer = "64"
    else:
        encode_layer = args.encode_layer
    # build agent model based on env + algorithms + user preference
    if args.policy_type in ['gru', 'lstm']:
        model = marl.build_model(env, algo, {"core_arch": args.policy_type, "hidden_state_size": args.rnn_hidden_size, "encode_layer": encode_layer})
    else:
        model = marl.build_model(env, algo, {"core_arch": args.policy_type, "encode_layer": encode_layer})

    # start training
    algo.fit(env, model, stop={'timesteps_total': args.timesteps_total}, share_policy=args.share_policy, local_mode=args.local_mode,
            local_dir=args.local_dir, checkpoint_freq=args.checkpoint_freq, wandb_entity=args.wandb_entity,
            exp_name=args.exp_name, seed=args.seed, cid=args.cid, pid=args.pid)
