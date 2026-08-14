import os
import argparse
import socket
# import wandb
import datetime
import random


RUN_DESCRIPTION = f'my_run'  # description for the run

# seed
START_SEED = 100
NUM_RANDOM_SEEDS = 3

ALGOS = ['hatrpo',  'mappo', 'happo', 'acppo'] # ,

TASKS = ['2AgentAnt', '4AgentAnt', '2AgentHalfCheetah', '6AgentHalfCheetah', '2AgentWalker2d',  '2AgentHumanoid']
TIMESTEPS_TOTAL = 10000000

PROJECT_DIR = ''
WANDB_API_KEY=''

WANDB_DIR = ''
SAVE_DIR = "/tmp"

CHECKPOINT_FREQ = 100000

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cid", help="condor cluster id", default=0, type=int)
    parser.add_argument("--pid", help="condor process id", default=0, type=int)
    parser.add_argument("--machine", help="machine", default='', type=str)
    
    args = parser.parse_args()
    
    print('machine:', args.machine)
    conda_env = 'acpo'

    PYTHON_BIN = f'/home/USER/anaconda3/envs/{conda_env}/bin/python'


    # Default configuration
    cmds = []

    for seed in range(START_SEED, START_SEED + NUM_RANDOM_SEEDS):
        # dataset collection commands
        for task in TASKS:
            if 'HalfCheetah' in task:
                hparam_source = "mamujoco_halfcheetah"
            elif 'Ant' in task:
                hparam_source = "mamujoco_ant"
            elif 'Walker2d' in task:
                hparam_source = "mamujoco_walker"
            elif 'Humanoid' in task:
                hparam_source = "mamujoco_humanoid"
            else:
                hparam_source = "mamujoco"
            for algo in ALGOS:
                hparam_str = ""
                run_description = f'{args.cid}_{args.pid}'
                run_description += '_' + RUN_DESCRIPTION
                                        
                cmd = ( 
                f"{PYTHON_BIN} train.py "
                f"--algo {algo} "
                f"--env {'gymnasium_mamujoco'} "
                f"--map {task} "
                f"--exp_name {run_description} "
                f"--local_dir {SAVE_DIR} "
                f"--cid {args.cid} "
                f"--pid {args.pid} "
                f"--seed {seed} "
                f"--machine {args.machine} "
                f"--hyperparam_source {hparam_source} "
                f"--checkpoint_freq {CHECKPOINT_FREQ} "
                f"--timesteps_total {TIMESTEPS_TOTAL} "
                f"--share_policy {'individual'} "
                f"--policy_type {'mlp'} "
                f"{hparam_str} "
                )
                cmds.append(cmd)

    
    for pid, cmd in enumerate(cmds):
        print(f'[{pid}] {cmd}', flush=True)
    print('==================', flush=True)
    print(args.pid, cmds[args.pid], flush=True)
    print('==================', flush=True)

    if args.cid == 0:
        print('Run:', flush=True)
        run_dir = os.path.join(PROJECT_DIR, 'examples')
        os.chdir(f"{run_dir}")
        print(f"cd {run_dir}")
        print("pwd: ", os.getcwd())
        
        print(f'condor_submit {PROJECT_DIR}/condor-scripts/condor_mamujoco.submit -queue {len(cmds)}', flush=True)
    else:
        print("Start running", flush=True)
        cmd = cmds[args.pid]
        run_dir = PROJECT_DIR
        os.chdir(f"{run_dir}")

        print(f"cd {run_dir}")
        print("pwd: ", os.getcwd())

        os.system(f"export WANDB_API_KEY={WANDB_API_KEY};" + cmd)
