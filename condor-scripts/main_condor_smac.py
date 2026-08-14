import os
import argparse
import socket
# import wandb
import datetime
import random

# cuda 10.2: 18, 36
# cuda 11.4: 20-29, 31, 33-35, 37, 38

RUN_DESCRIPTION = f'sensor_range_1' 

# seed
START_SEED = 100
NUM_RANDOM_SEEDS = 3

network = 'gru'

ALGOS = [ 'acppo', 'mappo','happo', 'hatrpo', 'qmix'] 


TASKS = ['zerg_5_vs_5', 'protoss_10_vs_11', 'terran_10_vs_11', 'protoss_5_vs_5']

TIMESTEPS_TOTAL = 10_000_000

SC2PATH = '/ext_hdd/USER/3rdparty/StarCraftII'

PROJECT_DIR = ''
WANDB_API_KEY=''

WANDB_DIR = ''
SAVE_DIR = "/tmp"

CHECKPOINT_PATH = '' 
CHECKPOINT_FREQ = 1_000_000

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cid", help="condor cluster id", default=0, type=int)
    parser.add_argument("--pid", help="condor process id", default=0, type=int)
    parser.add_argument("--machine", help="machine", default='', type=str)
    
    args = parser.parse_args()
    
    print('machine:', args.machine)

    conda_env = 'marllib'

    PYTHON_BIN = f'/home/USER/anaconda3/envs/{conda_env}/bin/python'


    # Default configuration
    cmds = []

    for seed in range(START_SEED, START_SEED + NUM_RANDOM_SEEDS):
    # dataset collection commands
        for task in TASKS:

            for algo in ALGOS:
                hparam_str = ""
                run_description = f'{args.cid}_{args.pid}'
                run_description += '_' + RUN_DESCRIPTION
             
                cmd = ( 
                f"{PYTHON_BIN} train.py "
                f"--algo {algo} "
                f"--env {'smacv2'} "
                f"--map {task} "
                f"--exp_name {run_description} "
                f"--local_dir {SAVE_DIR} "
                f"--cid {args.cid} "
                f"--pid {args.pid} "
                f"--seed {seed} "
                f"--machine {args.machine} "
                f"--hyperparam_source {'smac'} "
                f"--checkpoint_freq {CHECKPOINT_FREQ} "
                f"--timesteps_total {TIMESTEPS_TOTAL} "
                f"--wandb_entity {'acpo'} "
                f"--policy_type {network} "
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
        
        print(f'condor_submit {PROJECT_DIR}/condor-scripts/condor_smac.submit -queue {len(cmds)}', flush=True)
    else:
        print("Start running", flush=True)
        cmd = cmds[args.pid]
        run_dir = PROJECT_DIR
        os.chdir(f"{run_dir}")

        print(f"cd {run_dir}")
        print("pwd: ", os.getcwd())
        os.system(f"export WANDB_API_KEY={WANDB_API_KEY}; export SC2PATH={SC2PATH};" + cmd)
