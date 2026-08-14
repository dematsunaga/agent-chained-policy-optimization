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

ALGOS =  ['acppo', 'happo', 'mappo', 'hatrpo'] 
TASKS = ['tiny-2-easy', 'tiny-4-medium','tiny-8-hard', 'small-2-easy', 'small-4-medium', 'small-8-hard', 'tiny-12-hard', 'small-12-hard']


PROJECT_DIR = ''
WANDB_API_KEY=''

WANDB_DIR = ''
SAVE_DIR = "/tmp"

CHECKPOINT_PATH = '' 
CHECKPOINT_FREQ = 5_000_000

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

    # dataset collection commands
    for task in TASKS:
        if task in ['small-8-hard','tiny-12-hard', 'tiny-10-hard', 'small-10-hard']:
            TIMESTEPS_TOTAL = 60_000_000
        elif task in ['small-12-hard']:
            TIMESTEPS_TOTAL = 100_000_000
        else:
            TIMESTEPS_TOTAL = 40_000_000

        for algo in ALGOS:

            hparam_str = ""
            run_description = f'{args.cid}_{args.pid}'
            run_description += '_' + RUN_DESCRIPTION
             
            for seed in range(START_SEED, START_SEED + NUM_RANDOM_SEEDS):                    
                cmd = ( 
                f"{PYTHON_BIN} train.py "
                f"--algo {algo} "
                f"--env {'rware'} "
                f"--map {task} "
                f"--exp_name {run_description} "
                f"--local_dir {SAVE_DIR} "
                f"--cid {args.cid} "
                f"--pid {args.pid} "
                f"--seed {seed} "
                f"--machine {args.machine} "
                f"--hyperparam_source {'rware'} "
                f"--checkpoint_freq {CHECKPOINT_FREQ} "
                f"--timesteps_total {TIMESTEPS_TOTAL} "
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
        
        print(f'condor_submit {PROJECT_DIR}/condor-scripts/condor_rware.submit -queue {len(cmds)}', flush=True)
    else:
        print("Start running", flush=True)
        cmd = cmds[args.pid]
        run_dir = PROJECT_DIR
        os.chdir(f"{run_dir}")

        print(f"cd {run_dir}")
        print("pwd: ", os.getcwd())
        os.system(f"export WANDB_API_KEY={WANDB_API_KEY};" + cmd)
