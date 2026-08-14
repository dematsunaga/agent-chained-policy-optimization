#!/bin/bash
# When a condor job is on hold, use 'condor_q -analyze {cid}.{pid}'

echo "hostname: `hostname`"
echo "PATH=$PATH"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "======================================"

export DISABLE_TQDM=TRUE
export WANDB_API_KEY=

cd /home/USER/Codes/MARLlib-master/condor-scripts
/home/USER/anaconda3/envs/marllib/bin/python main_condor_mamujoco.py --cid=$1 --pid=$2 --machine=`hostname` 
