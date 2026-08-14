#!/bin/bash
# When a condor job is on hold, use 'condor_q -analyze {cid}.{pid}'

echo "hostname: `hostname`"
echo "PATH=$PATH"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "======================================"

export DISABLE_TQDM=TRUE
export WANDB_API_KEY= 


export SC2PATH=/ext_hdd/USER/3rdparty/StarCraftII
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:/usr/lib64:$SC2PATH/Libs:${LD_LIBRARY_PATH:-}"

cd /home/USER/Codes/MARLlib-master/condor-scripts
/home/USER/anaconda3/envs/marllib/bin/python main_condor_smac.py --cid=$1 --pid=$2 --machine=`hostname` #--wandb_run_id=$3
