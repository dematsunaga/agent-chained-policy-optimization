#!/bin/bash
export SC2PATH=/ext_hdd/3rdparty/StarCraftII


TASKS=terran_5_vs_5

#ALGOS=('acppo' 'mappo' 'phi_mappo' 'happo' 'hatrpo' 'qmix')
ALGO=happo

GPU_IDS=(0 1 2)

SEEDS=(100)
for i in {1..2}
do
    while : ; do
        RAND_SEED=$(( RANDOM % 10000 ))
        if [[ ! " ${SEEDS[@]} " =~ " ${RAND_SEED} " ]]; then
            SEEDS+=($RAND_SEED)
            break
        fi
    done
done

echo "=== Start Experiments ==="
echo "GPU IDs: ${GPU_IDS[@]}"
echo "TASKS: $TASKS"
echo "ALGO: $ALGO"
echo "SEEDS: ${SEEDS[@]}"
echo "Date: $(date)"
echo "=================="

for IDX in 0 1 2
do
    GPU_ID=${GPU_IDS[$IDX]}
    SEED=${SEEDS[$IDX]}
    LOG_NAME=${ALGO}_${TASKS}_seed_$((IDX+1))
    LOG_FILE="/home/my_username/acpo/logs/${LOG_NAME}.log"
    echo "GPU $GPU_ID : seed_$((IDX+1)) (Seed: $SEED), Logs: $LOG_FILE"
    echo "$((IDX+1)) Starting..." 

     nohup env CUDA_VISIBLE_DEVICES=$GPU_ID python train_gpu.py --algo $ALGO --map $TASKS --seed $SEED --num_gpus 1 --num_workers 2 --num_gpus_per_worker 0 > $LOG_FILE 2>&1 &

done

# Wait for all background jobs to finish
wait

