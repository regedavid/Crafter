#!/bin/bash

export OMP_NUM_THREADS=2 
EXP_NAME="nature_cnn_baseline"

for seed in 1 2 3; do
    echo "Starting run: $EXP_NAME | Seed: $seed"
    python train.py --exp_name $EXP_NAME --seed $seed &
done

wait
echo "All runs for $EXP_NAME completed!"