#!/bin/bash
# mychpc batch to check available GPUs
srun --account=soc-gpu-np --partition=soc-gpu-np --qos=soc-gpu-np \
     --time=12:00:00 --ntasks=12 --mem=256G --gres=gpu:a100:1 \
     --pty /bin/bash

# srun --account=soc-gpu-np \
#      --partition=soc-gpu-np \
#      --time=00:30:00 \
#      --ntasks=1 \
#      --cpus-per-task=2 \
#      --mem=16G \
#      --gres=gpu:1 \
#      --pty /bin/bash