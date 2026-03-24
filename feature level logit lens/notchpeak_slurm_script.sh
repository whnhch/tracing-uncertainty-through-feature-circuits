#!/bin/bash
# mychpc batch to check available GPUs
srun --account=soc-gpu-np --partition=soc-gpu-np --qos=soc-gpu-np \
     --time=00:30:00 --ntasks=2 --mem=32G --gres=gpu:1 \
     --pty /bin/bash

# srun --account=notchpeak-gpu \
#      --partition=notchpeak-gpu \
#      --time=00:30:00 \
#      --ntasks=1 \
#      --cpus-per-task=2 \
#      --mem=32G \
#      --gres=gpu:1 \
#      --pty /bin/bash