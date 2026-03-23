#!/bin/bash
# mychpc batch to check available GPUs
srun --account=soc-gpu-kp --partition=soc-gpu-kp --qos=soc-gpu-kp \
     --time=01:00:00 --ntasks=12 --mem=256G --gres=gpu:p100:1 \
     --pty /bin/bash