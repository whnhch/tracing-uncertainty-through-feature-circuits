#!/bin/bash
# mychpc batch to check available GPUs
srun --account=fariha --partition=granite-gpu --qos=granite-gpu-freecycle --exclude=grn008 \
     --time=12:00:00 --ntasks=12 --mem=256G --gres=gpu:l40s:1 --pty /bin/bash

# srun --account=fariha --partition=granite-gpu --qos=granite-gpu-freecycle \
#      --time=10:00:00 --ntasks=12 --mem=256G --gres=gpu:h100nvl:1 \
#      bash -c "./utopia/test/scripts/run_greedy_tuple.sh; exec /bin/bash"