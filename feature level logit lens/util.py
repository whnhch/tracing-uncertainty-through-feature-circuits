import os
import random
import torch
from pathlib import Path

import sys

def str2bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        v_lower = v.lower()
        if v_lower in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v_lower in ('no', 'false', 'f', 'n', '0'):
            return False

def set_environment(transformer_path, hf_token_path=None):
    project_root = Path().resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    os.environ['TRANSFORMERS_CACHE'] = transformer_path
    os.environ['HF_HOME'] = transformer_path
    os.environ['HF_DATASETS_CACHE'] = transformer_path
    
    if hf_token_path:
        with open(hf_token_path, "r") as f:
            hf_token = f.read().strip() 
        os.environ["HF_TOKEN"] = hf_token
        print("HF_TOKEN has been set:", os.environ["HF_TOKEN"])

def set_random_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)