import os
import random
import numpy as np
import torch
import argparse

def reset_random_seeds(n=1):
    os.environ['PYTHONHASHSEED'] = str(n)
    # tf.random.set_seed(n)
    torch.random.manual_seed(n)
    np.random.seed(n)
    random.seed(n)

def get_wandb_args(path):
    import wandb
    api = wandb.Api()
    run = api.run(path)
    args = argparse.Namespace(**run.config)
    exec('args.torch_int = ' + args.torch_int)

    if args.default_float == 'float32':
        torch.set_default_dtype(torch.float32)
    elif args.default_float == 'float64':
        torch.set_default_dtype(torch.float64)
    else:
        raise ValueError('default float type not supported')

    args.wandb_id = run.id
    args.name = run.name
    return args

