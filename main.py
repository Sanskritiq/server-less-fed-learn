import pickle
import sys
import json
import os
import time
import random
from pathlib import Path
from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from copy import deepcopy
from typing import Dict, List, OrderedDict

import torch
import numpy as np
from rich.console import Console
from rich.progress import track

PROJECT_DIR = Path(__file__).parent.parent.parent.absolute()

sys.path.append(PROJECT_DIR.as_posix())

from src.utils.tools import (
    OUT_DIR,
    Logger,
    fix_random_seed,
    parse_config_file,
    trainable_params,
    get_optimal_cuda_device,
)
from src.utils.models import MODELS
from data.utils.datasets import DATASETS
from src.client.client import LoopClient
from src.loop.loop import LoopImprovement


def get_argparser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        "-m", "--model", type=str, default="lenet5", choices=MODELS.keys()
    )
    parser.add_argument(
        "-d", "--dataset", type=str, choices=DATASETS.keys(), default="cifar10"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-jr", "--join_ratio", type=float, default=0.1)
    parser.add_argument("-ge", "--global_epoch", type=int, default=100)
    parser.add_argument("-le", "--local_epoch", type=int, default=5)
    parser.add_argument("-fe", "--finetune_epoch", type=int, default=0)
    parser.add_argument("-tg", "--test_gap", type=int, default=100)
    parser.add_argument("--eval_test", type=int, default=1)
    parser.add_argument("--eval_val", type=int, default=0)
    parser.add_argument("--eval_train", type=int, default=0)
    parser.add_argument(
        "-op", "--optimizer", type=str, default="sgd", choices=["sgd", "adam"]
    )
    parser.add_argument("-lr", "--local_lr", type=float, default=1e-2)
    parser.add_argument("-mom", "--momentum", type=float, default=0.0)
    parser.add_argument("-wd", "--weight_decay", type=float, default=0.0)
    parser.add_argument("-vg", "--verbose_gap", type=int, default=10)
    parser.add_argument("-bs", "--batch_size", type=int, default=32)
    parser.add_argument("-v", "--visible", type=int, default=0)
    parser.add_argument("--straggler_ratio", type=float, default=0)
    parser.add_argument("--straggler_min_local_epoch", type=int, default=1)
    parser.add_argument("--external_model_params_file", type=str, default="")
    parser.add_argument("--use_cuda", type=int, default=1)
    parser.add_argument("--save_log", type=int, default=1)
    parser.add_argument("--save_model", type=int, default=0)
    parser.add_argument("--save_fig", type=int, default=1)
    parser.add_argument("--save_metrics", type=int, default=1)
    parser.add_argument("--viz_win_name", type=str, required=False)
    parser.add_argument("-cfg", "--config_file", type=str, default="")
    parser.add_argument("--check_convergence", type=int, default=1)
    return parser

if __name__=="__main__":
    parser = get_argparser()
    args = parser.parse_args()
    if args.config_file:
        args = parse_config_file(args.config_file, args)
    
    loop_improvement = LoopImprovement(args)
    loop_improvement.train()
    
    # initialize clients
    
    