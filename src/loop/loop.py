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
from typing import Dict, List, OrderedDict, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
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
    transfer_weights_without_last_module,
)
from src.utils.models import DecoupledModel
from src.client.client import LoopClient
from src.utils.models import MODELS
from data.utils.datasets import DATASETS

'''
PSEUDOCODE:

Algorithm 1: Loop Improvement (LI)
Data: Total number of clients C, data loaders
{Dc} for each client c
Result: Updated backbone parameters θbackbone,
head parameters {θheadc
} for each client c
1 Initialize shared backbone parameters θbackbone;
2 for each client c in 1 to C do
3 Initialize head parameters θheadc
;
4 end
5 Function Train client (c, θbackbone, θheadc ,
Dc, Ehead, Ebackbone, Efull):
// Train head while backbone is frozen
6 Freeze parameters in θbackbone;
7 Unfreeze parameters in θheadc
;
8 Train θheadc
for Ehead epochs using Dc;
// Train backbone while head is frozen
9 Unfreeze parameters in θbackbone;
10 Freeze parameters in θheadc
;
11 Train θbackbone for Ebackbone epochs using Dc;
// Optional: Train full model
12 Unfreeze all parameters;
13 Train θbackbone and θheadc
for Efull epochs using
Dc;
14 return
15 for round in 1 to R do
16 for each client c in 1 to C do
17 θbackbone ← Receive backbone parameters
from previous client;
18 Train client(c, θbackbone, θheadc
, Dc,
Ehead, Ebackbone, Efull);
19 Send θbackbone to the next client;
20 end
21 end
'''

class LoopImprovement:
    def __init__(
        self,
        args: Namespace,
    ):
        self.args = args
        self.unique_model = False
        self.default_trainer = True
        if len(self.args.config_file) > 0 and os.path.exists(self.args.config_file):
            self.args = parse_config_file(self.args)
        fix_random_seed(self.args.seed)
        begin_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        self.output_dir = OUT_DIR / begin_time
        with open(PROJECT_DIR / "data" / self.args.dataset / "args.json", "r") as f:
            self.args.dataset_args = json.load(f)
        
        # get client party information   
        try:
            partition_path = PROJECT_DIR / "data" / self.args.dataset / "partition.pkl"
            with open(partition_path, "rb") as f:
                partition = pickle.load(f)
        except:
            raise FileNotFoundError(f"Please partition {args.dataset} first.")
        
        self.train_clients: List[int] = partition["separation"]["train"]
        self.test_clients: List[int] = partition["separation"]["test"]
        self.val_clients: List[int] = partition["separation"]["val"]
        self.client_num: int = partition["separation"]["total"]
        
        # init model(s) parameters
        self.device = get_optimal_cuda_device(self.args.use_cuda)
        
        self.model = MODELS[self.args.model](dataset=self.args.dataset).to(self.device)
        self.model.check_avaliability()
        
        self.client_trainable_params: List[List[torch.Tensor]] = None
        self.global_params_dict: OrderedDict[str, torch.Tensor] = None

        random_init_params, self.trainable_params_name = trainable_params(
            self.model, detach=True, requires_name=True
        )
        self.global_params_dict = OrderedDict(
            zip(self.trainable_params_name, random_init_params)
        )
        if (
            not self.unique_model
            and self.args.external_model_params_file
            and os.path.isfile(self.args.external_model_params_file)
        ):
            # load pretrained params
            self.global_params_dict = torch.load(
                self.args.external_model_params_file, map_location=self.device
            )
        else:
            self.client_trainable_params = [
                trainable_params(self.model, detach=True) for _ in self.train_clients
            ]

        # system heterogeneity (straggler) setting
        self.clients_local_epoch: List[int] = [self.args.local_epoch] * self.client_num
        if (
            self.args.straggler_ratio > 0
            and self.args.local_epoch > self.args.straggler_min_local_epoch
        ):
            straggler_num = int(self.client_num * self.args.straggler_ratio)
            normal_num = self.client_num - straggler_num
            self.clients_local_epoch = [self.args.local_epoch] * (
                normal_num
            ) + random.choices(
                range(self.args.straggler_min_local_epoch, self.args.local_epoch),
                k=straggler_num,
            )
            random.shuffle(self.clients_local_epoch)
            
        # To make sure all algorithms run through the same client sampling stream.
        # Some algorithms' implicit operations at client side may disturb the stream if sampling happens at each FL round's beginning.
        self.client_sample_stream = [
            self.train_clients
            for _ in range(self.args.global_epoch)
        ]
        self.selected_clients: List[int] = []
        self.current_epoch = 0
        # epoch that need test model on test clients.
        self.epoch_test = [
            epoch
            for epoch in range(0, self.args.global_epoch)
            if (epoch + 1) % self.args.test_gap == 0
        ]
        
        # logging
        # variables for logging
        if not os.path.isdir(self.output_dir) and (
            self.args.save_log or self.args.save_fig or self.args.save_metrics
        ):
            os.makedirs(self.output_dir, exist_ok=True)

        if self.args.visible:
            from visdom import Visdom

            self.viz = Visdom()
            if self.args.viz_win_name is not None:
                self.viz_win_name = self.args.viz_win_name
            else:
                self.viz_win_name = (
                    f"_{self.args.dataset}"
                    + f"_{self.args.global_epoch}"
                    + f"_{self.args.local_epoch}"
                )
        self.client_stats = {i: {} for i in self.train_clients}
        self.metrics = {
            "before": {
                "train": {"accuracy": []},
                "val": {"accuracy": []},
                "test": {"accuracy": []},
            },
            "after": {
                "train": {"accuracy": []},
                "val": {"accuracy": []},
                "test": {"accuracy": []},
            },
        }
        stdout = Console(log_path=False, log_time=False)
        self.logger = Logger(
            stdout=stdout,
            enable_log=self.args.save_log,
            logfile_path=OUT_DIR
            / self.output_dir
            / f"{self.args.dataset}_log.html",
        )
        self.eval_results: Dict[int, Dict[str, str]] = {}
        self.train_progress_bar = track(
            range(self.args.global_epoch), "[bold green]Training...", console=stdout
        )

        self.logger.log("=" * 20)
        self.logger.log("Experiment Arguments:", dict(self.args._get_kwargs()))

        
            
    def train(self):
        """The LI training process"""
        avg_round_time = 0
        self. selected_clients = self.train_clients
        # initialize clients
        print("Initializing Clients")
        self.clients: List[LoopClient] = [LoopClient(
            model=MODELS[self.args.model](dataset=self.args.dataset).to(self.device),
            client_id=client,
            args=self.args,
            logger=self.logger,
            device=self.device,
        ) for client in self.selected_clients]
        print("All clients initialized")
        for E in self.train_progress_bar:
            self.current_epoch = E

            if (E + 1) % self.args.verbose_gap == 0:
                self.logger.log("-" * 26, f"TRAINING EPOCH: {E + 1}", "-" * 26)

            if (E + 1) % self.args.test_gap == 0:
                self.test()

            # self.selected_clients = self.client_sample_stream[E]
            begin = time.time()
            self.run_main()
            end = time.time()
            avg_round_time = (avg_round_time * (self.current_epoch) + (end - begin)) / (
                self.current_epoch + 1
            )

        self.logger.log(
            f"loop's average time taken by each global epoch: {int(avg_round_time // 60)} m {(avg_round_time % 60):.2f} s."
        )
        
        save_client_stats()
    
              
    def run_main(self):
        self.logger.log(f"Round {self.current_epoch} starts.")
        for i, client in enumerate(self.clients):
            self.logger.log(f"Client {client.client_id} starts.")
            delta, weight, self.client_stats[client.client_id][self.current_epoch] = client.train(self.args.local_epoch)
            if i < len(self.clients) - 1:
                transfer_weights_without_last_module(client.model, self.clients[i+1].model)
            elif i == len(self.clients) - 1 and self.current_epoch < self.args.global_epoch - 1:
                transfer_weights_without_last_module(client.model, self.clients[0].model)
            self.logger.log(f"Client {client.client_id} ends.")
        self.logger.log(f"Round {self.current_epoch} ends.")
        
    def save_client_stats(self):
        """This function is for saving each client's training info."""
        with open(
            OUT_DIR / self.output_dir / f"{self.args.dataset}_client_stats.pkl",
            "wb",
        ) as f:
            pickle.dump(self.client_stats, f)
        
    def test(self):
        """The function for testing LI method's output (a single global model or personalized client models)."""
        self.test_flag = True
        client_ids = set(self.val_clients + self.test_clients)
        split_sample_flag = False
        if client_ids:
            if (set(self.train_clients) != set(self.val_clients)) or (
                set(self.train_clients) != set(self.test_clients)
            ):
                results = {
                    "val_clients": {
                        "before": {
                            "train": {"loss": [], "correct": [], "size": []},
                            "val": {"loss": [], "correct": [], "size": []},
                            "test": {"loss": [], "correct": [], "size": []},
                        },
                        "after": {
                            "train": {"loss": [], "correct": [], "size": []},
                            "val": {"loss": [], "correct": [], "size": []},
                            "test": {"loss": [], "correct": [], "size": []},
                        },
                    },
                    "test_clients": {
                        "before": {
                            "train": {"loss": [], "correct": [], "size": []},
                            "val": {"loss": [], "correct": [], "size": []},
                            "test": {"loss": [], "correct": [], "size": []},
                        },
                        "after": {
                            "train": {"loss": [], "correct": [], "size": []},
                            "val": {"loss": [], "correct": [], "size": []},
                            "test": {"loss": [], "correct": [], "size": []},
                        },
                    },
                }
            else:
                split_sample_flag = True
                results = {
                    "all_clients": {
                        "before": {
                            "train": {"loss": [], "correct": [], "size": []},
                            "val": {"loss": [], "correct": [], "size": []},
                            "test": {"loss": [], "correct": [], "size": []},
                        },
                        "after": {
                            "train": {"loss": [], "correct": [], "size": []},
                            "val": {"loss": [], "correct": [], "size": []},
                            "test": {"loss": [], "correct": [], "size": []},
                        },
                    }
                }
            for cid in client_ids:
                client_local_params = self.generate_client_params(cid)
                stats = self.trainer.test(cid, client_local_params)

                for stage in ["before", "after"]:
                    for split in ["train", "val", "test"]:
                        for metric in ["loss", "correct", "size"]:
                            if split_sample_flag:
                                results["all_clients"][stage][split][metric].append(
                                    stats[stage][split][metric]
                                )
                            else:
                                if cid in self.val_clients:
                                    results["val_clients"][stage][split][metric].append(
                                        stats[stage][split][metric]
                                    )
                                if cid in self.test_clients:
                                    results["test_clients"][stage][split][
                                        metric
                                    ].append(stats[stage][split][metric])
            for group in results.keys():
                for stage in ["before", "after"]:
                    for split in ["train", "val", "test"]:
                        for metric in ["loss", "correct", "size"]:
                            results[group][stage][split][metric] = torch.tensor(
                                results[group][stage][split][metric]
                            )
                        num_samples = results[group][stage][split]["size"].sum()
                        if num_samples > 0:
                            results[group][stage][split]["accuracy"] = (
                                results[group][stage][split]["correct"].sum()
                                / num_samples
                                * 100
                            )
                            results[group][stage][split]["loss"] = (
                                results[group][stage][split]["loss"].sum() / num_samples
                            )
                        else:
                            results[group][stage][split]["accuracy"] = 0
                            results[group][stage][split]["loss"] = 0

            self.eval_results[self.current_epoch + 1] = results

        self.test_flag = False
    
    def generate_client_params(self, client_id: int) -> OrderedDict[str, torch.Tensor]:
        """
        This function is for outputting model parameters that asked by `client_id`.

        Args:
            client_id (int): The ID of query client.

        Returns:
            OrderedDict[str, torch.Tensor]: The trainable model parameters.
        """
        if self.unique_model:
            return OrderedDict(
                zip(self.trainable_params_name, self.client_trainable_params[client_id])
            )
        else:
            return self.global_params_dict

    def run(self):
        begin = time.time()
        
        if self.args.visible:
            self.viz.close(win=self.viz_win_name)
            
        self.train()
        end = time.time()
        total = end - begin
        self.logger.log(
            f"{self.algo}'s total running time: {int(total // 3600)} h {int((total % 3600) // 60)} m {int(total % 60)} s."
        )
        self.logger.log("=" * 20, self.algo, "Experiment Results:", "=" * 20)
        self.logger.log(
            "Format: [green](before local fine-tuning) -> [blue](after local fine-tuning)"
        )
        self.logger.log(
            {
                epoch: {
                    group: {
                        split: {
                            "loss": f"{metrics['before'][split]['loss']:.4f} -> {metrics['after'][split]['loss']:.4f}",
                            "accuracy": f"{metrics['before'][split]['accuracy']:.2f}% -> {metrics['after'][split]['accuracy']:.2f}%",
                        }
                        for split in ["train", "val", "test"]
                    }
                    for group, metrics in results.items()
                }
                for epoch, results in self.eval_results.items()
            }
        )
        

