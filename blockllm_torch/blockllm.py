from collections import defaultdict
import math
from typing import List, Dict, Tuple, Optional, Any, Iterator
import torch
from torch.optim.optimizer import Optimizer
from torch.optim import SparseAdam, Adam
from dataclasses import dataclass
from torch import Tensor
import random
import numpy as np

# Configure logger
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BlockLLMConfig:
    """Configuration for BlockLLM optimizer."""
    lr: float = 1e-3
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0
    sparsity_level: float = 0.9
    update_freq: int = 1000
    num_bottom_to_sample: int = 1
    patience: int = 100
    param_update_interval: int = 20


class BlockLLM(Optimizer):
    """Block-wise Sparse Optimizer for Large Language Models.
    
    This optimizer implements a dynamic parameter selection strategy where only a subset
    of parameters are updated at each step, based on their gradient magnitudes and
    visitation frequency.
    """
    
    def __init__(
        self,
        named_params: Iterator[Tuple[str, Tensor]],
        config: Optional[BlockLLMConfig] = None,
    ):
        """Initialize the BlockLLM optimizer.
        
        Args:
            named_params: Iterator of (name, parameter) tuples
            config: Optional configuration object. If None, uses default values
        """
        config = config or BlockLLMConfig()
        
        if not 0.0 <= config.sparsity_level < 1.0:
            raise ValueError(f"Invalid sparsity level: {config.sparsity_level}")
        if not 0.0 <= config.lr:
            raise ValueError(f"Invalid learning rate: {config.lr}")
        if not 0.0 <= config.eps:
            raise ValueError(f"Invalid epsilon value: {config.eps}")
        if not 0.0 <= config.betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {config.betas[0]}")
        if not 0.0 <= config.betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {config.betas[1]}")

        self.internal_optimizer = None  # will be set in reset_optimizer
        optim_groups = []
        self.loss = 0.0

        # disable gradients for embedding and lm_head layers
        visit_freq = defaultdict(int)

        for param_name, param in named_params:
            # sparse_hook = self.sparse_update_hook(param_name)
            # for param in params:
            if ("embed" in param_name) or ("lm_head" in param_name):
                param.requires_grad = False
                continue
            if not param.requires_grad:
                continue
            state = {}
            state["name"] = param_name
            state["params"] = param
            optim_groups.append(state)
            visit_freq[param_name] = 1

        defaults = dict(
            lr=config.lr,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay,
            sparsity_level=config.sparsity_level,
            update_freq=config.update_freq,
            top_k_param_names=[],
            bottom_k_param_names=None,
            mask={},
            grad_norm_dict={},
            num_bottom_to_sample=config.num_bottom_to_sample,
            param_visit_freq={},
            loss_history=[],
            patience=config.patience,
            param_update_interval=config.param_update_interval,
        )
        super(BlockLLM, self).__init__(optim_groups, defaults)
        # super(BlockLLM, self).__init__(defaults)

        self.sparsity_level = config.sparsity_level
        self.update_freq = config.update_freq
        self.total_params = sum(
            p.numel() for group in self.param_groups for p in group["params"]
        )
        self.update_step = 0
        self.num_params_to_keep = int(
            (1 - self.sparsity_level) * len(self.param_groups)
        )

        # normalize the visit_freq and add it to defaults
        visit_freq = {k: v / len(self.param_groups) for k, v in visit_freq.items()}
        self.defaults["param_visit_freq"] = visit_freq

        # Compute top_k_params based on the weight magnitude
        self._choose_top_k_params_by_mag()

        # Make the requires_grad False for the remaining parameters
        for group in self.param_groups:
            name = group["name"]
            for param in group["params"]:
                if name not in self.defaults["top_k_param_names"]:
                    param.requires_grad = False
                    param.grad = None

        self._update_bottom_k_params()

    def set_loss(self, loss_value: float) -> None:
        """Set the loss value."""
        self.loss = loss_value

    @torch.no_grad()
    def _choose_top_k_params_by_mag(self) -> List[str]:
        """Choose the top k parameters based on the weight magnitude.
        
        This function is intended to be used before the training starts.
        """
        mag_param_names_weighted = [
            (group["name"], param.data.abs().norm().item(), param)
            for group in self.param_groups
            for param in group["params"]
        ]
        random.shuffle(mag_param_names_weighted)
        sorted_param_names = [(x[0], x[2]) for x in mag_param_names_weighted]

        top_k_param_names = []
        top_k_params = []
        for name, param in sorted_param_names:
            if "embed" in name or "lm_head" in name:
                continue
            top_k_param_names.append(name)
            top_k_params.append(param)
            if len(top_k_params) >= self.num_params_to_keep:
                break

        self.defaults["top_k_param_names"] = top_k_param_names
        self.reset_optimizer(top_k_params, mode="dense")
        del (
            sorted_param_names,
            mag_param_names_weighted,
        )
        torch.cuda.empty_cache()

    def _update_bottom_k_params(self) -> None:
        """Create a cycle for the bottom k parameters."""
        bottom_k_params = [
            group["name"]
            for group in self.param_groups
            for param in group["params"]
            if group["name"] not in self.defaults["top_k_param_names"]
        ]
        self.defaults["bottom_k_param_names"] = bottom_k_params

    def reset_optimizer(self, parameters: List[Tensor], mode: str = "sparse", lr: Optional[float] = None) -> None:
        """Reset the internal optimizer."""
        if lr is None:
            lr = self.defaults["lr"]
        if mode == "sparse":
            self.internal_optimizer = SparseAdam(
                parameters, lr=lr, betas=self.defaults["betas"]
            )
        else:
            self.internal_optimizer = Adam(
                parameters, lr=lr, betas=self.defaults["betas"]
            )

    def train(self):
        return self

    def eval(self):
        return self

    @torch.no_grad()
    def step(self, closure: Optional[Any] = None) -> Optional[float]:
        """Perform a single optimization step."""
        loss = None
        if self.update_step == 0:
            self._adjust_parameters()
            self.update_step += 1
            return 0.0
       
        self.update_step += 1
        self.internal_optimizer.step(closure)

        # Sample bottom k parameters and make them trainable

        # Update the gradient norms of all the trainable parameters
        # in the grad_norm_dict
        for group in self.param_groups:
            for param in group["params"]:
                if not param.requires_grad:
                    continue
                name = group["name"]
                self.defaults["grad_norm_dict"][name] = param.grad.abs().norm().item()

                if self.update_step % self.defaults["param_update_interval"] == 0:
                    if name not in self.defaults["top_k_param_names"]:
                        param.requires_grad = False
                        param.grad = None

        # Make the bottom k parameters not trainable

        if self.update_step % self.defaults["param_update_interval"] == 0:
            for _ in range(self.defaults["num_bottom_to_sample"]):
                random_bottom_param = random.choice(
                    self.defaults["bottom_k_param_names"]
                )

                # Make the bottom k parameters trainable
                for group in self.param_groups:
                    if group["name"] == random_bottom_param:
                        for param in group["params"]:
                            param.requires_grad = True
            torch.cuda.empty_cache()

        self.defaults["loss_history"].append(self.loss)
        self._adjust_parameters()
        return loss

    def _should_adjust_parameters(self) -> bool:
        """Determine if parameters should be adjusted based on loss history."""
        if len(self.defaults["loss_history"]) <= self.defaults["patience"] or self.update_step % self.defaults["update_freq"] != 0:
            return False
            
        loss_history = self.defaults["loss_history"]
        moving_avg_loss = sum(loss_history) / len(loss_history) if loss_history else 0
        recent_avg_loss = sum(loss_history[-self.defaults["patience"]:]) / self.defaults["patience"]
        
        return moving_avg_loss < recent_avg_loss

    def _adjust_parameters(self) -> None:
        """Find the top k parameters based on the gradient norms in the grad_norm_dict
        Update the bottom k parameters iterable
        """
        # If the loss history has more than "patience" elements, check
        # if the moving average of the loss_history is decreasing
        # If not, reset the optimizer and adjust the parameters

        if self._should_adjust_parameters():
            if self.internal_optimizer:
                curr_lr = self.internal_optimizer.param_groups[0]["lr"]
            else:
                curr_lr = None
            self.internal_optimizer = None

            # Find the top k parameters based on the gradient norms
            sorted_grad_norms = [
                name
                for name, _ in sorted(
                    self.defaults["grad_norm_dict"].items(),
                    key=lambda x: x[1]
                    / np.exp(
                        self.defaults["param_visit_freq"][x[0]] / self.update_step
                    ),
                    reverse=True,
                )
            ]

            logger.info(
                f"Updating parameters at step {self.update_step}. "
                f"Resetting optimizer with learning rate {curr_lr}. "
            )

            name_to_param = {
                group["name"]: group["params"] for group in self.param_groups
            }

            top_k_param_names = []
            top_k_params = []
            running_param_count = 0
            num_params_to_keep = int((1 - self.sparsity_level) * self.total_params)

            for name in sorted_grad_norms:
                if "embed" in name or "lm_head" in name:
                    continue
                running_param_count += name_to_param[name][0].numel()
                if running_param_count >= num_params_to_keep:
                    break
                top_k_param_names.append(name)
                top_k_params.extend(name_to_param[name])

            # If we don't have enough parameters to keep, choose the top k parameters randomly
            while running_param_count < num_params_to_keep:
                random_param = random.choice(list(name_to_param.keys()))
                if random_param in top_k_param_names:
                    continue
                if "embed" in random_param or "lm_head" in random_param:
                    continue
                top_k_param_names.append(random_param)
                top_k_params.extend(name_to_param[random_param])
                running_param_count += name_to_param[random_param][0].numel()

            self.defaults["top_k_param_names"] = top_k_param_names

            # Update the top k parameters
            self.reset_optimizer(top_k_params, mode="dense", lr=curr_lr)

            # Make the requires_grad False for the remaining parameters
            for group in self.param_groups:
                name = group["name"]
                for param in group["params"]:
                    if name not in top_k_param_names:
                        param.requires_grad = False
                        param.grad = None
                    else:
                        param.requires_grad = True
                        # update the visit frequency
                        self._update_param_visit_frequency(name)

            # re-normalize the visit frequency
            total_visits = sum(self.defaults["param_visit_freq"].values())
            self.defaults["param_visit_freq"] = {
                k: v / total_visits
                for k, v in self.defaults["param_visit_freq"].items()
            }

            # clear the gradient norms
            # self.defaults["grad_norm_dict"] = {}

            # clear loss history
            self.defaults["loss_history"] = []

            # Update the bottom k parameters iterable
            self._update_bottom_k_params()
            del (
                top_k_params,
                sorted_grad_norms,
                top_k_param_names,
                name_to_param,
            )
            torch.cuda.empty_cache()

    def _update_param_visit_frequency(self, param_name: str) -> None:
        """Update and normalize the visit frequency for a parameter."""
        self.defaults["param_visit_freq"][param_name] += 1
        total_visits = sum(self.defaults["param_visit_freq"].values())
        self.defaults["param_visit_freq"] = {
            k: v / total_visits for k, v in self.defaults["param_visit_freq"].items()
        }
